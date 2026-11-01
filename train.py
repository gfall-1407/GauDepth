#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import os
from random import randint, sample
import random
import sys
from typing import Sequence, TypedDict
import uuid

import torch
import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.graphics_utils import (
    depth_double_to_normal,
    depth_to_normal,
)
from utils.image_utils import psnr
from utils.loss_utils import (
    l1_loss,
    ssim,
    L1_loss_appearance,
    PatchMatch,
)
from training_depth_plugin import (ABLog, add_step_metrics, file_digest,
                                   parameter_digest, parameter_metrics, route_depth)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    depth_route='baseline',
    depth_target='expected',
    ab_run_id=None,
    ab_seed=0,
    ab_log_interval=100,
    warm_start_ply=None,
    warm_start_iteration=None,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if warm_start_ply:
        gaussians.load_ply(warm_start_ply)
        first_iter = warm_start_iteration
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if ab_run_id and first_iter < opt.regularization_from_iter:
        raise ValueError('A/B short run must start at or after regularization_from_iter')
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    ab_rng = None
    ab_log = None
    if ab_run_id:
        # The camera draws are isolated from candidate code's global RNG use.
        ab_rng = random.Random(ab_seed)
        random.seed(ab_seed)
        np.random.seed(ab_seed)
        torch.manual_seed(ab_seed)
        torch.cuda.manual_seed_all(ab_seed)
        ab_log = ABLog(dataset.model_path, ab_run_id, ab_log_interval)
        ab_log.write(dict(event='start', run_id=ab_run_id, seed=ab_seed,
                          initial_iteration=first_iter, parameter_sha256=parameter_digest(gaussians),
                          warm_start_ply_sha256=file_digest(warm_start_ply) if warm_start_ply else None,
                          checkpoint_sha256=file_digest(checkpoint) if checkpoint else None,
                          optimizer_state='fresh_from_ply' if warm_start_ply else 'restored_checkpoint',
                          depth_route=depth_route, depth_target=depth_target,
                          depth_ratio=dataset.depth_ratio,
                          regularization_from_iter=opt.regularization_from_iter,
                          lambda_depth_normal=opt.lambda_depth_normal,
                          lambda_multi_view_ncc=opt.lambda_multi_view_ncc,
                          lambda_multi_view_geo=opt.lambda_multi_view_geo))

    if opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0:
        patchmatch = PatchMatch(
            opt.multi_view_patch_size,
            opt.multi_view_pixel_noise_th,
            kernel_size=kernel_size,
            pipe=pipe,
            debug=True,
            model_path=dataset.model_path,
        )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_ncc_loss_for_log = 0.0
    os.makedirs(os.path.join(dataset.model_path, "debug"), exist_ok=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        kernel_size,
                        scaling_modifer,
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1) if ab_rng is None else ab_rng.randrange(len(viewpoint_stack)))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size,
            require_depth=reg_kick_on,
        )
        render_pkg = route_depth(render_pkg, depth_route, depth_target)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        # normal consistency
        if reg_kick_on and opt.lambda_depth_normal > 0:
            rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
            rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
            rendered_normal: torch.Tensor = render_pkg["normal"]
            if 0.0 < dataset.depth_ratio < 1.0:
                depth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal.unsqueeze(0), depth_normal, dim=1)
                depth_normal_loss = (1 - dataset.depth_ratio) * normal_error_map[0].mean() + dataset.depth_ratio * normal_error_map[1].mean()
                depth_normal = None
            else:
                depth_map = rendered_expected_depth if dataset.depth_ratio < 1.0 else rendered_median_depth
                depth_normal = depth_to_normal(viewpoint_cam, depth_map)
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
                depth_normal_loss = normal_error_map.mean()
        else:
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # patch match loss
        if reg_kick_on and opt.lambda_multi_view_ncc > 0:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[
                sample(viewpoint_cam.nearest_id, 1)[0] if ab_rng is None else ab_rng.choice(viewpoint_cam.nearest_id)]
            ncc_loss, geo_loss = patchmatch(gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0)))

        loss = rgb_loss + opt.lambda_depth_normal * depth_normal_loss + opt.lambda_multi_view_ncc * ncc_loss + opt.lambda_multi_view_geo * geo_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_ncc_loss_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "loss_normal": f"{ema_normal_loss_for_log:.{4}f}",
                        "loss_ncc": f"{ema_ncc_loss_for_log:.{4}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1_render,
                loss,
                depth_normal_loss,
                ncc_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, kernel_size),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.05,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Opt-in A/B trace: all camera choices; bounded parameter samples.
            log_metrics = ab_log is not None and iteration % ab_log.interval == 0
            metrics, before = parameter_metrics(gaussians.optimizer) if log_metrics else (None, None)
            step_applied = iteration < opt.iterations
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if ab_log is not None:
                ab_log.write(dict(event='iteration', iteration=iteration,
                                  camera=viewpoint_cam.image_name,
                                  nearest_camera=nearest_cam.image_name if reg_kick_on and opt.lambda_multi_view_ncc > 0 and nearest_cam is not None else None,
                                  regularization_active=reg_kick_on,
                                  loss=float(loss.item()), rgb_loss=float(rgb_loss.item()),
                                  depth_normal_loss=float(depth_normal_loss.item()),
                                  ncc_loss=float(ncc_loss.item()), geo_loss=float(geo_loss.item()),
                                  step_applied=step_applied,
                                  parameters=add_step_metrics(metrics, before, gaussians.optimizer, step_applied)
                                  if log_metrics else None))

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    if ab_log is not None:
        ab_log.close()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    normal_loss,
    ncc_loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ncc_loss", ncc_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    class ValidationConfig(TypedDict):
        name: str
        cameras: Sequence[Camera]

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs: tuple[ValidationConfig, ...] = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print(
                            "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test),
                            file=file_object,
                        )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--depth_route', choices=['baseline', 'identity', 'candidate'], default='baseline')
    parser.add_argument('--depth_target', choices=['expected', 'median', 'both'], default='expected')
    parser.add_argument('--ab_run_id', type=str, default=None)
    parser.add_argument('--ab_seed', type=int, default=0)
    parser.add_argument('--ab_log_interval', type=int, default=100)
    parser.add_argument('--warm_start_ply', type=str, default=None)
    parser.add_argument('--warm_start_iteration', type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.depth_route == 'candidate':
        parser.error('candidate readout is not implemented; baseline and identity only')
    if args.warm_start_ply and args.start_checkpoint:
        parser.error('warm_start_ply and start_checkpoint are mutually exclusive')
    if bool(args.warm_start_ply) != (args.warm_start_iteration is not None):
        parser.error('warm_start_ply requires warm_start_iteration, and vice versa')
    if args.warm_start_iteration is not None and not 0 <= args.warm_start_iteration < args.iterations:
        parser.error('warm_start_iteration must precede iterations')
    if args.ab_run_id and not (args.start_checkpoint or args.warm_start_ply):
        parser.error('A/B trace requires the same checkpoint or PLY warm start in both arms')
    if args.ab_run_id and args.ab_log_interval < 1:
        parser.error('ab_log_interval must be positive')
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        depth_route=args.depth_route,
        depth_target=args.depth_target,
        ab_run_id=args.ab_run_id,
        ab_seed=args.ab_seed,
        ab_log_interval=args.ab_log_interval,
        warm_start_ply=args.warm_start_ply,
        warm_start_iteration=args.warm_start_iteration,
    )

    # All done
    print("\nTraining complete.")
