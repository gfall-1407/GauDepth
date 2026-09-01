"""Render train-view depths and export per-view camera-space point clouds.

This follows the scene loading path used by ``mesh_extract.py``.  The only
difference is that the rendered depths are saved directly instead of being
integrated into a TSDF volume.

The rasterizer's ``expected_depth`` and ``median_depth`` are z-depths in the
camera coordinate system.  Therefore a pixel ``(u, v)`` is back-projected as

    x = (u - Cx) / Fx * depth
    y = (v - Cy) / Fy * depth
    z = depth

No world-space transformation is applied to the exported points.
"""

import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


DEPTH_TYPES = {
    "mean": "expected_depth",
    "median": "median_depth",
}


def _safe_view_stem(view_index, image_name):
    """Create a stable filename while retaining the training-view identity."""
    image_name = os.path.basename(str(image_name))
    image_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", image_name).strip("._")
    if not image_name:
        image_name = "view"
    return f"{view_index:05d}_{image_name}"


def _depth_visualization(depth):
    """Return an RGB uint8 visualization; raw depth values are not changed."""
    valid = np.isfinite(depth) & (depth > 0)
    visualization = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return visualization

    values = depth[valid]
    lower, upper = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(values.min())
        upper = float(values.max())
    normalized = np.zeros_like(depth, dtype=np.float32)
    if upper > lower:
        normalized[valid] = np.clip(
            (depth[valid] - lower) / (upper - lower), 0.0, 1.0
        )

    # A compact blue-cyan-yellow-red map, avoiding a matplotlib runtime
    # dependency for this export script.
    stops = np.array(
        [
            [0, 0, 128],
            [0, 192, 255],
            [255, 255, 0],
            [192, 0, 0],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.minimum(left + 1, len(stops) - 1)
    fraction = (scaled - left)[..., None]
    colored = stops[left] * (1.0 - fraction) + stops[right] * fraction
    visualization[valid] = np.clip(colored[valid], 0, 255).astype(np.uint8)
    return visualization


def _save_depth(depth, raw_path, visualization_path):
    """Save lossless raw float32 depth and a human-viewable PNG."""
    np.save(raw_path, depth.astype(np.float32, copy=False))
    Image.fromarray(_depth_visualization(depth), mode="RGB").save(visualization_path)


def _depth_to_camera_points(depth, viewpoint_cam):
    """Back-project a z-depth image to points in the current camera frame."""
    height, width = depth.shape
    pixel_x = (np.arange(width, dtype=np.float32) - float(viewpoint_cam.Cx)) / float(viewpoint_cam.Fx)
    pixel_y = (np.arange(height, dtype=np.float32) - float(viewpoint_cam.Cy)) / float(viewpoint_cam.Fy)
    ray_x, ray_y = np.meshgrid(pixel_x, pixel_y, indexing="xy")
    return np.stack(
        [ray_x * depth, ray_y * depth, depth],
        axis=-1,
    ).reshape(-1, 3)


def _save_point_cloud(points, colors, valid_mask, point_cloud_path):
    """Save valid camera-space points as a colored PLY file."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[valid_mask].astype(np.float64, copy=False))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors[valid_mask].astype(np.float64, copy=False))
    success = o3d.io.write_point_cloud(str(point_cloud_path), point_cloud, write_ascii=False)
    if not success:
        raise RuntimeError(f"Failed to write point cloud: {point_cloud_path}")
    return len(point_cloud.points)


def render_depths_and_pointclouds(dataset, pipe, iteration, output_dir=None):
    """Load a trained scene and export both depth variants for all train views."""
    gaussians = GaussianModel(dataset.sh_degree)
    # Keep this loading sequence aligned with mesh_extract.py.
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    viewpoint_cam_list = scene.getTrainCameras()

    if output_dir is None:
        output_root = Path(dataset.model_path) / "depth_pointcloud" / f"iteration_{scene.loaded_iter}"
    else:
        output_root = Path(output_dir)
    depth_root = output_root / "depth_maps"
    visualization_root = output_root / "depth_visualization"
    point_cloud_root = output_root / "point_clouds"
    for depth_type in DEPTH_TYPES:
        (depth_root / depth_type).mkdir(parents=True, exist_ok=True)
        (visualization_root / depth_type).mkdir(parents=True, exist_ok=True)
        (point_cloud_root / depth_type).mkdir(parents=True, exist_ok=True)

    # mesh_extract.py uses a white background and the dataset kernel size.
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    view_metadata = []

    for view_index, viewpoint_cam in enumerate(
        tqdm(viewpoint_cam_list, desc="Rendering depth and point clouds")
    ):
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        rendered_color = (
            torch.clamp(render_pkg["render"], min=0.0, max=1.0)
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Match mesh_extract.py's optional ground-truth alpha-mask handling.
        gt_mask = viewpoint_cam.gt_mask
        for depth_type, render_key in DEPTH_TYPES.items():
            depth_tensor = render_pkg[render_key].clone()
            if gt_mask is not None:
                depth_tensor[gt_mask < 0.5] = 0
            depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)

            valid_mask = np.isfinite(depth) & (depth > 0.0)
            points = _depth_to_camera_points(depth, viewpoint_cam)
            colors = np.clip(rendered_color.reshape(-1, 3), 0.0, 1.0)
            stem = _safe_view_stem(view_index, viewpoint_cam.image_name)
            file_tag = f"{depth_type}_depth"

            raw_path = depth_root / depth_type / f"{stem}_{file_tag}.npy"
            visualization_path = visualization_root / depth_type / f"{stem}_{file_tag}_visualization.png"
            point_cloud_path = point_cloud_root / depth_type / f"{stem}_{file_tag}_point_cloud.ply"
            _save_depth(depth, raw_path, visualization_path)
            point_count = _save_point_cloud(points, colors, valid_mask.reshape(-1), point_cloud_path)

            if depth_type == "mean":
                view_metadata.append(
                    {
                        "index": view_index,
                        "image_name": viewpoint_cam.image_name,
                        "depth_files": {},
                        "point_cloud_files": {},
                        "width": int(viewpoint_cam.image_width),
                        "height": int(viewpoint_cam.image_height),
                        "fx": float(viewpoint_cam.Fx),
                        "fy": float(viewpoint_cam.Fy),
                        "cx": float(viewpoint_cam.Cx),
                        "cy": float(viewpoint_cam.Cy),
                    }
                )
            view_metadata[-1]["depth_files"][depth_type] = str(raw_path.relative_to(output_root))
            view_metadata[-1]["point_cloud_files"][depth_type] = str(point_cloud_path.relative_to(output_root))
            view_metadata[-1].setdefault("point_count", {})[depth_type] = point_count

    metadata = {
        "iteration": scene.loaded_iter,
        "num_train_views": len(viewpoint_cam_list),
        "coordinate_system": "camera",
        "depth_definition": "z-depth; x=(u-cx)/fx*z, y=(v-cy)/fy*z, z=depth",
        "depth_types": {
            "mean": "expected_depth",
            "median": "median_depth",
        },
        "views": view_metadata,
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    torch.cuda.empty_cache()
    print(f"Saved {len(viewpoint_cam_list)} train views to {output_root}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render train-view depths and export camera-space point clouds")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory; defaults to <model_path>/depth_pointcloud/iteration_<loaded_iter>",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering depth maps and point clouds for " + args.model_path)
    safe_state(args.quiet)
    with torch.no_grad():
        render_depths_and_pointclouds(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            getattr(args, "output_dir", None),
        )
