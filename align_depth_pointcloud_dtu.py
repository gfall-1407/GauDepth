"""Align GauDepth per-view depth point clouds to the DTU coordinate system.

The point clouds produced by ``render_depth_pointcloud.py`` are in the
coordinate system of the trained COLMAP scene.  This script estimates the
same similarity transform used by ``evaluate_dtu_mesh.py`` from the training
camera centers and DTU camera centers, then applies it to every per-view PLY.

By default, the input and output directories are:

    <model_path>/depth_pointcloud/iteration_<loaded_iter>/point_clouds/
    <model_path>/depth_pointcloud/iteration_<loaded_iter>/point_clouds_dtu/

The source PLY files are never overwritten.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def best_fit_transform(source, target):
    """Return the rigid transform that maps corresponding source to target.

    Points are treated as column vectors internally.  For row-wise point
    arrays, the equivalent operation is ``points @ rotation.T + translation``.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must both have shape (N, 3), got "
            f"{source.shape} and {target.shape}"
        )
    if source.shape[0] < 3:
        raise ValueError("At least three camera correspondences are required")

    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = centroid_target - rotation @ centroid_source
    return rotation, translation


def load_dtu_camera_centers(dtu_path, camera_count=64):
    """Read DTU camera centers from Calibration/cal18/pos_XXX.txt."""
    dtu_path = Path(dtu_path)
    centers = []
    for index in range(1, camera_count + 1):
        camera_file = dtu_path / "Calibration" / "cal18" / f"pos_{index:03d}.txt"
        if not camera_file.is_file():
            raise FileNotFoundError(f"DTU camera file not found: {camera_file}")

        projection = np.loadtxt(camera_file, dtype=np.float32)
        _, rotation, homogeneous_center = cv2.decomposeProjectionMatrix(projection)[:3]

        # This is the same decomposition used in evaluate_dtu_mesh.py.
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation.transpose()
        pose[:3, 3] = (homogeneous_center[:3] / homogeneous_center[3])[:, 0]
        centers.append(pose[:3, 3])

    return np.asarray(centers, dtype=np.float64)


def load_model_camera_centers(dataset, iteration):
    """Load the trained scene and return train-camera centers and metadata."""
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()

    centers = []
    image_names = []
    for camera in train_cameras:
        camera_to_world = (camera.world_view_transform.T).inverse()
        centers.append(camera_to_world[:3, 3].detach().cpu().numpy())
        image_names.append(str(camera.image_name))

    if not centers:
        raise RuntimeError("The trained scene contains no training cameras")
    return scene, np.asarray(centers, dtype=np.float64), image_names


def estimate_model_to_dtu_transform(model_centers, dtu_centers):
    """Estimate the scale, rotation and translation used by DTU evaluation."""
    if model_centers.shape[0] > dtu_centers.shape[0]:
        raise ValueError(
            f"The model has {model_centers.shape[0]} train cameras, but only "
            f"{dtu_centers.shape[0]} DTU cameras were loaded"
        )

    # Match evaluate_dtu_mesh.py: use the first N DTU poses in order.
    dtu_centers = dtu_centers[: model_centers.shape[0]]
    model_radius = np.linalg.norm(
        model_centers - model_centers.mean(axis=0), axis=1
    ).mean()
    dtu_radius = np.linalg.norm(
        dtu_centers - dtu_centers.mean(axis=0), axis=1
    ).mean()
    if not np.isfinite(model_radius) or model_radius <= 1e-12:
        raise ValueError(f"Invalid model camera-center scale: {model_radius}")
    if not np.isfinite(dtu_radius) or dtu_radius <= 1e-12:
        raise ValueError(f"Invalid DTU camera-center scale: {dtu_radius}")

    scale = dtu_radius / model_radius
    scaled_model_centers = model_centers * scale
    rotation, translation = best_fit_transform(scaled_model_centers, dtu_centers)
    aligned_centers = scaled_model_centers @ rotation.T + translation
    residuals = np.linalg.norm(aligned_centers - dtu_centers, axis=1)

    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "camera_rmse": float(np.sqrt(np.mean(residuals**2))),
        "camera_max_error": float(np.max(residuals)),
        "model_camera_centers": model_centers,
        "dtu_camera_centers": dtu_centers,
        "aligned_model_camera_centers": aligned_centers,
    }


def align_point_cloud(point_cloud_path, output_path, scale, rotation, translation):
    """Apply the model-to-DTU similarity transform to one PLY point cloud."""
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        print(f"[warning] empty point cloud, skipped: {point_cloud_path}")
        return 0

    aligned_points = (points.astype(np.float64) * scale) @ rotation.T + translation
    point_cloud.points = o3d.utility.Vector3dVector(aligned_points)

    # Uniform scale does not change normal directions; rotate them with R.
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        point_cloud.normals = o3d.utility.Vector3dVector(normals @ rotation.T)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write aligned point cloud: {output_path}")
    return len(points)


def _to_serializable_array(array):
    return np.asarray(array).tolist()


def align_depth_pointclouds(dataset, iteration, dtu_path, input_dir=None, output_dir=None, depth_types=None):
    scene, model_centers, image_names = load_model_camera_centers(dataset, iteration)
    dtu_centers = load_dtu_camera_centers(dtu_path)
    transform = estimate_model_to_dtu_transform(model_centers, dtu_centers)

    if input_dir is None:
        input_root = (
            Path(dataset.model_path)
            / "depth_pointcloud"
            / f"iteration_{scene.loaded_iter}"
            / "point_clouds"
        )
    else:
        input_root = Path(input_dir)

    if output_dir is None:
        output_root = input_root.parent / "point_clouds_dtu"
    else:
        output_root = Path(output_dir)

    if depth_types is None:
        depth_types = ["mean", "median"]

    scale = transform["scale"]
    rotation = transform["rotation"]
    translation = transform["translation"]
    dtu_centers_used = transform["dtu_camera_centers"]

    print(f"Loaded iteration: {scene.loaded_iter}")
    print(f"Input point-cloud directory: {input_root}")
    print(f"Output point-cloud directory: {output_root}")
    print(f"Model-to-DTU scale: {scale:.9g}")
    print(f"Camera-center RMSE: {transform['camera_rmse']:.6g}")
    print(f"Camera-center max error: {transform['camera_max_error']:.6g}")
    print(
        "[warning] Camera matching follows evaluate_dtu_mesh.py: "
        "train-camera order <-> DTU pos_001, pos_002, ..."
    )

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_root}\n"
            "Run render_depth_pointcloud.py first or pass --input_dir."
        )

    processed = 0
    skipped_empty = 0
    for depth_type in depth_types:
        source_dir = input_root / depth_type
        if not source_dir.is_dir():
            print(f"[warning] depth-type directory not found, skipped: {source_dir}")
            continue

        files = sorted(source_dir.glob("*.ply"))
        if not files:
            print(f"[warning] no PLY files found: {source_dir}")
            continue

        for source_path in files:
            relative_path = source_path.relative_to(source_dir)
            # Keep the source name recognizable while making the coordinate
            # conversion explicit in the output filename.
            destination_name = f"{relative_path.stem}_dtu{relative_path.suffix}"
            destination_path = output_root / depth_type / relative_path.with_name(destination_name)
            count = align_point_cloud(
                source_path,
                destination_path,
                scale,
                rotation,
                translation,
            )
            if count == 0:
                skipped_empty += 1
            else:
                processed += 1

    homogeneous_transform = np.eye(4, dtype=np.float64)
    homogeneous_transform[:3, :3] = scale * rotation
    homogeneous_transform[:3, 3] = translation

    metadata = {
        "source_coordinate_system": "GauDepth/COLMAP scene world",
        "target_coordinate_system": "DTU official coordinate system",
        "iteration": int(scene.loaded_iter),
        "input_dir": str(input_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "depth_types": list(depth_types),
        "num_train_cameras": int(model_centers.shape[0]),
        "num_dtu_cameras_used": int(dtu_centers_used.shape[0]),
        "train_camera_image_names": image_names,
        "camera_matching": "train camera order matched to DTU pos_001 ... pos_N",
        "scale_model_to_dtu": scale,
        "rotation_model_to_dtu": _to_serializable_array(rotation),
        "translation_model_to_dtu": _to_serializable_array(translation),
        "homogeneous_column_vector_transform": _to_serializable_array(
            homogeneous_transform
        ),
        "row_point_formula": "P_dtu = (P_model * scale) @ rotation.T + translation",
        "camera_center_rmse": transform["camera_rmse"],
        "camera_center_max_error": transform["camera_max_error"],
        "model_camera_centers": _to_serializable_array(model_centers),
        "dtu_camera_centers_used": _to_serializable_array(dtu_centers_used),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "model_to_dtu.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    np.savetxt(
        output_root / "model_to_dtu.txt",
        homogeneous_transform,
        fmt="%.12g",
        header=(
            "Column-vector transform: p_dtu = H @ [p_model, 1]. "
            "For row-wise points use P_dtu = P_model_h @ H.T."
        ),
    )

    print(f"Aligned point-cloud files: {processed}")
    print(f"Empty point-cloud files skipped: {skipped_empty}")
    print(f"Saved transform: {output_root / 'model_to_dtu.json'}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Transform render_depth_pointcloud.py PLY files to DTU coordinates"
    )
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--DTU",
        type=str,
        default="dtu_eval/Offical_DTU_Dataset",
        help="DTU dataset root containing Calibration/cal18/pos_XXX.txt",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Override the point_clouds directory generated by render_depth_pointcloud.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory; defaults to the sibling point_clouds_dtu directory",
    )
    parser.add_argument(
        "--depth_types",
        nargs="+",
        choices=("mean", "median"),
        default=("mean", "median"),
        help="Depth variants to transform",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Aligning GauDepth depth point clouds for " + args.model_path)
    safe_state(args.quiet)
    torch.cuda.set_device(torch.device("cuda:0"))
    with torch.no_grad():
        align_depth_pointclouds(
            model.extract(args),
            args.iteration,
            args.DTU,
            getattr(args, "input_dir", None),
            getattr(args, "output_dir", None),
            getattr(args, "depth_types", None),
        )
