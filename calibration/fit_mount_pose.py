from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_io import camera_arrays, save_result_json, validate_point_sources
from calibration.lens_models import project_points, solve_pose, solve_pose_ransac
from camera_profiles import file_sha256, isolated_path, load_intrinsics, load_profiles


def read_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    body_points, pixels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            body_points.append([float(row[axis]) for axis in ("x_mm", "y_mm", "z_mm")])
            pixels.append([float(row[axis]) for axis in ("u_px", "v_px")])
    if len(body_points) < 6:
        raise ValueError("at least 6 correspondences are required; use 12 or more across the image")
    return np.asarray(body_points, np.float64), np.asarray(pixels, np.float64)


def validate_correspondences(body_points, pixels) -> None:
    body_points = np.asarray(body_points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if body_points.ndim != 2 or body_points.shape[1] != 3 or len(body_points) < 6:
        raise ValueError("at least 6 surveyed XYZ points are required; use 12 or more")
    if pixels.shape != (len(body_points), 2):
        raise ValueError("every surveyed point needs one corresponding raw (u, v) pixel")
    if not np.isfinite(body_points).all() or not np.isfinite(pixels).all():
        raise ValueError("point coordinates and pixels must be finite numbers")
    if len(np.unique(body_points, axis=0)) != len(body_points):
        raise ValueError("duplicate surveyed points; use distinct physical locations")
    if len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("duplicate pixels (possibly template 0,0 placeholders); collect real image points")
    if np.linalg.matrix_rank(body_points - body_points.mean(axis=0)) < 2:
        raise ValueError("surveyed points must not all lie on one line")
    if np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2:
        raise ValueError("image points must not all lie on one line")


def estimate_mount(body_points, pixels, camera_matrix, distortion, model="opencv_pinhole"):
    validate_correspondences(body_points, pixels)
    ok, rvec_cb, tvec_cb, inliers = solve_pose_ransac(
        body_points, pixels, camera_matrix, distortion, model)
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError("mount pose fit failed or produced fewer than 6 inliers")
    inlier_indices = inliers.reshape(-1)
    validate_correspondences(body_points[inlier_indices], pixels[inlier_indices])
    ok, rvec_cb, tvec_cb = solve_pose(
        body_points[inlier_indices], pixels[inlier_indices], camera_matrix, distortion,
        model, rvec=rvec_cb, tvec=tvec_cb,
    )
    if not ok:
        raise RuntimeError("mount pose refinement failed")
    rotation_cb, _ = cv2.Rodrigues(rvec_cb)
    camera_points = (rotation_cb @ body_points[inlier_indices].T + tvec_cb).T
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        raise RuntimeError("invalid mounting solution: inlier points are not in front of the camera")
    rotation_bc = rotation_cb.T
    camera_position_body = -rotation_bc @ tvec_cb
    projected = project_points(body_points, rvec_cb, tvec_cb, camera_matrix, distortion, model)
    errors = np.linalg.norm(projected - pixels, axis=1)
    return rotation_bc, camera_position_body.reshape(3), inlier_indices, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit camera-to-body pose from surveyed 3D points and pixels.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    intrinsics_path, intrinsics = load_intrinsics(args.camera)
    points_path = isolated_path(args.points, args.camera, result=False, default_name="mount_points.csv")
    output = isolated_path(args.output, args.camera, result=True, default_name="mount_pose.json")
    camera_matrix, distortion = camera_arrays(intrinsics)
    body_points, pixels = read_correspondences(points_path)
    validate_point_sources(points_path, args.camera)
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= intrinsics["image_width"]) or np.any(pixels[:, 1] >= intrinsics["image_height"]):
        raise ValueError("pixels fall outside the calibrated raw image; do not use scaled preview coordinates")
    rotation_bc, position_b, inliers, errors = estimate_mount(
        body_points, pixels, camera_matrix, distortion, intrinsics["model"],
    )
    result = {
        "schema": "camera.mount.v2",
        "camera_id": args.camera,
        "model": intrinsics["model"],
        "convention": {
            "body_axes": "X forward, Y left, Z up",
            "camera_axes": "X image-right, Y image-down, Z optical-forward",
            "rotation": "d_body = R_bc * d_camera",
            "units": "millimetres",
        },
        "intrinsics_file": str(intrinsics_path),
        "intrinsics_sha256": file_sha256(intrinsics_path),
        "rotation_camera_to_body": rotation_bc.tolist(),
        "camera_position_body_mm": position_b.tolist(),
        "point_count": int(len(body_points)),
        "inlier_indices_zero_based": inliers.tolist(),
        "inlier_rmse_px": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "all_point_max_error_px": float(np.max(errors)),
    }
    backup = save_result_json(output, result)
    print(f"saved {output}")
    if backup:
        print(f"previous mount pose backed up to {backup}")
    print(f"C_b mm = {position_b}")
    print(f"inliers={len(inliers)}/{len(body_points)}, inlier RMSE={result['inlier_rmse_px']:.3f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
