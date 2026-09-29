from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_io import camera_arrays, load_json, save_json


def read_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    body_points, pixels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            body_points.append([float(row[axis]) for axis in ("x_mm", "y_mm", "z_mm")])
            pixels.append([float(row[axis]) for axis in ("u_px", "v_px")])
    if len(body_points) < 6:
        raise ValueError("at least 6 correspondences are required; use 12 or more across the image")
    return np.asarray(body_points, np.float64), np.asarray(pixels, np.float64)


def estimate_mount(body_points, pixels, camera_matrix, distortion):
    ok, rvec_cb, tvec_cb, inliers = cv2.solvePnPRansac(
        body_points, pixels, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=300,
        reprojectionError=3.0,
        confidence=0.999,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError("mount pose fit failed or produced fewer than 6 inliers")
    inlier_indices = inliers.reshape(-1)
    ok, rvec_cb, tvec_cb = cv2.solvePnP(
        body_points[inlier_indices], pixels[inlier_indices], camera_matrix, distortion,
        rvec_cb, tvec_cb, True, cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("mount pose refinement failed")
    rotation_cb, _ = cv2.Rodrigues(rvec_cb)
    rotation_bc = rotation_cb.T
    camera_position_body = -rotation_bc @ tvec_cb
    projected, _ = cv2.projectPoints(body_points, rvec_cb, tvec_cb, camera_matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    return rotation_bc, camera_position_body.reshape(3), inlier_indices, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit camera-to-body pose from surveyed 3D points and pixels.")
    parser.add_argument("--intrinsics", type=Path, default=Path("calibration/results/lingyan/intrinsics.json"))
    parser.add_argument("--points", type=Path, default=Path("calibration/data/lingyan/mount_points.csv"))
    parser.add_argument("--output", type=Path, default=Path("calibration/results/lingyan/mount_pose.json"))
    args = parser.parse_args()

    intrinsics = load_json(args.intrinsics)
    camera_matrix, distortion = camera_arrays(intrinsics)
    body_points, pixels = read_correspondences(args.points)
    rotation_bc, position_b, inliers, errors = estimate_mount(
        body_points, pixels, camera_matrix, distortion,
    )
    result = {
        "schema": "lingyan.camera_mount.v1",
        "convention": {
            "body_axes": "X forward, Y left, Z up",
            "camera_axes": "X image-right, Y image-down, Z optical-forward",
            "rotation": "d_body = R_bc * d_camera",
            "units": "millimetres",
        },
        "intrinsics_file": str(args.intrinsics),
        "rotation_camera_to_body": rotation_bc.tolist(),
        "camera_position_body_mm": position_b.tolist(),
        "point_count": int(len(body_points)),
        "inlier_indices_zero_based": inliers.tolist(),
        "inlier_rmse_px": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "all_point_max_error_px": float(np.max(errors)),
    }
    save_json(args.output, result)
    print(f"saved {args.output}")
    print(f"C_b mm = {position_b}")
    print(f"inliers={len(inliers)}/{len(body_points)}, inlier RMSE={result['inlier_rmse_px']:.3f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
