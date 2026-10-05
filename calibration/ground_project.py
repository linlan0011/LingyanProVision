from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from calibration.calibration_io import camera_arrays
from calibration.lens_models import undistort_points
from camera_profiles import load_intrinsics, load_mount, load_profiles


def pixel_to_body_plane(
    pixel: tuple[float, float],
    target_z_mm: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation_bc: np.ndarray,
    camera_position_b: np.ndarray,
    model: str = "opencv_pinhole",
) -> np.ndarray:
    normalized = undistort_points(pixel, camera_matrix, distortion, model).reshape(2)
    direction_c = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    direction_b = rotation_bc @ direction_c
    if abs(direction_b[2]) < 1e-6:
        raise ValueError("ray is nearly parallel to target plane")
    scale = (target_z_mm - camera_position_b[2]) / direction_b[2]
    if scale <= 0:
        raise ValueError("intersection is behind the camera; verify pixel height and mount pose")
    return camera_position_b + scale * direction_b


def main() -> int:
    parser = argparse.ArgumentParser(description="Intersect an undistorted pixel ray with a known-height plane.")
    parser.add_argument("u", type=float)
    parser.add_argument("v", type=float)
    parser.add_argument("--target-z-mm", type=float, default=0.0)
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    args = parser.parse_args()

    _, intrinsics = load_intrinsics(args.camera)
    _, mount = load_mount(args.camera)
    camera_matrix, distortion = camera_arrays(intrinsics)
    point = pixel_to_body_plane(
        (args.u, args.v), args.target_z_mm, camera_matrix, distortion,
        np.asarray(mount["rotation_camera_to_body"], dtype=np.float64),
        np.asarray(mount["camera_position_body_mm"], dtype=np.float64),
        intrinsics["model"],
    )
    distance = math.hypot(point[0], point[1])
    bearing = math.degrees(math.atan2(point[1], point[0]))
    print(f"body point: X={point[0]:.1f} mm, Y={point[1]:.1f} mm, Z={point[2]:.1f} mm")
    print(f"horizontal distance={distance:.1f} mm, bearing={bearing:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
