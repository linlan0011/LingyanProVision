from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_io import (
    chessboard_object_points,
    find_chessboard,
    save_json,
)


def collect_images(folder: Path, pattern: tuple[int, int], object_template: np.ndarray):
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            rejected.append(str(path))
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            raise ValueError(f"mixed image resolutions: {path} is {size}, expected {image_size}")
        found, corners = find_chessboard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern)
        if not found or corners is None:
            rejected.append(str(path))
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(str(path))
    if image_size is None:
        raise ValueError(f"no images found in {folder}")
    return object_points, image_points, accepted, rejected, image_size


def per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix, distortion):
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        delta = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate pinhole intrinsics from chessboard images.")
    parser.add_argument("--images", type=Path, default=Path("calibration/data/lingyan/train"))
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("calibration/results/lingyan/intrinsics.json"))
    parser.add_argument("--camera-name", default="LINGYAN Pro USB camera")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    template = chessboard_object_points(args.cols, args.rows, args.square_mm)
    object_points, image_points, accepted, rejected, image_size = collect_images(
        args.images, pattern, template,
    )
    if image_size != (args.expected_width, args.expected_height):
        raise ValueError(
            f"images are {image_size[0]}x{image_size[1]}, expected "
            f"{args.expected_width}x{args.expected_height}; check camera profile"
        )
    if len(accepted) < 10:
        raise ValueError(f"only {len(accepted)} usable views; collect at least 10, preferably 20-30")

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
    )
    errors = per_view_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, distortion,
    )
    result = {
        "schema": "lingyan.camera_intrinsics.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "camera_name": args.camera_name,
        "model": "opencv_pinhole",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "chessboard_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_mm,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rms_reprojection_error_px": float(rms),
        "mean_view_rmse_px": float(np.mean(errors)),
        "max_view_rmse_px": float(np.max(errors)),
        "views": [
            {"file": path, "rmse_px": error}
            for path, error in zip(accepted, errors)
        ],
        "rejected_files": rejected,
        "notes": "Valid only for this resolution, lens focus, zoom and camera settings.",
    }
    save_json(args.output, result)
    print(f"saved {args.output}")
    print(f"usable={len(accepted)}, rejected={len(rejected)}, RMS={rms:.4f} px")
    print(f"fx={camera_matrix[0, 0]:.3f}, fy={camera_matrix[1, 1]:.3f}, "
          f"cx={camera_matrix[0, 2]:.3f}, cy={camera_matrix[1, 2]:.3f}")
    print(f"mean view RMSE={np.mean(errors):.4f} px, max={np.max(errors):.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
