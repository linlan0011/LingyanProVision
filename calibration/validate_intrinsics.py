from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration.calibrate_intrinsics import collect_images, per_view_errors
from calibration.calibration_io import camera_arrays, chessboard_object_points, load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate intrinsics on images excluded from fitting.")
    parser.add_argument("--intrinsics", type=Path, default=Path("calibration/results/lingyan/intrinsics.json"))
    parser.add_argument("--images", type=Path, default=Path("calibration/data/lingyan/validation"))
    parser.add_argument("--output", type=Path, default=Path("calibration/results/lingyan/validation.json"))
    parser.add_argument(
        "--preview-dir", type=Path,
        default=Path("calibration/results/lingyan/validation_previews"),
        help="side-by-side original and undistorted images",
    )
    args = parser.parse_args()

    data = load_json(args.intrinsics)
    camera_matrix, distortion = camera_arrays(data)
    cols, rows = map(int, data["chessboard_inner_corners"])
    square_mm = float(data["square_size_mm"])
    object_points, image_points, accepted, rejected, image_size = collect_images(
        args.images, (cols, rows), chessboard_object_points(cols, rows, square_mm),
    )
    expected = (int(data["image_width"]), int(data["image_height"]))
    if image_size != expected:
        raise ValueError(f"validation resolution {image_size} differs from calibration {expected}")
    if not accepted:
        raise ValueError("no validation image contains a complete chessboard")

    rvecs, tvecs = [], []
    for obj, image in zip(object_points, image_points):
        ok, rvec, tvec = cv2.solvePnP(obj, image, camera_matrix, distortion)
        if not ok:
            raise RuntimeError("solvePnP failed for a validation view")
        rvecs.append(rvec)
        tvecs.append(tvec)
    errors = per_view_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, distortion,
    )
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, 1.0, image_size,
    )
    preview_files = []
    for path_text in accepted:
        source = cv2.imread(path_text)
        corrected = cv2.undistort(source, camera_matrix, distortion, None, new_camera_matrix)
        cv2.putText(source, "original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(corrected, "undistorted", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)
        preview_path = args.preview_dir / f"{Path(path_text).stem}_comparison.png"
        if not cv2.imwrite(str(preview_path), np.hstack((source, corrected))):
            raise RuntimeError(f"failed to save {preview_path}")
        preview_files.append(str(preview_path))
    result = {
        "schema": "lingyan.camera_intrinsics_validation.v1",
        "intrinsics_file": str(args.intrinsics),
        "mean_view_rmse_px": float(np.mean(errors)),
        "max_view_rmse_px": float(np.max(errors)),
        "views": [{"file": path, "rmse_px": error} for path, error in zip(accepted, errors)],
        "rejected_files": rejected,
        "undistortion_previews": preview_files,
    }
    save_json(args.output, result)
    print(f"saved {args.output}")
    print(f"validation views={len(accepted)}, mean={np.mean(errors):.4f} px, max={np.max(errors):.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
