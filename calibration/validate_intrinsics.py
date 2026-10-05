from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration.calibrate_intrinsics import collect_images, per_view_errors
from calibration.calibration_io import camera_arrays, chessboard_object_points, save_result_json
from datetime import datetime
from calibration.lens_models import solve_pose, undistort_preview
from camera_profiles import isolated_path, load_intrinsics, load_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate intrinsics on images excluded from fitting.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preview-dir", type=Path,
        help="side-by-side original and undistorted images",
    )
    args = parser.parse_args()
    intrinsics_path, data = load_intrinsics(args.camera)
    images = isolated_path(args.images, args.camera, result=False, default_name="validation")
    output = isolated_path(args.output, args.camera, result=True, default_name="validation.json")
    preview_root = isolated_path(args.preview_dir, args.camera, result=True, default_name="validation_previews")
    preview_dir = preview_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

    camera_matrix, distortion = camera_arrays(data)
    cols, rows = map(int, data["chessboard_inner_corners"])
    square_mm = float(data["square_size_mm"])
    object_points, image_points, accepted, rejected, image_size = collect_images(
        images, (cols, rows), chessboard_object_points(cols, rows, square_mm),
    )
    expected = (int(data["image_width"]), int(data["image_height"]))
    if image_size != expected:
        raise ValueError(f"validation resolution {image_size} differs from calibration {expected}")
    if not accepted:
        raise ValueError("no validation image contains a complete chessboard")

    rvecs, tvecs = [], []
    for obj, image in zip(object_points, image_points):
        ok, rvec, tvec = solve_pose(obj, image, camera_matrix, distortion, data["model"])
        if not ok:
            raise RuntimeError("solvePnP failed for a validation view")
        rvecs.append(rvec)
        tvecs.append(tvec)
    errors = per_view_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, distortion, data["model"],
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_files = []
    for path_text in accepted:
        source = cv2.imread(path_text)
        corrected = undistort_preview(source, camera_matrix, distortion, data["model"])
        cv2.putText(source, "original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(corrected, "undistorted", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)
        preview_path = preview_dir / f"{Path(path_text).stem}_comparison.png"
        if not cv2.imwrite(str(preview_path), np.hstack((source, corrected))):
            raise RuntimeError(f"failed to save {preview_path}")
        preview_files.append(str(preview_path))
    result = {
        "schema": "camera.intrinsics_validation.v2",
        "camera_id": args.camera,
        "intrinsics_file": str(intrinsics_path),
        "mean_view_rmse_px": float(np.mean(errors)),
        "max_view_rmse_px": float(np.max(errors)),
        "views": [{"file": path, "rmse_px": error} for path, error in zip(accepted, errors)],
        "rejected_files": rejected,
        "undistortion_previews": preview_files,
    }
    backup = save_result_json(output, result)
    print(f"saved {output}")
    if backup:
        print(f"previous validation backed up to {backup}")
    print(f"validation views={len(accepted)}, mean={np.mean(errors):.4f} px, max={np.max(errors):.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
