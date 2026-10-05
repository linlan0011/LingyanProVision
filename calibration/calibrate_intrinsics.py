from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_io import (
    chessboard_object_points,
    find_chessboard,
    save_result_json,
    load_json,
)
from calibration.lens_models import calibrate, project_points
from camera_profiles import isolated_path, load_profiles


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


def per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix, distortion, model="opencv_pinhole"):
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected = project_points(obj, rvec, tvec, camera_matrix, distortion, model)
        delta = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate registered camera intrinsics from chessboard images.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.square_mm) or args.square_mm <= 0:
        raise ValueError("measured chessboard square size must be a positive finite number in millimetres")
    profile = load_profiles()[args.camera]
    images = isolated_path(args.images, args.camera, result=False, default_name="train")
    output = isolated_path(args.output, args.camera, result=True, default_name="intrinsics.json")
    capture_metadata = images / "capture_settings.json"
    if capture_metadata.exists():
        settings = load_json(capture_metadata)
        for key, value in {"camera_id": args.camera, "capture_width": profile.capture_width,
                           "capture_height": profile.capture_height, "rotation": profile.rotation,
                           "pixel_format": profile.pixel_format, "target_fps": profile.target_fps,
                           "backend": profile.backend, "device_vid_pid": profile.device_vid_pid,
                           "fixed_controls": profile.fixed_controls,
                           "automatic_controls": profile.automatic_controls,
                           "model": profile.lens_model, "board_inner_corners": [args.cols, args.rows]}.items():
            if settings.get(key) != value:
                raise ValueError(f"capture settings {key} differ from the selected camera/board")

    pattern = (args.cols, args.rows)
    template = chessboard_object_points(args.cols, args.rows, args.square_mm)
    object_points, image_points, accepted, rejected, image_size = collect_images(
        images, pattern, template,
    )
    if image_size != profile.image_size:
        raise ValueError(
            f"images are {image_size[0]}x{image_size[1]}, expected "
            f"{profile.image_size[0]}x{profile.image_size[1]}; check camera profile"
        )
    if len(accepted) < 10:
        raise ValueError(f"only {len(accepted)} usable views; collect at least 10, preferably 20-30")

    rms, camera_matrix, distortion, rvecs, tvecs = calibrate(
        object_points, image_points, image_size, profile.lens_model)
    errors = per_view_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, distortion, profile.lens_model,
    )
    result = {
        "schema": "camera.intrinsics.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "camera_id": args.camera,
        "camera_name": profile.display_name,
        "model": profile.lens_model,
        "capture_width": profile.capture_width,
        "capture_height": profile.capture_height,
        "rotation": profile.rotation,
        "pixel_format": profile.pixel_format,
        "target_fps": profile.target_fps,
        "backend": profile.backend,
        "device_vid_pid": profile.device_vid_pid,
        "fixed_controls": profile.fixed_controls,
        "automatic_controls": profile.automatic_controls,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "chessboard_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_mm,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_order": (["k1", "k2", "p1", "p2", "k3"] if profile.lens_model == "opencv_pinhole"
                             else ["k1", "k2", "k3", "k4"]),
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
    backup = save_result_json(output, result)
    print(f"saved {output}")
    if backup:
        print(f"previous intrinsics backed up to {backup}; refit mount pose with the new intrinsics")
    print(f"usable={len(accepted)}, rejected={len(rejected)}, RMS={rms:.4f} px")
    print(f"fx={camera_matrix[0, 0]:.3f}, fy={camera_matrix[1, 1]:.3f}, "
          f"cx={camera_matrix[0, 2]:.3f}, cy={camera_matrix[1, 2]:.3f}")
    print(f"mean view RMSE={np.mean(errors):.4f} px, max={np.max(errors):.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
