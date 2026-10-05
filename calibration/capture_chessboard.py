from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from calibration.calibration_io import find_chessboard, load_json, save_json
from camera_orientation import ROTATION_CHOICES, rotate_frame
from camera_profiles import isolated_path, load_profiles, verify_capture_format
from purple_camera_segmentation import open_camera


def read_frame(cap, attempts=8):
    """Allow a short UVC startup/dropout gap, without reusing an old frame."""
    for attempt in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        if attempt + 1 < attempts:
            time.sleep(.05)
    return False, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture diverse chessboard calibration images.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--index", type=int)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--rotate", choices=ROTATION_CHOICES, default=None,
                        help="override profile rotation; keep fixed for calibration")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    camera_profile = load_profiles()[args.camera]
    width = args.width if args.width is not None else camera_profile.capture_width
    height = args.height if args.height is not None else camera_profile.capture_height
    rotation = args.rotate if args.rotate is not None else camera_profile.rotation
    if (width, height, rotation) != (
        camera_profile.capture_width, camera_profile.capture_height, camera_profile.rotation
    ):
        raise ValueError("capture overrides differ from registered profile; register another camera ID for this mode")
    index = args.index if args.index is not None else camera_profile.default_index
    output = isolated_path(args.output, args.camera, result=False, default_name="train")

    output.mkdir(parents=True, exist_ok=True)
    cap = open_camera(index, width, height, fourcc=camera_profile.pixel_format,
                      fps=camera_profile.target_fps, profile=camera_profile)
    reported = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok, first_frame = read_frame(cap)
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("camera opened but did not provide a frame")
    actual = (first_frame.shape[1], first_frame.shape[0])
    try:
        verify_capture_format(cap, camera_profile)
    except Exception:
        cap.release()
        raise
    print(f"camera={args.camera}, index={index}, requested={width}x{height}, "
          f"reported={reported[0]}x{reported[1]}, frame={actual[0]}x{actual[1]}")
    if actual != (width, height):
        cap.release()
        raise RuntimeError("camera did not accept requested resolution; do not mix resolutions")
    output_frame = rotate_frame(first_frame, rotation)
    settings = {"camera_id": args.camera, "capture_width": width, "capture_height": height,
                "rotation": rotation, "image_width": output_frame.shape[1],
                "image_height": output_frame.shape[0], "pixel_format": camera_profile.pixel_format,
                "target_fps": camera_profile.target_fps, "model": camera_profile.lens_model,
                "backend": camera_profile.backend, "device_vid_pid": camera_profile.device_vid_pid,
                "fixed_controls": camera_profile.fixed_controls,
                "automatic_controls": camera_profile.automatic_controls,
                "board_inner_corners": [args.cols, args.rows]}
    settings_path = output / "capture_settings.json"
    previous_settings = load_json(settings_path) if settings_path.exists() else None
    if previous_settings is not None:
        previous_settings.setdefault("automatic_controls", None)
    if previous_settings is not None and previous_settings != settings:
        cap.release()
        raise ValueError("existing capture settings differ; use another camera ID or a new dataset")
    if not settings_path.exists() and not any(output.glob("*.png")):
        save_json(settings_path, settings)
    print(f"rotation={rotation}, saved image={output_frame.shape[1]}x{output_frame.shape[0]}, folder={output}")
    print(f"format={camera_profile.pixel_format or 'default'}, requested_fps={camera_profile.target_fps or 'default'}", flush=True)
    print(f"backend={cap.getBackendName()}, device={camera_profile.device_vid_pid or 'unbound'}", flush=True)

    window = "Calibration capture - SPACE save, Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1280, output_frame.shape[1]), min(720, output_frame.shape[0]))
    cv2.imshow(window, output_frame)
    cv2.waitKey(1)
    print(f"preview window ready; visible={cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)}", flush=True)
    saved = len(list(output.glob("*.png")))
    previous_found = None
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok or frame is None:
                raise RuntimeError("camera frame read failed")
            if (frame.shape[1], frame.shape[0]) != actual:
                raise RuntimeError("camera resolution changed during capture")
            frame = rotate_frame(frame, rotation)
            # Corner detection here is only a capture aid. Save the full-resolution frame;
            # the intrinsic solver detects/subpixel-refines corners on that original image.
            preview_scale = min(1.0, 960 / frame.shape[1], 720 / frame.shape[0])
            preview = cv2.resize(frame, None, fx=preview_scale, fy=preview_scale,
                                 interpolation=cv2.INTER_AREA) if preview_scale < 1 else frame.copy()
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard(gray, (args.cols, args.rows))
            if found != previous_found:
                print(f"checkerboard={'FOUND' if found else 'NOT FOUND'}; SPACE saves the native {frame.shape[1]}x{frame.shape[0]} image", flush=True)
                previous_found = found
            if found and corners is not None:
                cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, found)
            status = f"{'FOUND' if found else 'NOT FOUND'}  saved={saved}  SPACE=save  Q=quit"
            cv2.putText(preview, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 220, 0) if found else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not found:
                    print("not saved: chessboard was not fully detected")
                    continue
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = output / f"chessboard_{stamp}_{saved + 1:03d}.png"
                while path.exists():
                    saved += 1
                    path = output / f"chessboard_{stamp}_{saved + 1:03d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to save {path}")
                saved += 1
                print(f"saved {path}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
