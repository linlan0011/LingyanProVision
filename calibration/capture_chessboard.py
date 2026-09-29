from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from calibration.calibration_io import find_chessboard
from camera_orientation import ROTATION_CHOICES, rotate_frame
from purple_camera_segmentation import CAMERA_PROFILES, open_camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture diverse chessboard calibration images.")
    parser.add_argument("--camera", choices=CAMERA_PROFILES, default="lingyan",
                        help="camera capture profile; default is the original LINGYAN camera")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--rotate", choices=ROTATION_CHOICES, default=None,
                        help="override profile rotation; keep fixed for calibration")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    camera_profile = CAMERA_PROFILES[args.camera]
    width = args.width if args.width is not None else camera_profile.capture_width
    height = args.height if args.height is not None else camera_profile.capture_height
    rotation = args.rotate if args.rotate is not None else camera_profile.rotation
    output = args.output if args.output is not None else Path("calibration/data") / args.camera / "train"

    output.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.index, width, height)
    reported = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("camera opened but did not provide a frame")
    actual = (first_frame.shape[1], first_frame.shape[0])
    print(f"camera={args.camera}, index={args.index}, requested={width}x{height}, "
          f"reported={reported[0]}x{reported[1]}, frame={actual[0]}x{actual[1]}")
    if actual != (width, height):
        cap.release()
        raise RuntimeError("camera did not accept requested resolution; do not mix resolutions")
    output_frame = rotate_frame(first_frame, rotation)
    print(f"rotation={rotation}, saved image={output_frame.shape[1]}x{output_frame.shape[0]}, folder={output}")

    window = "Calibration capture - SPACE save, Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved = len(list(output.glob("*.png")))
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("camera frame read failed")
            if (frame.shape[1], frame.shape[0]) != actual:
                raise RuntimeError("camera resolution changed during capture")
            frame = rotate_frame(frame, rotation)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard(gray, (args.cols, args.rows))
            preview = frame.copy()
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
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to save {path}")
                saved += 1
                print(f"saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
