"""Report which requested capture modes a connected camera actually returns."""
from __future__ import annotations

import argparse

import cv2


COMMON_MODES = ((640, 480), (1280, 720), (1280, 960), (1280, 1024),
                (1600, 1200), (1920, 1080), (2048, 1536), (2560, 1440))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe actual camera frame sizes before registering a new profile")
    parser.add_argument("--index", type=int, default=0, help="device index; close other camera programs first")
    parser.add_argument("--modes", nargs="*", help="optional WxH requests; defaults to common modes")
    args = parser.parse_args()
    modes = COMMON_MODES if not args.modes else []
    for value in args.modes or ():
        try:
            width, height = map(int, value.lower().split("x"))
            if width < 16 or height < 16:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"invalid capture mode {value!r}; use WIDTHxHEIGHT") from exc
        modes.append((width, height))
    for width, height in modes:
        cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.index)
        if not cap.isOpened():
            raise RuntimeError(f"camera index {args.index} cannot be opened; check connection and other camera programs")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"requested {width}x{height}: no frame")
                continue
            actual = (frame.shape[1], frame.shape[0])
            fps = cap.get(cv2.CAP_PROP_FPS)
            status = "EXACT" if actual == (width, height) else "FALLBACK"
            print(f"requested {width}x{height}: actual {actual[0]}x{actual[1]}, "
                  f"reported fps {fps:.1f}, {status}")
        finally:
            cap.release()
    print("Mode support must be confirmed on the intended physical device; this probe cannot verify hardware identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
