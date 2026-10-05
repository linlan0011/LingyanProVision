from __future__ import annotations

import json
import csv
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_result_json(path: str | Path, data: dict[str, Any]) -> Path | None:
    """Atomically replace a computed result, preserving any prior version."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = output.with_name(f"{output.stem}.backup_{stamp}{output.suffix}")
        shutil.copy2(output, backup)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def validate_point_sources(path: Path, camera_id: str) -> None:
    """Reject collector CSVs that point to images in another camera's folder."""
    from camera_profiles import data_dir

    root = data_dir(camera_id).resolve()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "frame_file" not in (reader.fieldnames or ()):
            return  # Older manually prepared CSVs have no image provenance.
        for line, row in enumerate(reader, start=2):
            frame = row.get("frame_file", "").strip()
            if not frame:
                continue
            source = Path(frame).resolve()
            if not source.is_relative_to(root):
                raise ValueError(f"row {line} references another camera's image: {frame}")
            if not source.is_file():
                raise FileNotFoundError(f"row {line} source image is missing: {frame}")


def camera_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(data["distortion_coefficients"], dtype=np.float64)
    return camera_matrix, distortion


def chessboard_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    if cols < 2 or rows < 2 or not math.isfinite(square_size_mm) or square_size_mm <= 0:
        raise ValueError("chessboard dimensions and measured square size must be positive")
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= square_size_mm
    return points


def find_chessboard(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    # SB is more robust near image edges and already returns sub-pixel corners.
    import cv2

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray, pattern, sb_flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners
