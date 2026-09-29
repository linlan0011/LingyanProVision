"""Apply the same camera orientation to previews and calibration images."""

from __future__ import annotations

import cv2
import numpy as np


ROTATION_CHOICES = ("none", "clockwise", "counterclockwise", "half-turn")


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "none":
        return frame
    codes = {
        "clockwise": cv2.ROTATE_90_CLOCKWISE,
        "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "half-turn": cv2.ROTATE_180,
    }
    try:
        return cv2.rotate(frame, codes[rotation])
    except KeyError as error:
        raise ValueError(f"unsupported rotation: {rotation}") from error
