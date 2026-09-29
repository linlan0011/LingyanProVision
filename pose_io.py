"""Shared pose-file contract for the map selector and vision process."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Optional


def normalize_heading_deg(value: float) -> float:
    """Normalize an angle to [-180, 180)."""
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == -0.0 else normalized


@dataclass(frozen=True)
class Pose2D:
    x_mm: float
    y_mm: float
    heading_deg: float

    def validated(self) -> "Pose2D":
        values = (self.x_mm, self.y_mm, self.heading_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite numbers")
        return Pose2D(
            float(self.x_mm),
            float(self.y_mm),
            normalize_heading_deg(self.heading_deg),
        )

    def to_document(self) -> dict[str, object]:
        pose = self.validated()
        return {
            "schema": "lingyan.pose.v1",
            "coordinate_frame": "field_bottom_left_mm",
            "x_mm": round(pose.x_mm, 3),
            "y_mm": round(pose.y_mm, 3),
            "heading_deg": round(pose.heading_deg, 3),
            "heading_convention": "0deg=+X, 90deg=+Y, counterclockwise_positive",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def write_pose_atomic(path: str | Path, pose: Pose2D) -> None:
    """Replace the output atomically so a reader never sees partial JSON."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(pose.to_document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def read_pose(path: str | Path) -> Pose2D:
    return parse_pose_text(Path(path).read_text(encoding="utf-8"))


def parse_pose_text(text: str) -> Pose2D:
    document = json.loads(text)
    if not isinstance(document, dict) or document.get("schema") != "lingyan.pose.v1":
        raise ValueError("unsupported pose schema")
    return Pose2D(
        float(document["x_mm"]),
        float(document["y_mm"]),
        float(document["heading_deg"]),
    ).validated()


class PoseFileReader:
    """Check small atomic pose files each frame; parse only changed content.

    Content comparison avoids missed rapid updates with equal Windows mtimes.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._text: Optional[str] = None
        self.pose: Optional[Pose2D] = None
        self.error: Optional[str] = None

    def poll(self) -> Optional[Pose2D]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.pose = None
            self._text = None
            self.error = "waiting for pose file"
            return None
        except (OSError, UnicodeError) as exc:
            self.pose = None
            self._text = None
            self.error = str(exc)
            return None
        if text == self._text and self.pose is not None:
            return self.pose
        try:
            self.pose = parse_pose_text(text)
            self.error = None
            self._text = text
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.pose = None
            self.error = str(exc)
        return self.pose
