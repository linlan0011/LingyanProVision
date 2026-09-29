"""Named position and heading regions for recognition scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Optional

from pose_io import Pose2D, normalize_heading_deg


SCHEMA = "lingyan.field_zones.v1"


@dataclass(frozen=True)
class FieldZone:
    name: str
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    heading_deg: float
    heading_tolerance_deg: float

    def validated(self) -> "FieldZone":
        if not isinstance(self.name, str):
            raise ValueError("分区名称必须是文本")
        name = self.name.strip()
        if not name:
            raise ValueError("分区名称不能为空")
        values = (
            self.x_min_mm, self.x_max_mm, self.y_min_mm, self.y_max_mm,
            self.heading_deg, self.heading_tolerance_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("分区坐标和角度必须是有限数字")
        if self.x_min_mm >= self.x_max_mm or self.y_min_mm >= self.y_max_mm:
            raise ValueError("分区最大坐标必须大于最小坐标")
        if not 0 <= self.heading_tolerance_deg <= 180:
            raise ValueError("航向容差必须在 0° 到 180° 之间")
        return FieldZone(
            name,
            float(self.x_min_mm), float(self.x_max_mm),
            float(self.y_min_mm), float(self.y_max_mm),
            normalize_heading_deg(self.heading_deg),
            float(self.heading_tolerance_deg),
        )

    def contains(self, pose: Pose2D) -> bool:
        zone = self.validated()
        pose = pose.validated()
        angle_error = abs(normalize_heading_deg(pose.heading_deg - zone.heading_deg))
        return (
            zone.x_min_mm <= pose.x_mm <= zone.x_max_mm
            and zone.y_min_mm <= pose.y_mm <= zone.y_max_mm
            and angle_error <= zone.heading_tolerance_deg
        )

    def to_document(self) -> dict[str, str | float]:
        zone = self.validated()
        return {
            "name": zone.name,
            "x_min_mm": round(zone.x_min_mm, 3),
            "x_max_mm": round(zone.x_max_mm, 3),
            "y_min_mm": round(zone.y_min_mm, 3),
            "y_max_mm": round(zone.y_max_mm, 3),
            "heading_deg": round(zone.heading_deg, 3),
            "heading_tolerance_deg": round(zone.heading_tolerance_deg, 3),
        }


def read_zones(path: str | Path) -> list[FieldZone]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("不支持的分区文件格式")
    raw_zones = document.get("zones")
    if not isinstance(raw_zones, list):
        raise ValueError("分区文件缺少 zones 列表")
    zones: list[FieldZone] = []
    names: set[str] = set()
    for raw in raw_zones:
        if not isinstance(raw, dict):
            raise ValueError("分区条目必须是对象")
        zone = FieldZone(
            name=raw["name"],
            x_min_mm=float(raw["x_min_mm"]),
            x_max_mm=float(raw["x_max_mm"]),
            y_min_mm=float(raw["y_min_mm"]),
            y_max_mm=float(raw["y_max_mm"]),
            heading_deg=float(raw["heading_deg"]),
            heading_tolerance_deg=float(raw["heading_tolerance_deg"]),
        ).validated()
        if zone.name.casefold() in names:
            raise ValueError(f"分区名称重复：{zone.name}")
        names.add(zone.name.casefold())
        zones.append(zone)
    return zones


def write_zones_atomic(path: str | Path, zones: list[FieldZone]) -> None:
    checked = [zone.validated() for zone in zones]
    names = [zone.name.casefold() for zone in checked]
    if len(names) != len(set(names)):
        raise ValueError("分区名称不能重复")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    document = {
        "schema": SCHEMA,
        "coordinate_frame": "field_bottom_left_mm",
        "heading_convention": "0deg=+X, 90deg=+Y, counterclockwise_positive",
        "zones": [zone.to_document() for zone in checked],
    }
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def matching_zones(pose: Pose2D, zones: list[FieldZone]) -> list[FieldZone]:
    """Return every matching zone; overlapping zones are intentional."""
    return [zone for zone in zones if zone.contains(pose)]


class ZoneFileReader:
    """Reload the zone definitions only after the file changes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stamp_ns: Optional[int] = None
        self.zones: list[FieldZone] = []
        self.error: Optional[str] = None

    def poll(self) -> list[FieldZone]:
        try:
            stamp_ns = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._stamp_ns = None
            self.zones = []
            self.error = "waiting for zones file"
            return self.zones
        if stamp_ns == self._stamp_ns:
            return self.zones
        try:
            self.zones = read_zones(self.path)
            self.error = None
            self._stamp_ns = stamp_ns
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.zones = []
            self.error = str(exc)
        return self.zones
