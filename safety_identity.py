"""Map corner numbering, detector-view presets and visual identity records.

Metric directed-edge identity and absolute pose are solved in nearest_corner_pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from pathlib import Path


from field_zones import FieldZone, matching_zones
from pose_io import Pose2D, normalize_heading_deg

DEFAULT_TEMPLATES = Path(__file__).with_name("safety_templates.json")
SIDE_NAMES = {
    "": "未绑定", "east": "东侧（朝西）", "west": "西侧（朝东）",
    "north": "北侧（朝南）", "south": "南侧（朝北）",
    "northeast": "东北斜视", "northwest": "西北斜视",
    "southeast": "东南斜视", "southwest": "西南斜视",
}
CORNER_NAMES = {1: "西北角", 2: "东北角", 3: "东南角", 4: "西南角"}


def load_templates(path=DEFAULT_TEMPLATES) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(model, dict) or model.get("schema") != "lingyan.safety_templates.v1":
        raise ValueError("unsupported safety template schema")
    if model.get("coordinate_frame") != "field_bottom_left_mm" or model.get("boundary") != "outer_frame":
        raise ValueError("templates require field_bottom_left_mm / outer_frame")
    zones = model["safety_zones"]
    if len(zones) != 2 or {z["id"] for z in zones} != {1, 2}:
        raise ValueError("safety zones must have unique IDs 1 and 2")
    for zone in zones:
        x0, x1, y0, y1 = zone["bounds_mm"]
        if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or not (x0 < x1 and y0 < y1):
            raise ValueError("invalid safety-zone bounds")
    half_boundary = model.get("field_half_boundary_y_mm")
    if not isinstance(half_boundary, (int, float)) or not math.isfinite(half_boundary):
        raise ValueError("invalid field half boundary")
    required_views = {"east", "west", "north", "south", "northeast", "northwest",
                      "southeast", "southwest"}
    if not isinstance(model["views"], dict) or set(model["views"]) != required_views:
        raise ValueError("eight half-field observation templates required")
    for view in model["views"].values():
        if not isinstance(view, dict) or view.get("scene") not in (
                "start-facing", "front-facing", "general"):
            raise ValueError("invalid detector scene")
        if not all(isinstance(view.get(key), (int, float)) and math.isfinite(view[key])
                   for key in ("heading_deg", "heading_tolerance_deg")):
            raise ValueError("invalid view heading")
        if not 0 < view["heading_tolerance_deg"] <= 180:
            raise ValueError("invalid view heading tolerance")
    return model


class TemplateReader:
    def __init__(self, path=DEFAULT_TEMPLATES):
        self.path = Path(path)
        self.stamp = None
        self.model = None
        self.error = None
        self.checked_at = 0.0

    def poll(self):
        try:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            # Windows may coalesce timestamps for rapid same-size writes.
            if stamp != self.stamp or self.model is None or time.monotonic() - self.checked_at > 0.5:
                self.model = load_templates(self.path)
                self.stamp = stamp
                self.checked_at = time.monotonic()
            self.error = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.model = None  # Never use stale rules after an invalid edit.
            self.error = str(exc)
        return self.model


def map_corners(zone):
    x0, x1, y0, y1 = zone["bounds_mm"]
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


@dataclass(frozen=True)
class ActiveTemplate:
    safety_id: int
    side: str
    scene: str


def active_templates(pose: Pose2D | None, zones: list[FieldZone], model):
    """Return all admissible hypotheses and a user-readable rejection reason."""
    if pose is None:
        return [], "waiting for pose"
    if model is None:
        return [], "template file unavailable"
    hits = matching_zones(pose, zones)
    if not hits:
        return [], "no observation zone"
    active = set()
    reasons = set()
    half_boundary = model["field_half_boundary_y_mm"]
    expected_safety_id = 1 if pose.y_mm >= half_boundary else 2
    for hit in hits:
        if not hit.safety_zone_id or not hit.view_side:
            reasons.add("observation zone not bound")
            continue
        if hit.safety_zone_id != expected_safety_id:
            # FieldZone rectangles include their edges; the half-field rule is
            # explicit so Y=boundary belongs only to safety zone 1.
            continue
        view = model["views"][hit.view_side]
        if abs(normalize_heading_deg(pose.heading_deg - view["heading_deg"])) > view["heading_tolerance_deg"]:
            reasons.add("heading outside template")
            continue
        # The position has already been assigned to a half-field observation
        # rectangle.  Do not add an outer-frame transverse-span gate here:
        # that gate incorrectly rejects valid diagonal views.
        active.add(ActiveTemplate(
            hit.safety_zone_id, hit.view_side, view["scene"],
        ))
    return sorted(active, key=lambda t: (t.safety_id, t.side)), "; ".join(sorted(reasons)) if not active else "template ready"


def choose_scene(templates, fallback="front-facing"):
    scenes = {t.scene for t in templates}
    return next(iter(scenes)) if len(scenes) == 1 else fallback


@dataclass
class CornerObservation:
    point: tuple[float, float]
    inferred: bool
    rays: tuple[float, ...]
    group: int = 0


@dataclass
class CornerIdentity:
    observation: CornerObservation
    candidates: set[str] = field(default_factory=set)
    corner_id: str | None = None
    reason: str = "insufficient boundary support"
