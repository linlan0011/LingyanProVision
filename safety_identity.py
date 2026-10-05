"""Pose-gated, directed-boundary templates. No metric camera calibration assumed.

IDs are map-fixed: 1 north / 2 south, NW -> NE -> SE -> SW clockwise.
Template thresholds are initial engineering defaults, not measured accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

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
    projection = model.get("diagonal_pose_projection")
    if not isinstance(projection, dict):
        raise ValueError("missing diagonal pose projection")
    for key in ("horizontal_fov_deg", "image_bearing_tolerance_deg"):
        if (not isinstance(projection.get(key), (int, float)) or not math.isfinite(projection[key])
                or projection[key] <= 0):
            raise ValueError(f"invalid diagonal pose projection: {key}")
    if not 20 <= projection["horizontal_fov_deg"] < 180:
        raise ValueError("invalid diagonal horizontal field of view")
    for key in ("horizontal_tolerance_deg", "diagonal_tolerance_deg",
                "minimum_pair_angle_deg", "ray_line_distance_px_at_640", "visible_endpoint_gap_px_at_640",
                "inferred_endpoint_gap_px_at_640", "minimum_ray_length_px_at_640"):
        if not isinstance(model[key], (int, float)) or not math.isfinite(model[key]) or model[key] <= 0:
            raise ValueError(f"invalid template threshold: {key}")
    for key in ("horizontal_tolerance_deg", "diagonal_tolerance_deg"):
        if model[key] >= 45:
            raise ValueError(f"{key} must be below 45 degrees")
    required_views = {"east", "west", "north", "south", "northeast", "northwest",
                      "southeast", "southwest"}
    if not isinstance(model["views"], dict) or set(model["views"]) != required_views:
        raise ValueError("eight half-field observation templates required")
    for view in model["views"].values():
        enabled = view.get("identity_enabled", True)
        identity_mode = view.get("identity_mode", "directed_edges")
        if (not isinstance(enabled, bool) or view.get("scene") not in
                ("start-facing", "front-facing", "general")):
            raise ValueError("invalid detector scene or identity flag")
        if identity_mode not in ("directed_edges", "pose_projection"):
            raise ValueError("invalid identity mode")
        if not all(isinstance(view.get(key), (int, float)) and math.isfinite(view[key])
                   for key in ("heading_deg", "heading_tolerance_deg")):
            raise ValueError("invalid view heading")
        if not 0 < view["heading_tolerance_deg"] <= 180:
            raise ValueError("invalid view heading tolerance")
        if enabled and identity_mode == "directed_edges" and sorted(view.get("slots", [])) != [1, 2, 3, 4]:
            raise ValueError("invalid corner mapping")
        if identity_mode == "pose_projection" and sorted(view.get("image_slots", [])) != [1, 2, 3, 4]:
            raise ValueError("invalid diagonal image-extrema mapping")
    if len(model["slot_rules"]) != 4:
        raise ValueError("four slot rules required")
    for rule in model["slot_rules"]:
        if not all(math.isfinite(rule[k]) for k in ("horizontal_deg", "depth_deg")):
            raise ValueError("invalid edge direction")
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
    slots: tuple[int, ...]
    identity_enabled: bool = True
    identity_mode: str = "directed_edges"
    image_slots: tuple[int, ...] = ()


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
        target = next(z for z in model["safety_zones"] if z["id"] == hit.safety_zone_id)
        view = model["views"][hit.view_side]
        if abs(normalize_heading_deg(pose.heading_deg - view["heading_deg"])) > view["heading_tolerance_deg"]:
            reasons.add("heading outside template")
            continue
        # The position has already been assigned to a half-field observation
        # rectangle.  Do not add an outer-frame transverse-span gate here:
        # that gate incorrectly rejects valid diagonal views.
        active.add(ActiveTemplate(
            hit.safety_zone_id, hit.view_side, view["scene"], tuple(view.get("slots", ())),
            view.get("identity_enabled", True), view.get("identity_mode", "directed_edges"),
            tuple(view.get("image_slots", ())),
        ))
    return sorted(active, key=lambda t: (t.safety_id, t.side)), "; ".join(sorted(reasons)) if not active else "template ready"


def choose_scene(templates, fallback="front-facing"):
    scenes = {t.scene for t in templates}
    return next(iter(scenes)) if len(scenes) == 1 else fallback


def directed_rays(point, segments, inferred, width, model):
    """Use nearby *endpoints*, not an infinite unoriented line at the corner.

    segments are the detector's already white-supported fitted boundaries.
    A segment crossing through the point cannot serve as a corner ray.
    """
    p = np.asarray(point, dtype=float)
    scale = width / 640.0
    line_limit = model["ray_line_distance_px_at_640"] * scale
    gap = model["inferred_endpoint_gap_px_at_640" if inferred else "visible_endpoint_gap_px_at_640"] * scale
    min_length = model["minimum_ray_length_px_at_640"] * scale
    candidates = []
    for segment in segments:
        start = np.asarray(segment.fitted_start, dtype=float)
        end = np.asarray(segment.fitted_end, dtype=float)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < min_length:
            continue
        unit = vector / length
        projection = float(np.dot(p - start, unit))
        perpendicular = float(np.linalg.norm(p - start - projection * unit))
        if perpendicular > line_limit or min(np.linalg.norm(p-start), np.linalg.norm(p-end)) > gap:
            continue
        if min(projection, length - projection) > line_limit:
            continue
        # Orient toward the supported part of the boundary.
        direction = unit if np.linalg.norm(p-end) > np.linalg.norm(p-start) else -unit
        angle = math.degrees(math.atan2(-direction[1], direction[0]))
        candidates.append((perpendicular, angle))
    rays = []
    for _, angle in sorted(candidates):
        if not any(abs(normalize_heading_deg(angle - old)) < 10 for old in rays):
            rays.append(angle)
    return tuple(rays)


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


def projected_corner_candidates(observation, template, pose, image_width, model):
    """Return map corners consistent with a diagonal observation's image x position.

    The pose gives a map corner's bearing in the vehicle frame.  A pinhole
    horizontal projection maps that bearing to image x.  This deliberately
    does not use the apparent slope of the two boundary lines, which varies
    sharply with diagonal viewpoint.
    """
    if pose is None or not image_width:
        return set()
    target = next(zone for zone in model["safety_zones"] if zone["id"] == template.safety_id)
    config = model["diagonal_pose_projection"]
    half_fov = config["horizontal_fov_deg"] / 2
    focal_length = image_width / (2 * math.tan(math.radians(half_fov)))
    image_bearing = math.degrees(math.atan2(image_width / 2 - observation.point[0], focal_length))
    heading = math.radians(pose.heading_deg)
    candidates = set()
    for index, (x_mm, y_mm) in enumerate(map_corners(target), 1):
        dx, dy = x_mm - pose.x_mm, y_mm - pose.y_mm
        forward = math.cos(heading) * dx + math.sin(heading) * dy
        left = -math.sin(heading) * dx + math.cos(heading) * dy
        if forward <= 0:
            continue
        expected_bearing = math.degrees(math.atan2(left, forward))
        if (abs(expected_bearing) <= half_fov
                and abs(normalize_heading_deg(image_bearing - expected_bearing))
                <= config["image_bearing_tolerance_deg"]):
            candidates.add(f"{template.safety_id}.{index}")
    return candidates


def diagonal_extrema_candidates(observations, templates):
    """Map a complete diagonal quadrilateral by its four image extrema.

    The real-time 1-SW frame establishes this ordering: left, top, right,
    bottom map to the template's four image slots.  The other diagonal views
    use the reflected/rotated ordering configured with their own templates.
    A group must contain exactly four directly observed, distinct extrema;
    otherwise the caller falls back to the single-point pose projection.
    """
    diagonal = [template for template in templates
                if template.identity_enabled and template.identity_mode == "pose_projection"
                and len(template.image_slots) == 4]
    if len(diagonal) != 1:
        return {}
    template = diagonal[0]
    candidates = {}
    groups = {observation.group for observation in observations if not observation.inferred}
    for group in groups:
        indices = [index for index, observation in enumerate(observations)
                   if observation.group == group and not observation.inferred]
        if len(indices) != 4:
            continue
        left = min(indices, key=lambda index: observations[index].point[0])
        top = min(indices, key=lambda index: observations[index].point[1])
        right = max(indices, key=lambda index: observations[index].point[0])
        bottom = max(indices, key=lambda index: observations[index].point[1])
        extrema = (left, top, right, bottom)
        if len(set(extrema)) != 4:
            continue
        for index, slot in zip(extrema, template.image_slots):
            candidates[index] = f"{template.safety_id}.{slot}"
    return candidates


def associate_corners(observations, templates, model, pose=None, image_width=None, calibrated_ranger=None):
    results = []
    if model is None:
        return [CornerIdentity(o, reason="template file unavailable") for o in observations]
    extrema_matches = diagonal_extrema_candidates(observations, templates)
    for observation_index, observation in enumerate(observations):
        result = CornerIdentity(observation)
        identity_templates = [template for template in templates if template.identity_enabled]
        if templates and not identity_templates:
            result.reason = "diagonal region requires measured edge template"
            results.append(result)
            continue
        projection_templates = [template for template in identity_templates
                                if template.identity_mode == "pose_projection"]
        edge_templates = [template for template in identity_templates
                          if template.identity_mode == "directed_edges"]
        if observation_index in extrema_matches:
            result.candidates.add(extrema_matches[observation_index])
        else:
            for template in projection_templates:
                if calibrated_ranger is not None:
                    result.candidates.update(calibrated_ranger.identity_candidates(observation, template, pose, model))
                else:
                    result.candidates.update(projected_corner_candidates(
                        observation, template, pose, image_width, model,
                    ))
        for a, b in combinations(observation.rays, 2):
            separation = abs(normalize_heading_deg(a-b))
            if not model["minimum_pair_angle_deg"] <= separation <= 180-model["minimum_pair_angle_deg"]:
                continue
            for template in edge_templates:
                for slot, rule in zip(template.slots, model["slot_rules"]):
                    for horizontal, depth in ((a, b), (b, a)):
                        if (abs(normalize_heading_deg(horizontal-rule["horizontal_deg"])) <= model["horizontal_tolerance_deg"]
                            and abs(normalize_heading_deg(depth-rule["depth_deg"])) <= model["diagonal_tolerance_deg"]):
                            result.candidates.add(f"{template.safety_id}.{slot}")
        if len(result.candidates) == 1:
            result.corner_id = next(iter(result.candidates))
            result.reason = "unique template match"
        elif result.candidates:
            result.reason = "ambiguous templates"
        elif not templates:
            result.reason = "no applicable template"
        elif len(observation.rays) >= 2:
            result.reason = "boundary directions do not match"
        results.append(result)
    # Do not label two separate detections as the same physical corner.
    for corner_id in {r.corner_id for r in results if r.corner_id}:
        duplicates = [r for r in results if r.corner_id == corner_id]
        if len(duplicates) > 1:
            for result in duplicates:
                result.corner_id = None
                result.reason = "duplicate corner conflict"
    return results


def draw_identity_overlay(frame, results, templates, status, header=True):
    """Numbers at detections; footer separates pose candidates from visual IDs."""
    for result in results:
        x, y = map(int, result.observation.point)
        label = (result.corner_id + " " + {1: "NW", 2: "NE", 3: "SE", 4: "SW"}[int(result.corner_id.split('.')[1])]
                 if result.corner_id else "?")
        if result.observation.inferred:
            label += " (inferred)"
        color = (80, 255, 80) if result.corner_id else (0, 180, 255)
        x = max(2, min(x+12, frame.shape[1]-cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .65, 2)[0][0]-2))
        y = max(18, min(y+30, frame.shape[0]-8))
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    if not header:
        return frame
    footer = np.full((86, frame.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    corners = sorted({r.corner_id for r in results if r.corner_id})
    candidate_text = ", ".join(f"{t.safety_id}/{t.side}" for t in templates) or "none"
    identified = sorted({c.split('.')[0] for c in corners})
    pose_zone_ids = sorted({str(template.safety_id) for template in templates})
    zone_text = ",".join(identified or pose_zone_ids) or "pending"
    zone_source = "" if identified else (" (pose template)" if pose_zone_ids else "")
    match_source = ("matched by pose projection" if any(t.identity_mode == "pose_projection" for t in templates)
                    else "matched by boundary templates")
    lines = [f"POSE candidates: {candidate_text}",
             f"SAFETY ZONE: {zone_text}{zone_source}   CORNERS: {', '.join(corners) or 'pending'}",
             status if not templates else (match_source if corners else
                 (results[0].reason if results else "waiting for valid corners"))]
    for i, text in enumerate(lines):
        cv2.putText(footer, text, (8, 22+26*i), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (100, 255, 150) if i == 1 and corners else (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack((frame, footer))
