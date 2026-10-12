"""Live USB-camera preview with an OpenCV purple-region segmentation overlay.

Press q or Esc to stop. Camera profiles keep the earlier LINGYAN settings
separate from the current SeekFree camera.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from boundary_geometry import BoundaryConfig, PointGeometry, detect_boundaries, draw_boundary_result
from bottom_edge_classifier import BottomEdgeConfig, BottomEdgeResult, classify_bottom_edges
from corner_ranging import DEFAULT_MAX_DISTANCE_MM
from ground_corner_association import (GroundAssociationConfig, CalibratedGroundView,
    associate_ground_corners, apply_ground_identities, circle_templates)
from preview_display import ParameterPanel, preview_dimensions, compose_preview
from camera_orientation import ROTATION_CHOICES, rotate_frame
from camera_profiles import CameraProfile, load_intrinsics, load_mount, load_profiles, verify_capture_format
from field_zones import ZoneFileReader, matching_zones
from pose_io import PoseFileReader, normalize_heading_deg
from safety_identity import (DEFAULT_TEMPLATES, TemplateReader, active_templates,
    choose_scene, CornerObservation, directed_rays, supported_corner_rays,
    associate_corners, draw_identity_overlay)


@dataclass(frozen=True)
class SceneRules:
    h_low: int
    h_high: int
    s_min: int
    v_min: int
    min_area: int
    corner_min_area: int
    corner_min_side: int
    white_s_max: int
    white_v_min: int
    near_band_percent: int
    min_side_fraction: float
    min_perimeter_fraction: float
    min_corner_angle: float
    max_corner_angle: float
    min_white_support: float
    display_white_support: float
    require_both_white_edges: bool
    merge_structural_turns: bool
    merge_x_fraction: float
    merge_y_fraction: float
    infer_hidden_corners: bool
    adaptive_exposure: bool
    allow_split_components: bool
    split_max_gap_fraction: float
    front_approx_ratio: float = 0.008


CAMERA_PROFILES = load_profiles()


# Keep the completed launch-side settings independent of the frontal view.
# Future frontal tuning must change FRONT_RULES only.
SIDE_RULES = SceneRules(
    h_low=115, h_high=175, s_min=55, v_min=40,
    min_area=80, corner_min_area=400, corner_min_side=12,
    white_s_max=179, white_v_min=100, near_band_percent=10,
    min_side_fraction=0.12, min_perimeter_fraction=0.012,
    min_corner_angle=20.0, max_corner_angle=145.0,
    min_white_support=0.40, display_white_support=0.40,
    require_both_white_edges=True,
    merge_structural_turns=True, merge_x_fraction=0.05,
    merge_y_fraction=0.133, infer_hidden_corners=False,
    adaptive_exposure=False, allow_split_components=False,
    split_max_gap_fraction=0.10,
)
FRONT_RULES = SceneRules(
    h_low=115, h_high=175, s_min=55, v_min=40,
    min_area=80, corner_min_area=400, corner_min_side=12,
    white_s_max=80, white_v_min=100, near_band_percent=10,
    min_side_fraction=0.12, min_perimeter_fraction=0.012,
    min_corner_angle=15.0, max_corner_angle=145.0,
    min_white_support=0.40, display_white_support=0.35,
    require_both_white_edges=True,
    merge_structural_turns=True, merge_x_fraction=0.08,
    merge_y_fraction=0.133, infer_hidden_corners=True,
    adaptive_exposure=True, allow_split_components=True,
    split_max_gap_fraction=0.16,
)
GENERAL_RULES = SceneRules(
    h_low=115, h_high=175, s_min=55, v_min=40,
    min_area=80, corner_min_area=400, corner_min_side=12,
    white_s_max=179, white_v_min=100, near_band_percent=10,
    min_side_fraction=0.65, min_perimeter_fraction=0.012,
    min_corner_angle=20.0, max_corner_angle=160.0,
    min_white_support=0.40, display_white_support=0.40,
    require_both_white_edges=False,
    merge_structural_turns=False, merge_x_fraction=0.0,
    merge_y_fraction=0.0, infer_hidden_corners=True,
    adaptive_exposure=False, allow_split_components=True,
    split_max_gap_fraction=0.10,
)
SCENE_RULES = {
    "start-facing": SIDE_RULES,
    "front-facing": FRONT_RULES,
    "general": GENERAL_RULES,
}
# Preserve all prior LINGYAN scene values. SeekFree changes are scoped here.
SEEKFREE_SCENE_RULES = {
    "start-facing": replace(SIDE_RULES),
    "front-facing": replace(FRONT_RULES, max_corner_angle=160.0, front_approx_ratio=0.006),
    "general": replace(GENERAL_RULES),
}
# First-pass HSV tuning from the 2026-10-02 live frame for the 1920x1080
# Realtek USB camera. Its purple target is concentrated around H=126..141;
# max_corner_angle is retained for legacy helpers. The rewritten live front /
# general detector below uses its own broad 5..175-degree noise floor.
USB_0BDA_3035_SCENE_RULES = {
    scene: replace(rules, h_low=120, h_high=150, s_min=55, v_min=25,
                   max_corner_angle=165.0 if scene == "front-facing" else rules.max_corner_angle)
    for scene, rules in SCENE_RULES.items()
}
CAMERA_SCENE_RULES = {
    "lingyan": SCENE_RULES,
    "seekfree": SEEKFREE_SCENE_RULES,
    "usb0bda3035_1080p": USB_0BDA_3035_SCENE_RULES,
}


def nothing(_: int) -> None:
    pass


def open_camera(index: int, width: int, height: int, *, fourcc=None, fps=None, profile=None) -> cv2.VideoCapture:
    if profile is not None and profile.device_vid_pid:
        from camera_controls import configure_controls
        configure_controls(index, profile.device_vid_pid)
    backend = profile.backend if profile is not None else "auto"
    api = cv2.CAP_MSMF if backend == "msmf" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened() and backend == "auto":
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Setting resolution/FPS can rebuild the DSHOW media type, so FOURCC is last.
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if profile is not None and (profile.fixed_controls or profile.automatic_controls):
        try:
            state = configure_controls(index, profile.device_vid_pid, profile.fixed_controls,
                                       profile.automatic_controls)
            print(f"fixed camera controls verified: {profile.device_vid_pid}, "
                  f"{profile.fixed_controls}; automatic={profile.automatic_controls or {}}", flush=True)
        except Exception:
            cap.release()
            raise
    return cap


def find_largest_purple(mask: np.ndarray, min_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0
        detections.append((area, x, y, w, h, cx, cy, contour))
    detections.sort(reverse=True)
    return detections


def effective_brightness_thresholds(
    hsv: np.ndarray,
    rules: SceneRules,
    purple_v_setting: int,
    white_v_setting: int,
) -> tuple[int, int]:
    """Adapt only the frontal scene to dim floor illumination."""
    if not rules.adaptive_exposure:
        return purple_v_setting, white_v_setting
    height, width = hsv.shape[:2]
    floor = hsv[height // 2:int(0.83 * height), int(0.03 * width):int(0.97 * width)]
    floor_values = floor[:, :, 2][floor[:, :, 1] < 80]
    if floor_values.size < 0.10 * floor.shape[0] * floor.shape[1]:
        return purple_v_setting, white_v_setting
    floor_brightness = float(np.median(floor_values))
    exposure = float(np.clip((floor_brightness - 110.0) / 70.0, 0.0, 1.0))
    purple_v_min = round(purple_v_setting * (0.50 + 0.50 * exposure))
    white_v_min = round(white_v_setting * (0.80 + 0.20 * exposure))
    return int(purple_v_min), int(white_v_min)


def select_safety_component_ranks(
    detections: list,
    rules: SceneRules,
    image_shape: tuple[int, ...],
    *,
    minimum_area: Optional[float] = None,
) -> set[int]:
    """Keep nearby purple fragments above an absolute noise-area floor."""
    if not detections:
        return set()
    selected = {1}
    if not rules.allow_split_components:
        return selected
    # A distant narrow side can be much smaller than the near side of the same
    # frame. Area only removes speckle; white support and fitted geometry are
    # validated later. The caller can supply the current slider values.
    if minimum_area is None:
        minimum_area = max(float(rules.min_area), float(rules.corner_min_area))
    candidates = {
        rank for rank, detection in enumerate(detections[1:12], start=2)
        if detection[0] >= minimum_area
    }

    def adjacent(first: tuple, second: tuple) -> bool:
        _, ax, ay, aw, ah, *_ = first
        _, bx, by, bw, bh, *_ = second
        horizontal_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
        vertical_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
        # Splitting by an occluder is not limited to horizontal image pieces.
        # Candidate ownership remains anchored at the largest purple region;
        # the geometry stage independently validates every proposed merge.
        extent = max(aw, ah, bw, bh)
        return float(np.hypot(horizontal_gap, vertical_gap)) <= max(24., .35 * extent)

    # A block can split one long purple frame into more than two pieces.
    # Expand through neighbouring large pieces rather than comparing every
    # piece only with the largest one.
    while candidates:
        newly_selected = {
            rank for rank in candidates
            if any(adjacent(detections[rank - 1], detections[member - 1])
                   for member in selected)
        }
        if not newly_selected:
            break
        selected.update(newly_selected)
        candidates.difference_update(newly_selected)
    return selected


def make_white_field_mask(hsv: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    """Keep bright floor colors while excluding saturated warm objects."""
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # The field can appear dim and nearly neutral in the distance. Permit a
    # small brightness margin only for low-saturation pixels; coloured blocks
    # still have to pass the full V threshold.
    bright = (value >= v_min) & (saturation <= s_max)
    dim_neutral = (value >= max(0, v_min - 20)) & (saturation <= min(s_max, 60))
    neutral_or_cool = (saturation <= 50) | ((hue >= 40) & (hue <= 150))
    return ((bright | dim_neutral) & neutral_or_cool).astype(np.uint8) * 255


def make_white_core_mask(hsv: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    """Neutral colour evidence for the adjacent strip in front/general modes.

    Keep the loose field segmentation for mixed edge pixels; require a neutral
    core within the same contiguous strip so a pale blue face cannot support it
    merely by passing the loose S<=80 (or S<=179) threshold.
    """
    return make_white_field_mask(hsv, min(s_max, 50), v_min)


def region_mask_for_contour(
    contour: np.ndarray,
    image_shape: tuple[int, ...],
    purple_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Use the segmented purple mask, or reconstruct it for standalone calls."""
    if purple_mask is not None:
        return purple_mask
    region = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
    return region


def white_edge_support(
    start: np.ndarray,
    end: np.ndarray,
    white_mask: np.ndarray,
    purple_mask: np.ndarray,
    oriented_area: float,
) -> float:
    """Fraction of an edge whose first exterior nonpurple pixel is white."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return 0.0
    sample_count = max(8, int(length / 4.0))
    fractions = np.linspace(0.15, 0.85, sample_count)
    samples = start[None, :] + fractions[:, None] * direction[None, :]
    if oriented_area < 0.0:
        normal = np.array([-direction[1], direction[0]]) / length
    else:
        normal = np.array([direction[1], -direction[0]]) / length
    unresolved = np.ones(sample_count, dtype=bool)
    first_exterior_distance = np.zeros(sample_count, dtype=np.int32)
    image_height, image_width = white_mask.shape[:2]
    # ApproxPolyDP can pass a few pixels inside a thick purple strip. Walk
    # outward until reaching the FIRST nonpurple pixel; do not skip a dark or
    # coloured pixel to find white farther away.
    for distance in range(1, 17):
        sample_x = np.rint(samples[:, 0] + normal[0] * distance).astype(np.int32)
        sample_y = np.rint(samples[:, 1] + normal[1] * distance).astype(np.int32)
        in_bounds = (
            (sample_x >= 0) & (sample_x < image_width)
            & (sample_y >= 0) & (sample_y < image_height)
        )
        unresolved &= in_bounds
        active = np.flatnonzero(unresolved)
        if active.size == 0:
            break
        exterior = purple_mask[sample_y[active], sample_x[active]] == 0
        found = active[exterior]
        first_exterior_distance[found] = distance
        unresolved[found] = False

    # A one-pixel dark fringe often lies between a thresholded purple pixel
    # and the actual floor. Check a narrow band just beyond that fringe.
    # A black strip or a coloured block remains nonwhite throughout the band.
    valid = first_exterior_distance > 0
    white_votes = np.zeros(sample_count, dtype=np.int32)
    for offset in (2, 4, 6, 8):
        sample_x = np.rint(
            samples[:, 0] + normal[0] * (first_exterior_distance + offset)
        ).astype(np.int32)
        sample_y = np.rint(
            samples[:, 1] + normal[1] * (first_exterior_distance + offset)
        ).astype(np.int32)
        in_bounds = (
            valid & (sample_x >= 0) & (sample_x < image_width)
            & (sample_y >= 0) & (sample_y < image_height)
        )
        active = np.flatnonzero(in_bounds)
        white_votes[active] += (white_mask[sample_y[active], sample_x[active]] > 0)
    return float((white_votes >= 2).mean())


@dataclass
class ValidBoundarySegment:
    component_index: int
    segment_index: int
    segment_count: int
    start: np.ndarray
    end: np.ndarray
    length: float
    support: float
    boundary_path: np.ndarray
    fitted_start: np.ndarray
    fitted_end: np.ndarray
    fitted_origin: np.ndarray
    fitted_direction: np.ndarray


@dataclass
class MergedBoundaryLine:
    members: list[ValidBoundarySegment]
    fitted_start: np.ndarray
    fitted_end: np.ndarray
    fitted_origin: np.ndarray
    fitted_direction: np.ndarray
    length: float
    support: float


def extract_valid_boundary_segments(
    contour: np.ndarray,
    white_mask: np.ndarray,
    purple_mask: Optional[np.ndarray],
    minimum_white_support: float,
    approximation_ratio: float,
    component_index: int = 0,
) -> list[ValidBoundarySegment]:
    """Build straight-line fits from valid portions of the real blue contour."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return []
    region_mask = region_mask_for_contour(contour, white_mask.shape, purple_mask)
    oriented_area = cv2.contourArea(contour, oriented=True)
    points = cv2.approxPolyDP(
        contour, approximation_ratio * perimeter, True,
    ).reshape(-1, 2).astype(np.float32)
    if len(points) < 2:
        return []
    contour_points = contour.reshape(-1, 2).astype(np.int32)
    contour_float = contour_points.astype(np.float32)
    contour_indices = [
        int(np.argmin(np.sum((contour_float - point) ** 2, axis=1)))
        for point in points
    ]
    segments: list[ValidBoundarySegment] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            continue
        support = white_edge_support(
            start, end, white_mask, region_mask, oriented_area,
        )
        if support < minimum_white_support:
            continue
        start_index = contour_indices[index]
        end_index = contour_indices[(index + 1) % len(points)]
        if start_index <= end_index:
            boundary_path = contour_points[start_index:end_index + 1]
        else:
            boundary_path = np.vstack(
                (contour_points[start_index:], contour_points[:end_index + 1])
            )
        if len(boundary_path) < 2:
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            boundary_path.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01,
        ).reshape(-1)
        direction = np.array([float(vx), float(vy)], dtype=np.float32)
        direction /= max(1e-6, float(np.linalg.norm(direction)))
        origin = np.array([float(x0), float(y0)], dtype=np.float32)
        start_projection = origin + np.dot(start - origin, direction) * direction
        end_projection = origin + np.dot(end - origin, direction) * direction
        segments.append(ValidBoundarySegment(
            component_index=component_index,
            segment_index=index,
            segment_count=len(points),
            start=start,
            end=end,
            length=length,
            support=support,
            boundary_path=boundary_path,
            fitted_start=start_projection,
            fitted_end=end_projection,
            fitted_origin=origin,
            fitted_direction=direction,
        ))
    return segments


def fitted_line_intersection(
    first: ValidBoundarySegment | MergedBoundaryLine,
    second: ValidBoundarySegment | MergedBoundaryLine,
) -> Optional[np.ndarray]:
    """Return the infinite-line intersection of two fitted valid segments."""
    cross = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    denominator = cross(first.fitted_direction, second.fitted_direction)
    if abs(denominator) < 1e-5:
        return None
    offset = second.fitted_origin - first.fitted_origin
    parameter = cross(offset, second.fitted_direction) / denominator
    return first.fitted_origin + parameter * first.fitted_direction


def merge_collinear_valid_segments(
    segments: list[ValidBoundarySegment],
    image_shape: tuple[int, ...],
) -> list[MergedBoundaryLine]:
    """Join separated green pieces that support the same physical straight edge."""
    if not segments:
        return []
    image_height, image_width = image_shape[:2]
    max_angle_cosine = float(np.cos(np.deg2rad(10.0)))
    max_offset = max(6.0, 0.018 * min(image_width, image_height))
    max_gap = max(40.0, 0.16 * image_width)

    def build_line(members: list[ValidBoundarySegment]) -> MergedBoundaryLine:
        points = np.vstack([member.boundary_path for member in members]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01,
        ).reshape(-1)
        direction = np.array([float(vx), float(vy)], dtype=np.float32)
        direction /= max(1e-6, float(np.linalg.norm(direction)))
        origin = np.array([float(x0), float(y0)], dtype=np.float32)
        parameters = (points - origin) @ direction
        start = origin + float(parameters.min()) * direction
        end = origin + float(parameters.max()) * direction
        weighted_support = sum(member.support * member.length for member in members)
        total_length = sum(member.length for member in members)
        return MergedBoundaryLine(
            members=members,
            fitted_start=start,
            fitted_end=end,
            fitted_origin=origin,
            fitted_direction=direction,
            length=float(np.linalg.norm(end - start)),
            support=weighted_support / max(1e-6, total_length),
        )

    clusters: list[list[ValidBoundarySegment]] = []
    for segment in sorted(segments, key=lambda item: item.length, reverse=True):
        matching: Optional[list[ValidBoundarySegment]] = None
        for cluster in clusters:
            line = build_line(cluster)
            alignment = abs(float(np.dot(
                segment.fitted_direction, line.fitted_direction,
            )))
            if alignment < max_angle_cosine:
                continue
            normal = np.array(
                [-line.fitted_direction[1], line.fitted_direction[0]],
                dtype=np.float32,
            )
            perpendicular_offset = abs(float(np.dot(
                segment.fitted_origin - line.fitted_origin, normal,
            )))
            if perpendicular_offset > max_offset:
                continue
            line_interval = sorted((
                float(np.dot(line.fitted_start - line.fitted_origin, line.fitted_direction)),
                float(np.dot(line.fitted_end - line.fitted_origin, line.fitted_direction)),
            ))
            segment_interval = sorted((
                float(np.dot(segment.fitted_start - line.fitted_origin, line.fitted_direction)),
                float(np.dot(segment.fitted_end - line.fitted_origin, line.fitted_direction)),
            ))
            interval_gap = max(
                0.0,
                max(line_interval[0], segment_interval[0])
                - min(line_interval[1], segment_interval[1]),
            )
            if interval_gap <= max_gap:
                matching = cluster
                break
        if matching is None:
            clusters.append([segment])
        else:
            matching.append(segment)
    return [build_line(cluster) for cluster in clusters]


def draw_valid_boundary(
    image: np.ndarray,
    contour: np.ndarray,
    white_mask: Optional[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    purple_mask: Optional[np.ndarray] = None,
    minimum_white_support: float = 0.40,
    approximation_ratio: float = 0.008,
) -> None:
    """Draw purple contour segments that have white-field support in green."""
    if white_mask is None:
        return
    segments = extract_valid_boundary_segments(
        contour, white_mask, purple_mask,
        minimum_white_support, approximation_ratio,
    )
    for segment in segments:
        # Follow the actual segmented contour. A straight chord here could
        # cross a bend or an occlusion gap and leave the blue boundary.
        cv2.polylines(
            image,
            [segment.boundary_path.reshape(-1, 1, 2)],
            False,
            color,
            3,
            cv2.LINE_AA,
        )


def draw_fitted_valid_boundary(
    image: np.ndarray,
    contour: np.ndarray,
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    minimum_white_support: float,
    approximation_ratio: float,
    color: tuple[int, int, int] = (0, 165, 255),
) -> None:
    """Draw orange fitted lines derived only from green contour segments."""
    if white_mask is None:
        return
    for segment in extract_valid_boundary_segments(
        contour, white_mask, purple_mask,
        minimum_white_support, approximation_ratio,
    ):
        cv2.line(
            image,
            tuple(np.rint(segment.fitted_start).astype(np.int32)),
            tuple(np.rint(segment.fitted_end).astype(np.int32)),
            color,
            2,
            cv2.LINE_AA,
        )


def draw_completed_front_polygon(
    image: np.ndarray,
    contours: list[np.ndarray],
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    rules: SceneRules,
    color: tuple[int, int, int] = (0, 165, 255),
) -> None:
    """Draw fitted orange sides, bridging supported gaps into a polygon."""
    if white_mask is None or not contours:
        return
    segments: list[ValidBoundarySegment] = []
    for component_index, contour in enumerate(contours):
        segments.extend(extract_valid_boundary_segments(
            contour, white_mask, purple_mask,
            rules.display_white_support, rules.front_approx_ratio,
            component_index,
        ))
    minimum_structural_length = max(20.0, 0.06 * image.shape[1])
    lines = [
        line for line in merge_collinear_valid_segments(segments, image.shape)
        if line.length >= minimum_structural_length
    ]
    if not lines:
        return
    image_height, image_width = image.shape[:2]
    max_extension = max(30.0, 0.12 * image_width)
    max_endpoint_gap = max(30.0, 0.12 * image_width)
    completed_parameters: list[list[float]] = []
    for line in lines:
        completed_parameters.append([
            float(np.dot(line.fitted_start - line.fitted_origin, line.fitted_direction)),
            float(np.dot(line.fitted_end - line.fitted_origin, line.fitted_direction)),
        ])
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            sine = abs(float(
                first.fitted_direction[0] * second.fitted_direction[1]
                - first.fitted_direction[1] * second.fitted_direction[0]
            ))
            endpoint_gap = min(
                float(np.linalg.norm(first_endpoint - second_endpoint))
                for first_endpoint in (first.fitted_start, first.fitted_end)
                for second_endpoint in (second.fitted_start, second.fitted_end)
            )
            if endpoint_gap > max_endpoint_gap:
                continue
            # Opposite sides of the trapezoid can converge weakly in the
            # image. Only a genuinely nearby pair may form a shallower turn.
            close_endpoint = endpoint_gap <= max(10.0, 0.025 * image_width)
            if sine < 0.50 and not (close_endpoint and sine >= 0.25):
                continue
            point = fitted_line_intersection(first, second)
            if point is None:
                continue
            first_parameter = float(np.dot(
                point - first.fitted_origin, first.fitted_direction,
            ))
            second_parameter = float(np.dot(
                point - second.fitted_origin, second.fitted_direction,
            ))

            def extension(parameter: float, values: list[float]) -> float:
                low, high = sorted(values[:2])
                return max(0.0, low - parameter, parameter - high)

            if (extension(first_parameter, completed_parameters[first_index]) > max_extension
                    or extension(second_parameter, completed_parameters[second_index]) > max_extension):
                continue
            if (point[0] < -max_extension or point[0] > image_width + max_extension
                    or point[1] < -max_extension or point[1] > image_height + max_extension):
                continue
            completed_parameters[first_index].append(first_parameter)
            completed_parameters[second_index].append(second_parameter)
    for line, parameters in zip(lines, completed_parameters):
        start = line.fitted_origin + min(parameters) * line.fitted_direction
        end = line.fitted_origin + max(parameters) * line.fitted_direction
        cv2.line(
            image,
            tuple(np.rint(start).astype(np.int32)),
            tuple(np.rint(end).astype(np.int32)),
            color,
            2,
            cv2.LINE_AA,
        )


def find_inferred_corners(
    contour: np.ndarray,
    white_mask: Optional[np.ndarray],
    corner_min_side: int,
    visible_corners: list[tuple[np.ndarray, float]],
    purple_mask: Optional[np.ndarray] = None,
    minimum_white_support: float = 0.40,
) -> list[tuple[np.ndarray, float]]:
    """Infer an occluded corner from the intersection of two valid edges.

    Only green-qualified segments are used. Intersections at an already
    visible corner, intersections requiring a very long extrapolation, and
    intersections whose endpoint gap is too small are rejected.
    """
    if white_mask is None:
        return []
    region_mask = region_mask_for_contour(contour, white_mask.shape, purple_mask)
    oriented_area = cv2.contourArea(contour, oriented=True)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    _, _, width, height = cv2.boundingRect(contour)
    if perimeter < 20.0 or area <= 0.0:
        return []

    minimum_edge = max(
        float(corner_min_side),
        0.65 * min(width, height),
        0.012 * perimeter,
    )
    minimum_extension = max(8.0, 0.25 * min(width, height))
    # The hidden corner should be close to the visible component. A much
    # longer extrapolation usually comes from intersecting opposite edges.
    maximum_extension = max(20.0, 1.20 * min(width, height))
    approx = cv2.approxPolyDP(contour, 0.008 * perimeter, True)
    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return []

    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        length = float(np.linalg.norm(end - start))
        if length < minimum_edge:
            continue
        support = white_edge_support(start, end, white_mask, region_mask, oriented_area)
        if support >= minimum_white_support:
            segments.append((start, end, support))
    if len(segments) < 2:
        return []

    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    def point_line_intersection(
        start_a: np.ndarray,
        end_a: np.ndarray,
        start_b: np.ndarray,
        end_b: np.ndarray,
    ) -> Optional[tuple[np.ndarray, float, float]]:
        direction_a = end_a - start_a
        direction_b = end_b - start_b
        denominator = cross(direction_a, direction_b)
        if abs(denominator) < 1e-5:
            return None
        offset = start_b - start_a
        parameter_a = cross(offset, direction_b) / denominator
        parameter_b = cross(offset, direction_a) / denominator
        intersection = start_a + parameter_a * direction_a
        return intersection, float(parameter_a), float(parameter_b)

    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
    inferred: list[tuple[np.ndarray, float, float]] = []
    for first_index in range(len(segments)):
        start_a, end_a, support_a = segments[first_index]
        for second_index in range(first_index + 1, len(segments)):
            start_b, end_b, support_b = segments[second_index]
            result = point_line_intersection(start_a, end_a, start_b, end_b)
            if result is None:
                continue
            intersection, parameter_a, parameter_b = result
            length_a = float(np.linalg.norm(end_a - start_a))
            length_b = float(np.linalg.norm(end_b - start_b))
            extension_a = max(0.0, -parameter_a * length_a, (parameter_a - 1.0) * length_a)
            extension_b = max(0.0, -parameter_b * length_b, (parameter_b - 1.0) * length_b)
            if (extension_a < minimum_extension or extension_b < minimum_extension or
                    extension_a > maximum_extension or extension_b > maximum_extension):
                continue
            if (intersection[0] < bbox_x - maximum_extension or
                    intersection[0] > bbox_x + bbox_w + maximum_extension or
                    intersection[1] < bbox_y - maximum_extension or
                    intersection[1] > bbox_y + bbox_h + maximum_extension):
                continue

            endpoint_gap = min(
                float(np.linalg.norm(first - second))
                for first in (start_a, end_a)
                for second in (start_b, end_b)
            )
            # Segments meeting at a visible corner have a tiny endpoint gap;
            # an occluded corner leaves a larger gap but it may still be
            # shorter than the edge-length gate used for corner detection.
            if endpoint_gap < minimum_extension:
                continue

            nearest_a = start_a if np.linalg.norm(intersection - start_a) <= np.linalg.norm(intersection - end_a) else end_a
            nearest_b = start_b if np.linalg.norm(intersection - start_b) <= np.linalg.norm(intersection - end_b) else end_b
            ray_a = nearest_a - intersection
            ray_b = nearest_b - intersection
            ray_norm = float(np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
            if ray_norm <= 1e-5:
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(ray_a, ray_b) / ray_norm, -1.0, 1.0))))
            if not 20.0 <= angle <= 160.0:
                continue
            if any(float(np.linalg.norm(intersection - visible)) <= max(10.0, 0.08 * perimeter)
                   for visible, _ in visible_corners):
                continue
            score = min(support_a, support_b) / (1.0 + extension_a + extension_b)
            inferred.append((intersection, angle, score))

    inferred.sort(key=lambda item: item[2], reverse=True)
    merged: list[tuple[np.ndarray, float]] = []
    merge_distance = max(8.0, min(16.0, 0.04 * perimeter))
    for point, angle, _ in inferred:
        if any(float(np.linalg.norm(point - existing)) <= merge_distance for existing, _ in merged):
            continue
        merged.append((np.rint(point).astype(np.int32), angle))
    return merged[:2]


def order_box_points(points: np.ndarray) -> np.ndarray:
    """Return image corners in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.int32)
    point_sum = pts.sum(axis=1)
    point_diff = pts[:, 0] - pts[:, 1]
    ordered[0] = np.rint(pts[np.argmin(point_sum)]).astype(np.int32)  # top-left
    ordered[1] = np.rint(pts[np.argmax(point_diff)]).astype(np.int32)  # top-right
    ordered[2] = np.rint(pts[np.argmax(point_sum)]).astype(np.int32)  # bottom-right
    ordered[3] = np.rint(pts[np.argmin(point_diff)]).astype(np.int32)  # bottom-left
    return ordered


def get_quad_points(contour: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    """Return four reliable contour vertices, or None if the shape is ambiguous."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0.0:
        for epsilon_ratio in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_box_points(approx.reshape(4, 2)), "contour"
    return None, "no-quad"


def find_front_fitted_visible_corners(
    contour: np.ndarray,
    corner_min_area: int,
    corner_min_side: int,
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    rules: SceneRules,
) -> list[tuple[np.ndarray, float]]:
    """Use intersections of adjacent fitted green lines as visible corners."""
    if white_mask is None:
        return []
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    _, _, width, height = cv2.boundingRect(contour)
    if (perimeter < 20.0 or area < corner_min_area
            or min(width, height) < corner_min_side):
        return []
    points = cv2.approxPolyDP(
        contour, rules.front_approx_ratio * perimeter, True,
    ).reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return []
    segments = extract_valid_boundary_segments(
        contour, white_mask, purple_mask,
        rules.min_white_support, rules.front_approx_ratio,
    )
    by_index = {segment.segment_index: segment for segment in segments}
    oriented_area = cv2.contourArea(contour, oriented=True)
    convex_cross_sign = 1.0 if oriented_area < 0.0 else -1.0
    region_mask = region_mask_for_contour(contour, white_mask.shape, purple_mask)
    minimum_edge = max(
        float(corner_min_side),
        rules.min_side_fraction * min(width, height),
        rules.min_perimeter_fraction * perimeter,
        20.0 * white_mask.shape[1] / 640.0,
    )
    image_height, image_width = white_mask.shape[:2]
    maximum_fit_offset = max(10.0, 0.03 * min(image_width, image_height))
    corners: list[tuple[np.ndarray, float]] = []
    for vertex_index, shared_vertex in enumerate(points):
        previous = by_index.get((vertex_index - 1) % len(points))
        following = by_index.get(vertex_index)
        if previous is None or following is None:
            continue
        if previous.length < minimum_edge or following.length < minimum_edge:
            continue
        raw_previous = previous.start - shared_vertex
        raw_following = following.end - shared_vertex
        cross_value = float(
            raw_previous[0] * raw_following[1]
            - raw_previous[1] * raw_following[0]
        )
        if cross_value * convex_cross_sign <= 0.0:
            continue
        intersection = fitted_line_intersection(previous, following)
        if intersection is None:
            continue
        if float(np.linalg.norm(intersection - shared_vertex)) > maximum_fit_offset:
            continue
        previous_local = shared_vertex + raw_previous * min(
            1.0, 18.0 / max(1.0, previous.length),
        )
        following_local = shared_vertex + raw_following * min(
            1.0, 18.0 / max(1.0, following.length),
        )
        if min(
            white_edge_support(
                previous_local, shared_vertex,
                white_mask, region_mask, oriented_area,
            ),
            white_edge_support(
                shared_vertex, following_local,
                white_mask, region_mask, oriented_area,
            ),
        ) < rules.min_white_support:
            continue
        first_ray = previous.fitted_start - intersection
        second_ray = following.fitted_end - intersection
        ray_norm = float(np.linalg.norm(first_ray) * np.linalg.norm(second_ray))
        if ray_norm < 1.0:
            continue
        angle = float(np.degrees(np.arccos(np.clip(
            np.dot(first_ray, second_ray) / ray_norm, -1.0, 1.0,
        ))))
        if not rules.min_corner_angle <= angle <= rules.max_corner_angle:
            continue
        if (intersection[0] < -2 or intersection[0] > image_width + 1
                or intersection[1] < -2 or intersection[1] > image_height + 1):
            continue
        corner = np.rint(intersection).astype(np.int32)
        corner[0] = int(np.clip(corner[0], 0, image_width - 1))
        corner[1] = int(np.clip(corner[1], 0, image_height - 1))
        corners.append((corner, angle))
    return corners


def find_front_connected_fitted_corners_across_components(
    contours: list[np.ndarray],
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    rules: SceneRules,
    corner_supports: Optional[dict] = None,
) -> list[tuple[np.ndarray, float]]:
    """Mark intersections whose fitted sides are joined by a short green turn."""
    if white_mask is None or not contours:
        return []
    segments: list[ValidBoundarySegment] = []
    for component_index, contour in enumerate(contours):
        segments.extend(extract_valid_boundary_segments(
            contour, white_mask, purple_mask,
            rules.min_white_support, rules.front_approx_ratio,
            component_index,
        ))
    image_height, image_width = white_mask.shape[:2]
    minimum_structural_length = max(20.0, 0.06 * image_width)
    lines = [
        line for line in merge_collinear_valid_segments(segments, white_mask.shape)
        if line.length >= minimum_structural_length
    ]
    segments_by_component: dict[int, dict[int, ValidBoundarySegment]] = {}
    for segment in segments:
        segments_by_component.setdefault(segment.component_index, {})[
            segment.segment_index
        ] = segment
    maximum_transition = max(24.0, 0.06 * image_width)
    maximum_endpoint_gap = max(30.0, 0.12 * image_width)

    def transition_length(
        first: ValidBoundarySegment,
        second: ValidBoundarySegment,
    ) -> float:
        if first.component_index != second.component_index:
            return float("inf")
        available = segments_by_component[first.component_index]
        count = first.segment_count
        best = float("inf")
        for step in (1, -1):
            current = (first.segment_index + step) % count
            total = 0.0
            while current != second.segment_index:
                segment = available.get(current)
                if segment is None:
                    break
                total += segment.length
                if total > maximum_transition:
                    break
                current = (current + step) % count
            else:
                best = min(best, total)
        return best

    def lines_have_short_green_turn(
        first: MergedBoundaryLine,
        second: MergedBoundaryLine,
    ) -> bool:
        return any(
            transition_length(first_member, second_member) <= maximum_transition
            for first_member in first.members
            for second_member in second.members
        )

    corners: list[tuple[np.ndarray, float]] = []
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1:]:
            if not lines_have_short_green_turn(first, second):
                continue
            endpoint_gap = min(
                float(np.linalg.norm(first_endpoint - second_endpoint))
                for first_endpoint in (first.fitted_start, first.fitted_end)
                for second_endpoint in (second.fitted_start, second.fitted_end)
            )
            if endpoint_gap > maximum_endpoint_gap:
                continue
            intersection = fitted_line_intersection(first, second)
            if intersection is None:
                continue
            first_endpoint = min(
                (first.fitted_start, first.fitted_end),
                key=lambda point: np.linalg.norm(point - intersection),
            )
            second_endpoint = min(
                (second.fitted_start, second.fitted_end),
                key=lambda point: np.linalg.norm(point - intersection),
            )
            if (float(np.linalg.norm(first_endpoint - intersection)) > maximum_endpoint_gap
                    or float(np.linalg.norm(second_endpoint - intersection)) > maximum_endpoint_gap):
                continue
            rays = supported_corner_rays(intersection, (first, second))
            if len(rays) != 2:
                continue
            angle = abs(normalize_heading_deg(rays[0] - rays[1]))
            if not rules.min_corner_angle <= angle <= rules.max_corner_angle:
                continue
            if (intersection[0] < -2 or intersection[0] > image_width + 1
                    or intersection[1] < -2 or intersection[1] > image_height + 1):
                continue
            point = np.rint(intersection).astype(np.int32)
            point[0] = int(np.clip(point[0], 0, image_width - 1))
            point[1] = int(np.clip(point[1], 0, image_height - 1))
            if any(float(np.linalg.norm(point - existing)) <= 15.0
                   for existing, _ in corners):
                continue
            corners.append((point, angle))
            if corner_supports is not None:
                corner_supports[tuple(map(int, point))] = (first, second)
    return corners


def find_effective_boundary_corners(
    contours: list[np.ndarray],
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    rules: SceneRules,
    corner_min_side: int = 12,
    corner_supports: Optional[dict] = None,
    boundary_result: Optional[dict] = None,
    point_geometry: Optional[PointGeometry] = None,
    white_core_mask: Optional[np.ndarray] = None,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    """Ordered purple/white observations -> multiple robust lines -> corners.

    The result is shared with rendering and identity, so both consume exactly
    the same accepted observations and supporting line groups.
    """
    if white_mask is None or not contours:
        return [], []
    scale = min(white_mask.shape[:2]) / 480.0
    config = BoundaryConfig(
        minimum_edge=max(3.0, float(corner_min_side)),
        minimum_white_support=rules.display_white_support,
        approximation_px=max(1.0, 2.4 * scale),
        fit_tolerance_px=max(1.5, 3.0 * scale),
        infer_corners=rules.infer_hidden_corners,
        point_geometry=point_geometry,
    )
    result = detect_boundaries(contours, white_mask, purple_mask, config,
                               white_core_mask=white_core_mask)
    if corner_supports is not None:
        corner_supports.update(result.corner_supports)
    if boundary_result is not None:
        boundary_result['result'] = result
    return result.visible, result.inferred


def find_visible_corners(
    contour: np.ndarray,
    corner_min_area: int,
    corner_min_side: int,
    white_mask: Optional[np.ndarray] = None,
    scene: str = "general",
    purple_mask: Optional[np.ndarray] = None,
    rules: Optional[SceneRules] = None,
) -> list[tuple[np.ndarray, float]]:
    """Find visible contour turns without requiring a complete quadrilateral.

    If ``white_mask`` is supplied, a candidate must also touch a sufficiently
    long purple-to-white boundary segment. This rejects purple mask breaks
    next to coloured blocks while retaining the visible outer safety-zone
    edge.
    """
    perimeter = cv2.arcLength(contour, True)
    rules = rules if rules is not None else SCENE_RULES[scene]
    area = cv2.contourArea(contour)
    _, _, width, height = cv2.boundingRect(contour)
    if (perimeter < 20.0 or area < corner_min_area or
            min(width, height) < corner_min_side):
        return []

    candidates: list[tuple[np.ndarray, float]] = []
    # RETR_EXTERNAL contours are normally clockwise on this path. Use the
    # contour orientation to keep convex outer corners and reject inward turns
    # caused by an occluding object or an internal notch.
    oriented_area = cv2.contourArea(contour, oriented=True)
    convex_cross_sign = 1.0 if oriented_area < 0.0 else -1.0
    region_mask = (
        region_mask_for_contour(contour, white_mask.shape, purple_mask)
        if white_mask is not None else None
    )
    # The launch view can show a large safety zone with one genuinely short
    # side at its near corner. Require white-field support on both adjacent
    # sides there instead of demanding two long projected sides.
    minimum_edge = max(
        float(corner_min_side),
        rules.min_side_fraction * min(width, height),
        rules.min_perimeter_fraction * perimeter,
    )
    if scene == "front-facing" and white_mask is not None:
        # A white occluder makes the cut end of a purple strip look like a
        # purple-white corner. Its short transverse edge is usually only the
        # strip thickness. Reserve shorter green edges for extrapolation;
        # directly observed front-view corners need two longer edges.
        minimum_edge = max(minimum_edge, 20.0 * white_mask.shape[1] / 640.0)
    # In the frontal scene, visible corners must be the literal shared
    # endpoint of the same green segments drawn by draw_valid_boundary().
    # Other scenes retain multiple scales for their earlier behaviour.
    approximation_ratios = (
        (rules.front_approx_ratio,) if scene == "front-facing"
        else (0.008, 0.012, 0.018, 0.025, 0.035)
    )
    for epsilon_ratio in approximation_ratios:
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        points = approx.reshape(-1, 2).astype(np.float32)
        if len(points) < 3:
            continue
        for index, point in enumerate(points):
            if white_mask is not None:
                image_height, image_width = white_mask.shape[:2]
                # A turn where the contour meets the image boundary is made
                # by cropping; its true second edge is not observed.
                if (point[0] <= 1 or point[0] >= image_width - 2 or
                        point[1] <= 1 or point[1] >= image_height - 2):
                    continue
            previous = points[(index - 1) % len(points)] - point
            following = points[(index + 1) % len(points)] - point
            previous_length = float(np.linalg.norm(previous))
            following_length = float(np.linalg.norm(following))
            if previous_length < minimum_edge or following_length < minimum_edge:
                continue
            cross = float(previous[0] * following[1] - previous[1] * following[0])
            if cross * convex_cross_sign <= 0.0:
                continue
            cosine = float(np.dot(previous, following) / (previous_length * following_length))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            # A safety-zone corner should be a strong turn. This deliberately
            # accepts a broad range because of perspective and lens distortion.
            # A far-away safety zone can become a very flat trapezoid in the
            # image. Its projected corners may be acute or obtuse rather than
            # close to 90 degrees.
            # In the launch view the raised frame can make a shallow bend
            # before the true corner leaves the image. Require a stronger
            # turn there; other scenes retain the wider perspective range.
            if rules.min_corner_angle <= angle <= rules.max_corner_angle:
                if white_mask is not None and region_mask is not None:
                    # The safety-zone edge is the purple line adjacent to
                    # the white field. Check the middle of both neighbouring
                    # segments and require one segment to have clear white
                    # support. A short obstruction break has no such segment.
                    previous_support = white_edge_support(
                        point + previous, point, white_mask, region_mask, oriented_area,
                    )
                    following_support = white_edge_support(
                        point, point + following, white_mask, region_mask, oriented_area,
                    )
                    if rules.require_both_white_edges:
                        if min(previous_support, following_support) < rules.min_white_support:
                            continue
                        if scene == "front-facing":
                            # Whole-edge support can hide an occluder at the
                            # vertex. A directly visible corner must also
                            # meet white floor on both edges immediately
                            # beside that vertex; otherwise it is a candidate
                            # for extrapolation instead.
                            local_previous = point + previous * min(1.0, 18.0 / previous_length)
                            local_following = point + following * min(1.0, 18.0 / following_length)
                            previous_local_support = white_edge_support(
                                local_previous, point, white_mask, region_mask, oriented_area,
                            )
                            following_local_support = white_edge_support(
                                point, local_following, white_mask, region_mask, oriented_area,
                            )
                            if min(previous_local_support, following_local_support) < rules.min_white_support:
                                continue
                    elif max(previous_support, following_support) < rules.min_white_support:
                        continue
                candidates.append((point, angle))

    # Merge detections from different approximation levels.
    merged: list[tuple[np.ndarray, float, int]] = []
    merge_distance = max(6.0, min(14.0, 0.035 * perimeter))
    for point, angle in candidates:
        matching = None
        for merged_index, (center, best_angle, count) in enumerate(merged):
            if float(np.linalg.norm(point - center)) <= merge_distance:
                matching = merged_index
                break
        if matching is None:
            merged.append((point.copy(), angle, 1))
        else:
            center, best_angle, count = merged[matching]
            new_center = (center * count + point) / (count + 1)
            new_angle = angle if abs(angle - 90.0) < abs(best_angle - 90.0) else best_angle
            merged[matching] = (new_center, new_angle, count + 1)

    # Large safety-zone components can still produce a valid corner at only one
    # approximation scale when part of an edge is hidden. Small components must
    # be supported at multiple scales to avoid marking speckle as a corner.
    minimum_support = 1 if area >= 2.0 * corner_min_area else 2
    return [
        (np.rint(center).astype(np.int32), angle)
        for center, angle, count in merged
        if count >= minimum_support
    ]


def select_scene_corners(
    corners: list[tuple[np.ndarray, float]],
    scene: str,
    image_shape: tuple[int, ...],
    rules: Optional[SceneRules] = None,
) -> list[tuple[np.ndarray, float]]:
    """Resolve nearby structural turns using this scene's own thresholds.

    At either end of the safety zone, a frame can create an upper turn near
    the ground corner. Keep the lower image point from a local group. The
    operation is symmetric in image x; each scene owns its grouping range.
    """
    rules = rules if rules is not None else SCENE_RULES[scene]
    if not rules.merge_structural_turns or len(corners) < 2:
        return corners

    # Candidate proximity is measured in image pixels, independent of the
    # safety-zone contour's width, height, and aspect ratio.
    image_height, image_width = image_shape[:2]
    horizontal_gap = max(10.0, rules.merge_x_fraction * image_width)
    vertical_gap = max(12.0, rules.merge_y_fraction * image_height)
    groups: list[list[tuple[np.ndarray, float]]] = []
    for corner in sorted(corners, key=lambda item: int(item[0][0])):
        point = corner[0]
        matching = next(
            (
                group for group in groups
                if abs(float(point[0] - group[0][0][0])) <= horizontal_gap
                and abs(float(point[1] - group[0][0][1])) <= vertical_gap
            ),
            None,
        )
        if matching is None:
            groups.append([corner])
        else:
            matching.append(corner)

    selected = [max(group, key=lambda item: int(item[0][1])) for group in groups]
    return sorted(selected, key=lambda item: int(item[0][0]))


def resolve_front_corner_overlap(
    visible_corners: list[tuple[np.ndarray, float]],
    inferred_corners: list[tuple[np.ndarray, float]],
    image_shape: tuple[int, ...],
    rules: Optional[SceneRules] = None,
    corner_supports: Optional[dict] = None,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    """Return exactly one result for each physical frontal-corner region."""
    image_height, image_width = image_shape[:2]
    rules = rules if rules is not None else FRONT_RULES
    horizontal_gap = max(10.0, rules.merge_x_fraction * image_width)
    vertical_gap = max(12.0, rules.merge_y_fraction * image_height)

    def same_end(first: np.ndarray, second: np.ndarray) -> bool:
        return (abs(float(first[0] - second[0])) <= horizontal_gap
                and abs(float(first[1] - second[1])) <= vertical_gap)

    def same_support_pair(first, second) -> bool:
        def same_line(a, b):
            start_a = np.asarray(a.fitted_start, dtype=float)
            start_b = np.asarray(b.fitted_start, dtype=float)
            direction_a = np.asarray(a.fitted_end, dtype=float) - start_a
            direction_b = np.asarray(b.fitted_end, dtype=float) - start_b
            length_a = np.linalg.norm(direction_a)
            length_b = np.linalg.norm(direction_b)
            if min(length_a, length_b) < 1.0:
                return False
            direction_a /= length_a
            direction_b /= length_b
            if abs(float(np.dot(direction_a, direction_b))) < np.cos(np.deg2rad(10.0)):
                return False
            offset = start_b - start_a
            perpendicular = abs(float(offset[0]*direction_a[1] - offset[1]*direction_a[0]))
            return perpendicular <= max(6.0, 0.018 * min(image_width, image_height))
        return ((same_line(first[0], second[0]) and same_line(first[1], second[1]))
                or (same_line(first[0], second[1]) and same_line(first[1], second[0])))

    def same_corner(first, second) -> bool:
        if not same_end(first[1], second[1]):
            return False
        if first[0] == second[0] == "visible" and corner_supports is not None:
            first_pair = corner_supports.get(tuple(map(int, first[1])))
            second_pair = corner_supports.get(tuple(map(int, second[1])))
            if first_pair is not None and second_pair is not None:
                # Adjacent real corners can share one side and still be close
                # in a distant view. Merge only the same pair of physical sides.
                return same_support_pair(first_pair, second_pair)
        return True

    tagged = [
        ("visible", point, angle) for point, angle in visible_corners
    ] + [
        ("inferred", point, angle) for point, angle in inferred_corners
    ]
    groups: list[list[tuple[str, np.ndarray, float]]] = []
    for candidate in sorted(tagged, key=lambda item: int(item[1][0])):
        matching = next(
            (
                group for group in groups
                if any(same_corner(candidate, member) for member in group)
            ),
            None,
        )
        if matching is None:
            groups.append([candidate])
        else:
            matching.append(candidate)

    kept_visible: list[tuple[np.ndarray, float]] = []
    kept_inferred: list[tuple[np.ndarray, float]] = []
    for group in groups:
        inferred_group = [item for item in group if item[0] == "inferred"]
        if inferred_group:
            _, point, angle = max(inferred_group, key=lambda item: int(item[1][1]))
            kept_inferred.append((point, angle))
        else:
            _, point, angle = max(group, key=lambda item: int(item[1][1]))
            kept_visible.append((point, angle))
    return kept_visible, kept_inferred


def select_position_references(
    visible_corners: list[tuple[np.ndarray, float]],
    inferred_corners: list[tuple[np.ndarray, float]],
    image_height: int,
    near_band_percent: int,
) -> set[tuple[int, int]]:
    """Select every corner within a relative near band of the lowest one.

    With this fixed camera view, larger image y is a provisional indicator of
    a closer ground point. The band is relative to the nearest detected corner,
    so more than one corner can qualify in the same frame.
    """
    points = [point for point, _ in visible_corners + inferred_corners]
    if not points:
        return set()
    lowest_y = max(int(point[1]) for point in points)
    near_band_pixels = max(0.0, image_height * near_band_percent / 100.0)
    return {
        (int(point[0]), int(point[1]))
        for point in points
        if lowest_y - int(point[1]) <= near_band_pixels
    }


def find_front_occluded_corners_across_components(
    contours: list[np.ndarray],
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    visible_corners: list[tuple[np.ndarray, float]],
    rules: SceneRules,
    corner_supports: Optional[dict] = None,
) -> list[tuple[np.ndarray, float]]:
    """Intersect valid front edges even when an occluder splits the frame."""
    if white_mask is None or not contours:
        return []
    x, y, width, height = cv2.boundingRect(np.vstack(contours))

    segments: list[ValidBoundarySegment] = []
    for component_index, contour in enumerate(contours):
        segments.extend(extract_valid_boundary_segments(
            contour, white_mask, purple_mask,
            rules.display_white_support, rules.front_approx_ratio,
            component_index,
        ))

    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    valid_indices_by_component: dict[int, set[int]] = {}
    segment_counts: dict[int, int] = {}
    for segment in segments:
        valid_indices_by_component.setdefault(segment.component_index, set()).add(
            segment.segment_index,
        )
        segment_counts[segment.component_index] = segment.segment_count

    def green_segments_connected(
        first: ValidBoundarySegment,
        second: ValidBoundarySegment,
    ) -> bool:
        """True when two segments belong to one unbroken green polyline."""
        if first.component_index != second.component_index:
            return False
        valid_indices = valid_indices_by_component[first.component_index]
        segment_count = segment_counts[first.component_index]
        pending = [first.segment_index]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current == second.segment_index:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbour in (
                (current - 1) % segment_count,
                (current + 1) % segment_count,
            ):
                if neighbour in valid_indices and neighbour not in visited:
                    pending.append(neighbour)
        return False

    lines = merge_collinear_valid_segments(segments, white_mask.shape)

    def green_lines_connected(
        first: MergedBoundaryLine,
        second: MergedBoundaryLine,
    ) -> bool:
        return any(
            green_segments_connected(first_member, second_member)
            for first_member in first.members
            for second_member in second.members
        )

    candidates: list[tuple[np.ndarray, float, float, MergedBoundaryLine, MergedBoundaryLine]] = []
    max_extension = max(40.0, 0.80 * height)
    for near_index, near_segment in enumerate(lines):
        near_start = near_segment.fitted_start
        near_end = near_segment.fitted_end
        near_length = near_segment.length
        near_support = near_segment.support
        near_direction = near_end - near_start
        if (near_length < max(20.0, 0.25 * width)
                or abs(float(near_direction[1])) > 0.35 * near_length):
            continue
        for side_index, side_segment in enumerate(lines):
            if side_index == near_index:
                continue
            # A continuous green polyline already contains its true turn.
            # Only a real gap may produce a magenta inferred corner.
            if green_lines_connected(near_segment, side_segment):
                continue
            side_start = side_segment.fitted_start
            side_end = side_segment.fitted_end
            side_length = side_segment.length
            side_support = side_segment.support
            side_direction = side_end - side_start
            denominator = cross(side_direction, near_direction)
            if abs(denominator) < 0.34 * side_length * near_length:
                continue
            offset = near_start - side_start
            side_parameter = cross(offset, near_direction) / denominator
            near_parameter = cross(offset, side_direction) / denominator
            point = side_start + side_parameter * side_direction
            side_extension = max(
                0.0,
                -side_parameter * side_length,
                (side_parameter - 1.0) * side_length,
            )
            near_extension = max(
                0.0,
                -near_parameter * near_length,
                (near_parameter - 1.0) * near_length,
            )
            if side_extension > max_extension or near_extension > max_extension:
                continue
            if max(side_extension, near_extension) < 8.0:
                continue
            short_gap = (
                near_extension <= max(6.0, 0.15 * near_length)
                and (-0.15 <= near_parameter <= 0.20
                     or 0.80 <= near_parameter <= 1.15)
            )
            # A wide occluder can hide the near-edge endpoint too. Permit
            # both lines to extend farther only when each has substantial
            # visible support and the near edge is mostly purple-white.
            wide_gap = (
                side_length >= max(35.0, 0.60 * height)
                and near_length >= max(30.0, 0.30 * width)
                and side_support >= rules.min_white_support
                and near_support >= 0.75
                and near_extension <= min(max_extension, 0.60 * near_length)
                and (-0.60 <= near_parameter <= 0.20
                     or 0.80 <= near_parameter <= 1.60)
            )
            if not (short_gap or wide_gap):
                continue
            side_endpoint = min((side_start, side_end), key=lambda q: np.linalg.norm(q - point))
            near_endpoint = min((near_start, near_end), key=lambda q: np.linalg.norm(q - point))
            # If the near edge already ends at a directly observed corner,
            # extending a different slanted segment past it creates a second,
            # false corner beside the real one.
            if any(float(np.linalg.norm(near_endpoint - visible)) < 12.0
                   for visible, _ in visible_corners):
                continue
            if float(np.linalg.norm(side_endpoint - near_endpoint)) > max_extension:
                continue
            if (point[0] < x - max_extension or point[0] > x + width + max_extension
                    or point[1] < y + 0.45 * height or point[1] > y + height + max_extension):
                continue
            if (point[0] <= 2 or point[0] >= white_mask.shape[1] - 3
                    or point[1] <= 2 or point[1] >= white_mask.shape[0] - 3):
                continue
            if any(float(np.linalg.norm(point - visible)) < 12.0
                   for visible, _ in visible_corners):
                continue
            rays = supported_corner_rays(point, (side_segment, near_segment))
            if len(rays) != 2:
                continue
            angle = abs(normalize_heading_deg(rays[0] - rays[1]))
            if not 20.0 <= angle <= 160.0:
                continue
            score = min(side_support, near_support) / (1.0 + side_extension + near_extension)
            candidates.append((point, angle, score, side_segment, near_segment))

    candidates.sort(key=lambda item: item[2], reverse=True)
    inferred: list[tuple[np.ndarray, float]] = []
    for point, angle, _, side_segment, near_segment in candidates:
        if any(float(np.linalg.norm(point - existing)) <= 15.0 for existing, _ in inferred):
            continue
        rounded = np.rint(point).astype(np.int32)
        inferred.append((rounded, angle))
        if corner_supports is not None:
            corner_supports[tuple(map(int, rounded))] = (side_segment, near_segment)
        if len(inferred) == 2:
            break
    return inferred[:2]


def find_front_occluded_corners(
    contour: np.ndarray,
    white_mask: Optional[np.ndarray],
    purple_mask: Optional[np.ndarray],
    visible_corners: list[tuple[np.ndarray, float]],
    rules: SceneRules,
) -> list[tuple[np.ndarray, float]]:
    """Compatibility wrapper for one connected purple component."""
    return find_front_occluded_corners_across_components(
        [contour], white_mask, purple_mask, visible_corners, rules,
    )


def find_scene_inferred_corners(
    scene: str,
    contour: np.ndarray,
    white_mask: Optional[np.ndarray],
    corner_min_side: int,
    visible_corners: list[tuple[np.ndarray, float]],
    purple_mask: Optional[np.ndarray] = None,
    rules: Optional[SceneRules] = None,
) -> list[tuple[np.ndarray, float]]:
    """Use extrapolated corners only in scenes that permit inference."""
    rules = rules if rules is not None else SCENE_RULES[scene]
    if scene == "front-facing":
        return find_front_occluded_corners(
            contour, white_mask, purple_mask, visible_corners, rules,
        )
    if not rules.infer_hidden_corners:
        return []
    return find_inferred_corners(
        contour,
        white_mask,
        corner_min_side,
        visible_corners,
        purple_mask,
        rules.min_white_support,
    )


def identify_frame_corners(recognitions, safety_ranks, white_mask, mask, rules,
                           templates, model, scene, pose=None, calibrated_ranger=None,
                           corner_supports=None):
    """Attach map IDs only to the detector's accepted visible/inferred corners."""
    segments_by_rank = {}
    use_supports = scene in ("front-facing", "general") and corner_supports is not None
    if model is not None and white_mask is not None and not use_supports:
        for rank, (detection, _, _) in enumerate(recognitions, 1):
            if rank in safety_ranks:
                segments_by_rank[rank] = extract_valid_boundary_segments(
                    detection[-1], white_mask, mask, rules.display_white_support,
                    rules.front_approx_ratio, component_index=rank)
    observations = []
    all_segments = [s for group in segments_by_rank.values() for s in group]
    seen = set()
    for rank, (_, visible, inferred) in enumerate(recognitions, 1):
        for is_inferred, corners in ((False, visible), (True, inferred)):
            for point, _ in corners:
                key = (int(point[0]), int(point[1]))
                if key in seen:
                    continue
                seen.add(key)
                segments = all_segments if is_inferred and scene == "front-facing" else segments_by_rank.get(rank, [])
                if use_supports:
                    rays = supported_corner_rays(point, corner_supports.get(key, ())) if model else ()
                else:
                    rays = directed_rays(point, segments, is_inferred, mask.shape[1], model) if model else ()
                observations.append(CornerObservation(tuple(map(float, point)), is_inferred, rays, rank))
    return associate_corners(observations, templates, model, pose=pose, image_width=mask.shape[1],
                             calibrated_ranger=calibrated_ranger)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=CAMERA_PROFILES, default="lingyan",
                        help="camera-specific capture geometry and recognition rules")
    parser.add_argument("--index", type=int)
    parser.add_argument("--ranging", action="store_true", help="enable calibrated body-frame corner ranging")
    parser.add_argument("--target-z-mm", type=float, help="required known corner height when ranging; ground=0")
    parser.add_argument("--range-max-mm", type=float, default=DEFAULT_MAX_DISTANCE_MM,
                        help="maximum horizontal body-origin corner range; default 1000 mm")
    parser.add_argument("--range-inferred", action="store_true", help="also range inferred corners, explicitly labelled estimates")
    parser.add_argument("--association-output", type=Path,
                        help="read-only simulated pose / corner identity / range feedback file")
    parser.add_argument("--width", type=int, default=None,
                        help="override the camera profile's capture width")
    parser.add_argument("--height", type=int, default=None,
                        help="override the camera profile's capture height")
    parser.add_argument("--rotate", choices=ROTATION_CHOICES, default=None,
                        help="override the camera profile's image rotation")
    parser.add_argument(
        "--scene",
        choices=("auto", "front-facing", "start-facing", "general"),
        default="auto",
        help="auto selects front/side detector from pose templates; explicit scenes retain manual tuning",
    )
    parser.add_argument(
        "--pose-file",
        default="selected_pose.json",
        help="JSON pose produced by pose_map_gui.py",
    )
    parser.add_argument(
        "--zones-file",
        default="field_zones.json",
        help="named recognition zones produced by pose_map_gui.py",
    )
    parser.add_argument("--templates-file", default=str(DEFAULT_TEMPLATES),
                        help="safety-zone numbering and directed-edge templates")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="launcher session stop signal; normally managed by run_app.py")
    parser.add_argument("--parameters", action="store_true", help="show the detachable HSV parameter window at startup")
    parser.add_argument("--bottom-edge-test", action=argparse.BooleanOptionalAction, default=True,
                        help="experimental post-detection upper/lower classification (this test branch defaults on)")
    parser.add_argument("--bottom-edge-tolerance-mm", type=int, default=30,
                        help="bottom length/width matching tolerance, also adjustable in the parameter window")
    parser.add_argument('--ground-association', action=argparse.BooleanOptionalAction, default=True,
                        help='associate lower corners using a bounded simulated pose and calibrated view')
    parser.add_argument('--pose-radius-mm', type=int, default=200)
    parser.add_argument('--ground-visible-tol-mm', type=int, default=30)
    parser.add_argument('--ground-inferred-tol-mm', type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.bottom_edge_tolerance_mm <= 200:
        parser.error("--bottom-edge-tolerance-mm must be between 0 and 200")
    bottom_config = BottomEdgeConfig(tolerance_mm=args.bottom_edge_tolerance_mm)
    if not 0 <= args.pose_radius_mm <= 3000:
        parser.error('--pose-radius-mm must be between 0 and 3000')
    if not all(0 <= value <= 200 for value in (args.ground_visible_tol_mm, args.ground_inferred_tol_mm)):
        parser.error('ground measurement tolerances must be between 0 and 200 mm')
    association_writer = None
    association_report = None
    if args.association_output is not None:
        if not args.ranging:
            parser.error("--association-output requires --ranging")
        from pose_association import association_output_path, AssociationWriter
        association_writer = AssociationWriter(association_output_path(
            args.camera, args.association_output, args.pose_file))
    ranger = None
    if args.ranging:
        if args.target_z_mm is None:
            parser.error("--ranging requires an explicit --target-z-mm (ground corners: 0)")
        from corner_ranging import CornerRanger
        ranger = CornerRanger.from_camera(args.camera, target_z_mm=args.target_z_mm,
                                         max_distance_mm=args.range_max_mm,
                                         include_inferred=args.range_inferred)
        print(f"body ranging enabled: camera={args.camera}, Z={args.target_z_mm:g}mm, "
              f"limit={args.range_max_mm:g}mm; inferred={args.range_inferred}", flush=True)
    bottom_ranger = None
    if args.bottom_edge_test and ranger is not None and ranger.target_z_mm == bottom_config.ground_z_mm:
        from copy import copy
        bottom_ranger = copy(ranger)
        bottom_ranger.include_inferred = True
        print(f"bottom-edge layer test: all visible/inferred corners projected to Z=0mm; "
              f"L={bottom_config.length_mm:g} W={bottom_config.width_mm:g}mm, "
              f"tolerance=+/-{bottom_config.tolerance_mm:g}mm", flush=True)
    requested_scene = args.scene
    ground_view = (CalibratedGroundView(bottom_ranger)
                   if args.ground_association and bottom_ranger is not None else None)
    scene = requested_scene if requested_scene != "auto" else "front-facing"
    camera_profile = CAMERA_PROFILES[args.camera]
    camera_index = args.index if args.index is not None else camera_profile.default_index
    capture_width = args.width if args.width is not None else camera_profile.capture_width
    capture_height = args.height if args.height is not None else camera_profile.capture_height
    rotation = args.rotate if args.rotate is not None else camera_profile.rotation
    if (capture_width, capture_height, rotation) != (
        camera_profile.capture_width, camera_profile.capture_height, camera_profile.rotation
    ):
        raise ValueError("capture overrides differ from registered profile; select/register the matching camera mode")
    rules_for_camera = CAMERA_SCENE_RULES.get(args.camera, SCENE_RULES)
    rules = rules_for_camera[scene]

    point_geometry = None
    try:
        _, intrinsic_data = load_intrinsics(args.camera)
        point_geometry = PointGeometry.from_intrinsics(intrinsic_data)
        intrinsic_status = "matched"
    except (FileNotFoundError, ValueError) as error:
        intrinsic_status = f"pending ({error})"
    try:
        load_mount(args.camera)
        mount_status = "matched"
    except (FileNotFoundError, ValueError) as error:
        mount_status = f"pending/invalid ({error})"
    print(f"selected camera calibration: intrinsics={intrinsic_status}; mount={mount_status}", flush=True)

    pose_reader = PoseFileReader(args.pose_file)
    zone_reader = ZoneFileReader(args.zones_file)
    template_reader = TemplateReader(args.templates_file)
    template_model = template_reader.poll()
    previous_zone_names: Optional[tuple[str, ...]] = None
    trackbar_fields = {
        "H low": "h_low", "H high": "h_high", "S min": "s_min", "V min": "v_min",
        "min area": "min_area", "corner min area": "corner_min_area",
        "corner min side": "corner_min_side", "white S max": "white_s_max",
        "white V min": "white_v_min", "near band %": "near_band_percent",
    }
    scene_settings = {}

    cap = open_camera(camera_index, capture_width, capture_height,
                      fourcc=camera_profile.pixel_format, fps=camera_profile.target_fps, profile=camera_profile)
    ok, initial_frame = cap.read()
    if not ok or initial_frame is None:
        cap.release()
        raise RuntimeError("camera opened but did not provide a frame")
    actual_height, actual_width = initial_frame.shape[:2]
    try:
        verify_capture_format(cap, camera_profile)
    except Exception:
        cap.release()
        raise
    if (actual_width, actual_height) != (capture_width, capture_height):
        cap.release()
        raise RuntimeError(
            f"camera returned {actual_width}x{actual_height}, expected "
            f"{capture_width}x{capture_height} for profile {args.camera}"
        )
    display_width = actual_height if rotation in ("clockwise", "counterclockwise") else actual_width
    display_height = actual_width if rotation in ("clockwise", "counterclockwise") else actual_height
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"camera={args.camera}, index={camera_index}, raw={actual_width}x{actual_height}, "
        f"rotation={rotation}, image={display_width}x{display_height}, "
        f"reported_fps={actual_fps:.2f}",
        flush=True,
    )

    window = "Camera - Purple Segmentation + Body Ranging" if ranger is not None else "Camera - Purple Segmentation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    preview_size = preview_dimensions((display_height, display_width))
    cv2.resizeWindow(window, *preview_size)
    parameter_values = {name: getattr(rules, field) for name, field in trackbar_fields.items()}
    if args.bottom_edge_test:
        parameter_values['bottom tol mm'] = args.bottom_edge_tolerance_mm
    if args.ground_association:
        parameter_values.update({'pose radius mm': args.pose_radius_mm,
                                 'map tol mm': args.ground_visible_tol_mm,
                                 'infer tol mm': args.ground_inferred_tol_mm})
    parameters = ParameterPanel(parameter_values, visible=args.parameters)

    last_time = time.perf_counter()
    shown_fps = 0.0
    frame_count = 0
    color_mode: Optional[bool] = None
    try:
        while True:
            if args.stop_file is not None and args.stop_file.exists():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                print("camera frame read failed", flush=True)
                break
            if frame.shape[:2] != (actual_height, actual_width):
                raise RuntimeError("camera resolution changed during recognition")
            frame = rotate_frame(frame, rotation)
            # Closing the preview with its title-bar X removes its trackbars.
            # Exit cleanly before reading them on the next iteration.
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            parameters.poll()
            frame_count += 1
            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                shown_fps = 0.9 * shown_fps + 0.1 / dt
            last_time = now
            selected_pose = pose_reader.poll()
            zones = zone_reader.poll()
            active_zones = matching_zones(selected_pose, zones) if selected_pose else []
            template_model = template_reader.poll()
            identity_templates, identity_status = active_templates(selected_pose, zones, template_model)
            center_templates = identity_templates
            if ground_view is not None and selected_pose is not None and template_model is not None:
                possible = ground_view.landmarks(selected_pose, parameters.get('pose radius mm'), template_model)
                safety_ids = {p['safety_id'] for p in possible if p['possibly_visible']}
                identity_templates = circle_templates(selected_pose, zones, template_model, safety_ids,
                                                      parameters.get('pose radius mm'))
                identity_status = 'circle-based calibrated view'
            if requested_scene == "auto":
                next_scene = choose_scene(identity_templates, choose_scene(center_templates, scene))
                if next_scene != scene:
                    scene_settings[scene] = {name: parameters.get(name) for name in trackbar_fields}
                    scene = next_scene
                    rules = rules_for_camera[scene]
                    settings = scene_settings.get(scene, {name: getattr(rules, field) for name, field in trackbar_fields.items()})
                    for name, value in settings.items():
                        parameters.set(name, value)
                    print(f"pose template selected detector: {scene}", flush=True)
            if identity_templates and any(t.scene != scene for t in identity_templates):
                identity_templates = []
                identity_status = "detector/template scene mismatch; use --scene auto"
            if rotation != camera_profile.rotation:
                identity_templates = []
                identity_status = "image rotation differs from identity template"
            active_zone_names = tuple(zone.name for zone in active_zones)
            if active_zone_names != previous_zone_names:
                print(
                    "识别分区：" + ("、".join(active_zone_names) if active_zone_names else "未匹配"),
                    flush=True,
                )
                previous_zone_names = active_zone_names

            color_mode = frame.ndim == 3 and frame.shape[2] >= 3
            white_mask: Optional[np.ndarray] = None
            white_core_mask: Optional[np.ndarray] = None
            if color_mode:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h_low = parameters.get("H low")
                h_high = parameters.get("H high")
                s_min = parameters.get("S min")
                v_setting = parameters.get("V min")
                white_s_max = parameters.get("white S max")
                white_v_setting = parameters.get("white V min")
                v_min, white_v_min = effective_brightness_thresholds(
                    hsv, rules, v_setting, white_v_setting,
                )
                if h_low <= h_high:
                    mask = cv2.inRange(hsv, (h_low, s_min, v_min), (h_high, 255, 255))
                else:
                    # Support a hue interval wrapping around 179.
                    m1 = cv2.inRange(hsv, (0, s_min, v_min), (h_high, 255, 255))
                    m2 = cv2.inRange(hsv, (h_low, s_min, v_min), (179, 255, 255))
                    mask = cv2.bitwise_or(m1, m2)
                white_mask = make_white_field_mask(hsv, white_s_max, white_v_min)
                if scene in ("front-facing", "general"):
                    white_core_mask = make_white_core_mask(hsv, white_s_max, white_v_min)
            else:
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Grayscale fallback can only find a brightness range; it cannot
                # distinguish purple from another color with the same luminance.
                mask = cv2.inRange(gray, 0, 120)
                white_mask = cv2.inRange(gray, 180, 255)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            min_area = max(1, parameters.get("min area"))
            corner_min_area = max(1, parameters.get("corner min area"))
            corner_min_side = max(3, parameters.get("corner min side"))
            near_band_percent = parameters.get("near band %")
            detections = find_largest_purple(mask, min_area)
            safety_ranks = select_safety_component_ranks(
                detections, rules, frame.shape,
                minimum_area=max(min_area, corner_min_area),
            )
            recognitions = []
            all_visible_corners = []
            all_inferred_corners = []
            frame_corner_supports = {}
            frame_boundary_result = {}
            for rank, detection in enumerate(detections[:12], start=1):
                contour = detection[-1]
                raw_corners = (
                    (
                        [] if scene in ("front-facing", "general") else find_visible_corners(
                            contour, corner_min_area, corner_min_side,
                            white_mask, scene, mask, rules,
                        )
                    )
                    if rank in safety_ranks else []
                )
                visible_corners = select_scene_corners(raw_corners, scene, frame.shape, rules)
                inferred_corners = (
                    [] if scene in ("front-facing", "general") else find_scene_inferred_corners(
                        scene, contour, white_mask,
                        corner_min_side, raw_corners, mask, rules,
                    )
                    if rank in safety_ranks else []
                )
                recognitions.append((detection, visible_corners, inferred_corners))
                all_visible_corners.extend(visible_corners)
                all_inferred_corners.extend(inferred_corners)
            if scene in ("front-facing", "general"):
                selected_indices = [
                    rank - 1 for rank in sorted(safety_ranks)
                    if 1 <= rank <= len(recognitions)
                ]
                selected_contours = [
                    recognitions[index][0][-1] for index in selected_indices
                ]
                all_visible_corners, all_inferred_corners = find_effective_boundary_corners(
                    selected_contours, white_mask, mask, rules, corner_min_side,
                    corner_supports=frame_corner_supports,
                    boundary_result=frame_boundary_result,
                    point_geometry=point_geometry,
                    white_core_mask=white_core_mask,
                )
                recognitions = [
                    (detection, [], [])
                    for detection, _, _ in recognitions
                ]
                for visible_corner in all_visible_corners:
                    point = visible_corner[0]
                    owner_index = max(
                        selected_indices,
                        key=lambda index: cv2.pointPolygonTest(
                            recognitions[index][0][-1],
                            (float(point[0]), float(point[1])),
                            True,
                        ),
                    )
                    recognitions[owner_index][1].append(visible_corner)
                for inferred_corner in all_inferred_corners:
                    point = inferred_corner[0]
                    owner_index = max(
                        selected_indices,
                        key=lambda index: cv2.pointPolygonTest(
                            recognitions[index][0][-1],
                            (float(point[0]), float(point[1])),
                            True,
                        ),
                    )
                    recognitions[owner_index][2].append(inferred_corner)
            reference_points = select_position_references(
                all_visible_corners, all_inferred_corners,
                frame.shape[0], near_band_percent,
            )

            if not color_mode:
                identity_templates = []
                identity_status = "color frame required"
            identities = identify_frame_corners(
                recognitions, safety_ranks, white_mask, mask, rules,
                identity_templates, template_model, scene, selected_pose,
                ranger if args.ranging and args.target_z_mm == 0 else None,
                corner_supports=frame_corner_supports if scene in ("front-facing", "general") else None,
            )
            bottom_result = None
            frame_ranger = ranger
            if args.bottom_edge_test and scene in ("front-facing", "general"):
                bottom_config = BottomEdgeConfig(tolerance_mm=parameters.get('bottom tol mm'))
                if bottom_ranger is not None:
                    frame_ranger = bottom_ranger
                    range_readings = bottom_ranger.measure(recognitions, identities)
                    bottom_result = classify_bottom_edges(
                        range_readings, frame_boundary_result.get('result'), bottom_config)
                else:
                    range_readings = ranger.measure(recognitions, identities) if ranger is not None else []
                    bottom_result = BottomEdgeResult(
                        bottom_config, 'ground_ranging_required',
                        len(all_visible_corners) + len(all_inferred_corners))
                # Replace only the provisional reference ring, never detections.
                reference_points = ({bottom_result.lower_seed}
                                    if bottom_result.lower_seed is not None else set())
            else:
                range_readings = ranger.measure(recognitions, identities) if ranger is not None else []
            ground_report = None
            if args.ground_association and bottom_result is not None:
                ground_config = GroundAssociationConfig(
                    prior_radius_mm=parameters.get('pose radius mm'),
                    visible_tolerance_mm=parameters.get('map tol mm'),
                    inferred_tolerance_mm=parameters.get('infer tol mm'))
                identities, ground_report = associate_ground_corners(
                    range_readings, bottom_result, selected_pose,
                    template_model if color_mode and rotation == camera_profile.rotation else None,
                    ground_view, ground_config)
                range_readings = apply_ground_identities(range_readings, identities)
                identity_status = ground_report['status']
            overlay = frame.copy()
            if scene in ("front-facing", "general"):
                cv2.drawContours(overlay, [d[0][-1] for d in recognitions], -1, (255, 0, 0), 2)
                if "result" in frame_boundary_result:
                    draw_boundary_result(overlay, frame_boundary_result["result"], labels=False)
            for rank, (detection, visible_corners, inferred_corners) in enumerate(recognitions, start=1):
                area, x, y, w, h, cx, cy, contour = detection
                # The blue contour is the actual boundary of the segmented mask.
                # Do not replace it with a larger external rectangle.
                if scene == "start-facing":
                    cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)
                # Green marks only the portions adjacent to the white field.
                if scene == "start-facing":
                    draw_valid_boundary(
                        overlay, contour, white_mask, purple_mask=mask,
                        minimum_white_support=rules.display_white_support,
                        approximation_ratio=rules.front_approx_ratio,
                    )
                for point, _ in visible_corners:
                    cv2.circle(overlay, tuple(map(int, point)), 6, (0, 255, 255), -1)
                for point, _ in inferred_corners:
                    cv2.circle(overlay, tuple(map(int, point)), 7, (255, 0, 255), -1)
            for px, py in reference_points:
                cv2.circle(overlay, (px, py), 11, (255, 255, 0), 3)
            if selected_pose is not None:
                pose_text = (
                    f"INPUT POSE x={selected_pose.x_mm:.1f} y={selected_pose.y_mm:.1f} "
                    f"heading={selected_pose.heading_deg:.1f} deg"
                )
            else:
                pose_text = "INPUT POSE waiting for selected_pose.json"
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if scene in ("front-facing", "general"):
                cv2.drawContours(mask_bgr, [d[0][-1] for d in recognitions], -1, (255, 0, 0), 2)
                if "result" in frame_boundary_result:
                    draw_boundary_result(mask_bgr, frame_boundary_result["result"], labels=False)
            # Draw the exact same blue contour on the binary view. The contour
            # coordinates are unchanged; only the display image differs.
            for rank, (detection, corners, inferred_corners) in enumerate(recognitions, 1):
                contour = detection[-1]
                if scene == "start-facing":
                    cv2.drawContours(mask_bgr, [contour], -1, (255, 0, 0), 2)
                if scene == "start-facing":
                    draw_valid_boundary(
                        mask_bgr, contour, white_mask, purple_mask=mask,
                        minimum_white_support=rules.display_white_support,
                        approximation_ratio=rules.front_approx_ratio,
                    )
                for point, _ in corners:
                    px, py = [int(value) for value in point]
                    cv2.circle(mask_bgr, (px, py), 5, (0, 255, 255), -1)
                for point, _ in inferred_corners:
                    px, py = [int(value) for value in point]
                    cv2.circle(mask_bgr, (px, py), 6, (255, 0, 255), -1)
            for px, py in reference_points:
                cv2.circle(mask_bgr, (px, py), 11, (255, 255, 0), 3)
            parameter_text = (
                f"HSV H={h_low}-{h_high} S>={s_min} V>={v_min} | White S<={white_s_max} V>={white_v_min}"
                if color_mode else "GRAYSCALE: colour segmentation unavailable"
            )
            preview = compose_preview(
                overlay, mask_bgr, recognitions, identities, range_readings,
                size=preview_size,
                title=f"{args.camera} | {scene} | FPS {shown_fps:.1f} | visible {len(all_visible_corners)} / estimated {len(all_inferred_corners)}",
                parameters=parameter_text, pose_text=pose_text, status=identity_status,
                boundary_result=frame_boundary_result.get("result"), ranger=frame_ranger,
                bottom_edge_result=bottom_result,
                ground_association=ground_report,
            )
            if ranger is not None:
                from pose_association import association_payload
                association_report = association_payload(args.camera, frame_count, frame_ranger, range_readings,
                                                         selected_pose, identities, identity_templates,
                                                         scene, args.pose_file)
                if bottom_result is not None:
                    association_report['bottom_edge_test'] = bottom_result.payload()
                if ground_report is not None:
                    association_report['ground_association'] = ground_report
                if association_writer is not None:
                    association_writer.write(association_report)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")):
                parameters.toggle()
            if key == ord("s"):
                output_dir = Path(__file__).resolve().parent / "照片"
                output_dir.mkdir(parents=True, exist_ok=True)
                raw_path = output_dir / f"latest_live_raw_{args.camera}.png"
                preview_path = output_dir / f"latest_live_preview_{args.camera}.png"
                if ranger is not None:
                    from corner_ranging import save_range_image
                    save_range_image(raw_path, frame)
                    save_range_image(preview_path, preview)
                else:
                    cv2.imwrite(str(raw_path), frame)
                    cv2.imwrite(str(preview_path), preview)
                print(f"saved {raw_path} and {preview_path}", flush=True)
                if ranger is not None:
                    from corner_ranging import range_payload
                    from calibration.calibration_io import save_result_json
                    from datetime import datetime, timezone
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snapshot = output_dir / f"corner_ranges_{args.camera}_{stamp}.json"
                    snapshot_raw = snapshot.with_suffix(".raw.png")
                    snapshot_preview = snapshot.with_suffix(".preview.png")
                    save_range_image(snapshot_raw, frame)
                    save_range_image(snapshot_preview, preview)
                    payload = range_payload(args.camera, frame_count, frame_ranger, range_readings)
                    if bottom_result is not None:
                        payload['bottom_edge_test'] = bottom_result.payload()
                    if ground_report is not None:
                        payload['ground_association'] = ground_report
                    payload["pose_association"] = association_report
                    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
                    payload["raw_image"] = str(snapshot_raw)
                    payload["preview_image"] = str(snapshot_preview)
                    save_result_json(snapshot, payload)
                    print(f"saved {snapshot}", flush=True)
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"frames={frame_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
