"""Live USB-camera preview with an OpenCV purple-region segmentation overlay.

Press q or Esc to stop. All scenes share fitted boundary geometry and
nearest-corner visual pose; camera profiles keep independent calibration.
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
from corner_ranging import DEFAULT_MAX_DISTANCE_MM
from ground_corner_association import (CalibratedGroundView, candidate_safety_view, circle_templates)
from nearest_corner_pose import (VisualPoseConfig, solve_nearest_corner_pose,
    apply_visual_identities)
from preview_display import ParameterPanel, preview_dimensions, compose_preview
from camera_orientation import ROTATION_CHOICES, rotate_frame
from camera_profiles import load_intrinsics, load_mount, load_profiles, verify_capture_format
from field_zones import ZoneFileReader, matching_zones
from pose_io import PoseFileReader
from safety_identity import (DEFAULT_TEMPLATES, TemplateReader, active_templates,
    choose_scene)


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
    display_white_support: float
    infer_hidden_corners: bool
    adaptive_exposure: bool
    allow_split_components: bool


CAMERA_PROFILES = load_profiles()

# Geometry is shared across all scenes; only colour and exposure presets vary.
SIDE_RULES = SceneRules(
    h_low=115, h_high=175, s_min=55, v_min=40,
    min_area=80, corner_min_area=400, corner_min_side=12,
    white_s_max=179, white_v_min=100, display_white_support=0.40,
    infer_hidden_corners=True, adaptive_exposure=False, allow_split_components=True,
)
FRONT_RULES = replace(SIDE_RULES, white_s_max=80, display_white_support=0.35,
                      adaptive_exposure=True)
GENERAL_RULES = replace(SIDE_RULES)
SCENE_RULES = {
    "start-facing": SIDE_RULES,
    "front-facing": FRONT_RULES,
    "general": GENERAL_RULES,
}
USB_0BDA_3035_SCENE_RULES = {
    scene: replace(rules, h_low=120, h_high=150, s_min=55, v_min=25)
    for scene, rules in SCENE_RULES.items()
}
CAMERA_SCENE_RULES = {
    "lingyan": SCENE_RULES,
    "seekfree": SCENE_RULES,
    "usb0bda3035_1080p": USB_0BDA_3035_SCENE_RULES,
}


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
    """Neutral colour evidence for the adjacent strip in every scene.

    Keep the loose field segmentation for mixed edge pixels; require a neutral
    core within the same contiguous strip so a pale blue face cannot support it
    merely by passing the loose S<=80 (or S<=179) threshold.
    """
    return make_white_field_mask(hsv, min(s_max, 50), v_min)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=CAMERA_PROFILES, default="lingyan",
                        help="camera-specific capture geometry and recognition rules")
    parser.add_argument("--index", type=int)
    parser.add_argument("--ranging", action="store_true", help="enable calibrated body-frame corner ranging")
    parser.add_argument("--target-z-mm", type=float, help="required known corner height when ranging; ground=0")
    parser.add_argument("--range-max-mm", type=float, default=DEFAULT_MAX_DISTANCE_MM,
                        help="maximum horizontal body-origin localization range; default 1000 mm")
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
    parser.add_argument('--visual-pose', '--ground-association', dest='ground_association',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='nearest-corner directed-edge identity and independent visual pose')
    parser.add_argument('--pose-radius-mm', type=int, default=200)
    parser.add_argument('--identity-angle-tol-deg', type=int, default=15)
    parser.add_argument('--orthogonal-angle-tol-deg', type=int, default=15)
    parser.add_argument('--heading-consistency-deg', type=int, default=15)
    args = parser.parse_args()
    if not 0 <= args.pose_radius_mm <= 3000:
        parser.error('--pose-radius-mm must be between 0 and 3000')
    if not all(0 <= value < 45 for value in (args.identity_angle_tol_deg,
            args.orthogonal_angle_tol_deg, args.heading_consistency_deg)):
        parser.error('visual angle tolerances must be between 0 and 44 degrees')
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
        range_limit = 'unlimited' if args.range_max_mm is None else f'{args.range_max_mm:g}mm'
        print(f"body ranging enabled: camera={args.camera}, Z={args.target_z_mm:g}mm, "
              f"limit={range_limit}; inferred={args.range_inferred}", flush=True)
    pose_ranger = None
    if args.ground_association and ranger is not None and ranger.target_z_mm == 0:
        from copy import copy
        pose_ranger = copy(ranger)
        pose_ranger.include_inferred = True
        print('visual pose: nearest visible/inferred point; directed ground edges; gyro used for identity only', flush=True)
    requested_scene = args.scene
    ground_view = CalibratedGroundView(pose_ranger) if pose_ranger is not None else None
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
        "white V min": "white_v_min",
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
    if args.ground_association:
        parameter_values.update({'pose radius mm': args.pose_radius_mm,
                                 'identity tol deg': args.identity_angle_tol_deg,
                                 'orthogonal tol deg': args.orthogonal_angle_tol_deg,
                                 'heading tol deg': args.heading_consistency_deg})
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
            candidate_view = None
            if args.ground_association:
                candidate_view = candidate_safety_view(selected_pose, template_model, ground_view,
                                                       parameters.get('pose radius mm'))
                identity_templates = circle_templates(selected_pose, zones, template_model,
                                                      set(candidate_view['possible_safety_ids']),
                                                      parameters.get('pose radius mm'))
                identity_status = candidate_view['status']
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
            detections = find_largest_purple(mask, min_area)
            safety_ranks = select_safety_component_ranks(
                detections, rules, frame.shape,
                minimum_area=max(min_area, corner_min_area),
            )
            frame_corner_supports = {}
            frame_boundary_result = {}
            recognitions = [(detection, [], []) for detection in detections[:12]]
            selected_indices = [
                rank - 1 for rank in sorted(safety_ranks)
                if 1 <= rank <= len(recognitions)
            ]
            selected_contours = [
                recognitions[index][0][-1] for index in selected_indices
            ]
            # Detection, display and ranging use the same fitted support pair
            # in side, front and general views, including occluded corners.
            all_visible_corners, all_inferred_corners = find_effective_boundary_corners(
                selected_contours, white_mask, mask, rules, corner_min_side,
                corner_supports=frame_corner_supports,
                boundary_result=frame_boundary_result,
                point_geometry=point_geometry,
                white_core_mask=white_core_mask,
            )
            for group_index, corners in ((1, all_visible_corners), (2, all_inferred_corners)):
                for corner in corners:
                    point = corner[0]
                    owner_index = max(
                        selected_indices,
                        key=lambda index: cv2.pointPolygonTest(
                            recognitions[index][0][-1],
                            (float(point[0]), float(point[1])),
                            True,
                        ),
                    )
                    recognitions[owner_index][group_index].append(corner)
            frame_ranger = pose_ranger or ranger
            range_readings = frame_ranger.measure(recognitions) if frame_ranger is not None else []
            geometry_readings = (frame_ranger.measure_geometry(recognitions)
                                 if frame_ranger is not None else [])
            identities = []
            visual_report = None
            reference_points = set()
            if args.ground_association:
                visual_config = VisualPoseConfig(
                    identity_tolerance_deg=parameters.get('identity tol deg'),
                    orthogonality_tolerance_deg=parameters.get('orthogonal tol deg'),
                    heading_consistency_deg=parameters.get('heading tol deg'))
                identities, visual_report = solve_nearest_corner_pose(
                    range_readings, geometry_readings, frame_corner_supports, selected_pose,
                    template_model if color_mode and rotation == camera_profile.rotation else None,
                    pose_ranger, candidate_view, boundary_result=frame_boundary_result.get('result'),
                    config=visual_config)
                range_readings = apply_visual_identities(range_readings, identities)
                selected = visual_report['selected_corner']
                if selected is not None:
                    reference_points = {tuple(selected['pixel'])}
                identity_status = visual_report['status']
            overlay = frame.copy()
            cv2.drawContours(overlay, [d[0][-1] for d in recognitions], -1, (255, 0, 0), 2)
            if "result" in frame_boundary_result:
                draw_boundary_result(overlay, frame_boundary_result["result"], labels=False)
            for _, visible_corners, inferred_corners in recognitions:
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
            cv2.drawContours(mask_bgr, [d[0][-1] for d in recognitions], -1, (255, 0, 0), 2)
            if "result" in frame_boundary_result:
                draw_boundary_result(mask_bgr, frame_boundary_result["result"], labels=False)
            # Draw the exact same blue contour on the binary view. The contour
            # coordinates are unchanged; only the display image differs.
            for _, corners, inferred_corners in recognitions:
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
                visual_pose=visual_report, geometry_readings=geometry_readings,
            )
            if ranger is not None:
                from pose_association import association_payload
                association_report = association_payload(args.camera, frame_count, frame_ranger, range_readings,
                                                         selected_pose, identities, identity_templates,
                                                         scene, args.pose_file, visual_pose=visual_report)
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
                    if visual_report is not None:
                        payload['visual_pose'] = visual_report
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
