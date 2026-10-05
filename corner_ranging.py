"""Calibrated body-frame ranging of raw detected pixels on a known-height plane."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_io import camera_arrays
from calibration.ground_project import pixel_to_body_plane
from calibration.lens_models import undistort_points
from camera_profiles import load_intrinsics, load_mount, file_sha256


@dataclass(frozen=True)
class CornerRange:
    label: str
    pixel: tuple[float, float]
    inferred: bool
    valid: bool
    reason: str = ""
    x_mm: float | None = None
    y_mm: float | None = None
    z_mm: float | None = None
    distance_mm: float | None = None
    bearing_deg: float | None = None
    corner_id: str | None = None


class CornerRanger:
    def __init__(self, intrinsics, mount, target_z_mm, max_distance_mm=2000.,
                 min_plane_angle_deg=3., include_inferred=False):
        if not math.isfinite(target_z_mm):
            raise ValueError("target height must be finite")
        if not math.isfinite(max_distance_mm) or max_distance_mm <= 0:
            raise ValueError("maximum ranging distance must be positive and finite")
        if not math.isfinite(min_plane_angle_deg) or not 0 < min_plane_angle_deg < 90:
            raise ValueError("minimum ray/plane angle must be between 0 and 90 degrees")
        self.matrix, self.distortion = camera_arrays(intrinsics)
        self.rotation = np.asarray(mount["rotation_camera_to_body"], dtype=float)
        self.center = np.asarray(mount["camera_position_body_mm"], dtype=float)
        self.size = (int(intrinsics["image_width"]), int(intrinsics["image_height"]))
        self.model = intrinsics["model"]
        self.target_z_mm = float(target_z_mm)
        self.max_distance_mm = float(max_distance_mm)
        self.min_sine = math.sin(math.radians(min_plane_angle_deg))
        self.include_inferred = include_inferred
        self.sources = []

    @classmethod
    def from_camera(cls, camera, **kwargs):
        intrinsic_path, intrinsics = load_intrinsics(camera)
        mount_path, mount = load_mount(camera)
        instance = cls(intrinsics, mount, **kwargs)
        instance.sources = [(path, file_sha256(path)) for path in (intrinsic_path, mount_path)]
        return instance

    def check_sources(self):
        # Never continue using cached parameters after a calibration is replaced.
        try:
            return all(file_sha256(path) == digest for path, digest in self.sources)
        except OSError:
            return False

    def project(self, pixel, *, label="", inferred=False):
        pixel = tuple(map(float, pixel))
        def reject(reason):
            return CornerRange(label, pixel, inferred, False, reason)
        if len(pixel) != 2 or not all(math.isfinite(v) for v in pixel):
            return reject("invalid_pixel")
        if not (0 <= pixel[0] < self.size[0] and 0 <= pixel[1] < self.size[1]):
            return reject("pixel_outside_image")
        if inferred and not self.include_inferred:
            return reject("inferred_not_measured")
        try:
            normalized = undistort_points(pixel, self.matrix, self.distortion, self.model)[0]
            ray = self.rotation @ np.array([*normalized, 1.])
            if not np.isfinite(ray).all() or abs(ray[2]) / np.linalg.norm(ray) < self.min_sine:
                return reject("near_horizon")
            point = pixel_to_body_plane(pixel, self.target_z_mm, self.matrix, self.distortion,
                                       self.rotation, self.center, self.model)
            if not np.isfinite(point).all():
                return reject("nonfinite_intersection")
            if point[0] <= 0:
                return reject("behind_body_origin")
            distance = math.hypot(point[0], point[1])
            if distance > self.max_distance_mm:
                return reject("beyond_distance_limit")
            return CornerRange(label, pixel, inferred, True, "estimated" if inferred else "",
                               *map(float, point), distance,
                               math.degrees(math.atan2(point[1], point[0])))
        except (ValueError, cv2.error):
            return reject("no_forward_plane_intersection")

    def measure(self, recognitions, identities=()):
        names = {tuple(r.observation.point): r.corner_id for r in identities if r.corner_id}
        readings = []
        seen = set()
        changed = not self.check_sources()
        # Direct observations take precedence over inferred duplicates.
        for inferred in (False, True):
            for rank, (_, visible, hidden) in enumerate(recognitions, 1):
                for point, _ in (hidden if inferred else visible):
                    pixel = tuple(map(float, point))
                    if pixel in seen:
                        continue
                    seen.add(pixel)
                    label = f"R{len(readings)+1}"
                    if names.get(pixel):
                        label += f"/{names[pixel]}"
                    reading = (CornerRange(label, pixel, inferred, False, "calibration_changed")
                               if changed else self.project(pixel, label=label, inferred=inferred))
                    if names.get(pixel):
                        reading = replace(reading, corner_id=names[pixel])
                    readings.append(reading)
        return readings

    def identity_candidates(self, observation, template, prior, model):
        """Use actual K/D/R/C and the known plane, not an assumed horizontal FOV."""
        if prior is None or not self.check_sources():
            return set()
        measured = self.project(observation.point)
        if not measured.valid:
            return set()
        from safety_identity import map_corners
        from calibration.lens_models import project_points
        from pose_io import normalize_heading_deg
        zone = next((z for z in model["safety_zones"] if z["id"] == template.safety_id), None)
        if zone is None:
            return set()
        theta = math.radians(prior.heading_deg)
        tolerance = model["diagonal_pose_projection"]["image_bearing_tolerance_deg"]
        rvec, _ = cv2.Rodrigues(self.rotation.T)
        tvec = -self.rotation.T @ self.center
        candidates = set()
        for index, (x, y) in enumerate(map_corners(zone), 1):
            dx, dy = x-prior.x_mm, y-prior.y_mm
            body = np.array([math.cos(theta)*dx+math.sin(theta)*dy,
                             -math.sin(theta)*dx+math.cos(theta)*dy, self.target_z_mm])
            camera = self.rotation.T @ (body-self.center)
            if body[0] <= 0 or camera[2] <= 0:
                continue
            predicted_pixel = project_points(body.reshape(1, 3), rvec, tvec, self.matrix,
                                             self.distortion, self.model)[0]
            if not (0 <= predicted_pixel[0] < self.size[0] and 0 <= predicted_pixel[1] < self.size[1]):
                continue
            expected_bearing = math.degrees(math.atan2(body[1], body[0]))
            if abs(normalize_heading_deg(expected_bearing-measured.bearing_deg)) <= tolerance:
                candidates.add(f"{template.safety_id}.{index}")
        return candidates


def range_payload(camera, frame_number, ranger, readings):
    return {"schema": "camera.live_corner_ranges.v1", "camera_id": camera,
            "frame_number": frame_number, "coordinate_frame": "body_X_forward_Y_left_Z_up",
            "units": "millimetres", "target_z_mm": ranger.target_z_mm,
            "max_distance_mm": ranger.max_distance_mm,
            "calibration_valid": ranger.check_sources(),
            "intrinsics_sha256": ranger.sources[0][1] if ranger.sources else None,
            "mount_sha256": ranger.sources[1][1] if ranger.sources else None,
            "corners": [asdict(r) for r in readings]}


def save_range_image(path, image):
    """OpenCV imwrite can fail on Windows paths containing Chinese characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode ranging snapshot")
    Path(path).write_bytes(encoded.tobytes())


def draw_range_tags(image, readings):
    scale = max(.6, image.shape[1] / 960 * .7)
    for reading in readings:
        if not reading.valid:
            continue
        text = f"{reading.label} r={reading.distance_mm/1000:.3f}m"
        if reading.inferred:
            text += " EST"
        color = (0, 165, 255) if reading.inferred else (100, 255, 100)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        x = max(4, min(int(reading.pixel[0])+12, image.shape[1]-text_width-4))
        y = max(28, min(int(reading.pixel[1])+60, image.shape[0]-8))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def append_range_panel(preview, ranger, readings):
    # Resize only the display, after measuring native pixels; retain readable text.
    if preview.shape[1] > 1280:
        preview = cv2.resize(preview, (1280, round(preview.shape[0]*1280/preview.shape[1])))
    panel = np.full((224, preview.shape[1], 3), 24, dtype=np.uint8)
    lines = [(f"BODY RANGE: Z={ranger.target_z_mm:g}mm | limit={ranger.max_distance_mm:g}mm | "
              "X forward, Y left | estimates, not field pose", (230, 230, 230))]
    if not ranger.check_sources():
        lines.append(("DISABLED: calibration changed; restart with matching files", (0, 80, 255)))
    elif not readings:
        lines.append(("Waiting for directly detected ground corners", (0, 200, 255)))
    for r in readings[:7]:
        if r.valid:
            text = (f"{r.label}: X={r.x_mm:.1f} Y={r.y_mm:.1f}mm  "
                    f"r={r.distance_mm:.1f}mm  bearing={r.bearing_deg:+.2f}deg")
            if r.inferred:
                text += " INFERRED/EST"
            color = (0, 165, 255) if r.inferred else (100, 255, 100)
        else:
            text, color = f"{r.label}: no range ({r.reason})", (0, 165, 255)
        lines.append((text, color))
    if len(readings) > 7:
        lines.append((f"+{len(readings)-7} more corners (see snapshot JSON)", (220, 220, 220)))
    for index, (text, color) in enumerate(lines[:9]):
        cv2.putText(panel, text, (10, 24+24*index), cv2.FONT_HERSHEY_SIMPLEX, .56, color, 1, cv2.LINE_AA)
    return np.vstack((preview, panel))
