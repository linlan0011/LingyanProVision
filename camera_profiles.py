"""Registered camera modes and isolated calibration paths.

One ID identifies a physical camera *and* a fixed capture mode, orientation,
focus/zoom setting, and lens model. Register another ID if any of these change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from camera_orientation import ROTATION_CHOICES

PROJECT = Path(__file__).resolve().parent
REGISTRY = PROJECT / "camera_profiles.json"
MODELS = ("opencv_pinhole", "opencv_fisheye")
ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{1,31}\Z")


@dataclass(frozen=True)
class CameraProfile:
    display_name: str
    capture_width: int
    capture_height: int
    rotation: str
    lens_model: str = "opencv_pinhole"
    default_index: int = 0
    notes: str = ""
    pixel_format: str | None = None
    target_fps: float | None = None
    backend: str = "auto"
    device_vid_pid: str | None = None
    fixed_controls: dict[str, int] | None = None
    automatic_controls: dict[str, int] | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        if self.rotation in ("clockwise", "counterclockwise"):
            return self.capture_height, self.capture_width
        return self.capture_width, self.capture_height


BUILTINS = {
    "lingyan": CameraProfile("原灵眼 Pro", 640, 480, "none"),
    "seekfree": CameraProfile("逐飞彩钻风 Pro", 768, 1024, "clockwise"),
}


def validate_profile(camera_id: str, profile: CameraProfile) -> None:
    if not ID_PATTERN.fullmatch(camera_id):
        raise ValueError("camera ID must be 2-32 lowercase letters, digits, '_' or '-', starting with a letter")
    if not profile.display_name.strip():
        raise ValueError("display name is required")
    if profile.capture_width < 16 or profile.capture_height < 16:
        raise ValueError("capture width and height must be at least 16 pixels")
    if profile.rotation not in ROTATION_CHOICES:
        raise ValueError(f"unsupported rotation {profile.rotation}")
    if profile.lens_model not in MODELS:
        raise ValueError(f"unsupported lens model {profile.lens_model}")
    if profile.default_index < 0:
        raise ValueError("camera index must be nonnegative")
    if profile.pixel_format is not None and not re.fullmatch(r"[A-Z0-9]{4}", profile.pixel_format):
        raise ValueError("pixel format must be a four-character FOURCC such as MJPG or YUY2")
    if profile.target_fps is not None and (not math.isfinite(profile.target_fps) or profile.target_fps <= 0):
        raise ValueError("target FPS must be a positive finite number")
    if profile.backend not in ("auto", "dshow", "msmf"):
        raise ValueError("unsupported capture backend")
    if profile.device_vid_pid is not None and not re.fullmatch(r"[0-9A-F]{4}:[0-9A-F]{4}", profile.device_vid_pid):
        raise ValueError("USB identity must be uppercase VID:PID, for example 0BDA:3035")
    if profile.fixed_controls or profile.automatic_controls:
        from camera_controls import CONTROL_IDS
        if not profile.device_vid_pid:
            raise ValueError("fixed controls require an identity-bound camera profile")
        if any(name not in CONTROL_IDS or not isinstance(value, int)
               for name, value in {**(profile.fixed_controls or {}), **(profile.automatic_controls or {})}.items()):
            raise ValueError("invalid fixed camera controls")
        if set(profile.fixed_controls or {}) & set(profile.automatic_controls or {}):
            raise ValueError("a control cannot be both manual and automatic")


def verify_capture_format(cap, profile: CameraProfile) -> None:
    if profile.backend != "auto" and cap.getBackendName().lower() != profile.backend:
        raise RuntimeError("camera backend differs from profile")
    if profile.pixel_format is None:
        return
    import cv2
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual = "".join(chr((value >> (8 * i)) & 255) for i in range(4))
    if actual != profile.pixel_format:
        raise RuntimeError(f"camera returned {actual!r}, expected {profile.pixel_format}; do not mix capture formats")


def load_profiles(registry: Path | None = None) -> dict[str, CameraProfile]:
    registry = REGISTRY if registry is None else Path(registry)
    profiles = dict(BUILTINS)
    if not registry.exists():
        return profiles
    data = json.loads(registry.read_text(encoding="utf-8"))
    if data.get("schema") != "lingyan.camera_profiles.v1" or not isinstance(data.get("cameras"), dict):
        raise ValueError(f"invalid camera registry: {registry}")
    for camera_id, fields in data["cameras"].items():
        if camera_id in profiles:
            raise ValueError(f"built-in camera ID cannot be replaced: {camera_id}")
        profile = CameraProfile(**fields)
        validate_profile(camera_id, profile)
        profiles[camera_id] = profile
    return profiles


def register_profile(camera_id: str, profile: CameraProfile, registry: Path | None = None) -> None:
    registry = REGISTRY if registry is None else Path(registry)
    validate_profile(camera_id, profile)
    if camera_id in load_profiles(registry):
        raise ValueError(f"camera ID {camera_id!r} already exists; choose a new ID for a changed mode")
    data = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {
        "schema": "lingyan.camera_profiles.v1", "cameras": {},
    }
    data["cameras"][camera_id] = asdict(profile)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=registry.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, registry)
    finally:
        temporary.unlink(missing_ok=True)


def data_dir(camera_id: str) -> Path:
    load_profiles()[camera_id]
    return PROJECT / "calibration" / "data" / camera_id


def results_dir(camera_id: str) -> Path:
    load_profiles()[camera_id]
    return PROJECT / "calibration" / "results" / camera_id


def isolated_path(path: Path | None, camera_id: str, *, result: bool, default_name: str) -> Path:
    base = results_dir(camera_id) if result else data_dir(camera_id)
    resolved = (Path(path) if path is not None else base / default_name).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"camera {camera_id!r} files must stay under {base}; got {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_intrinsics(camera_id: str):
    path = results_dir(camera_id) / "intrinsics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = load_profiles()[camera_id]
    if data.get("camera_id", "lingyan" if camera_id == "lingyan" else None) != camera_id:
        raise ValueError(f"intrinsics belong to another camera: {path}")
    if (data.get("image_width"), data.get("image_height")) != profile.image_size:
        raise ValueError(f"intrinsics image size differs from {camera_id} profile")
    if data.get("model") != profile.lens_model:
        raise ValueError(f"intrinsics lens model differs from {camera_id} profile")
    if "rotation" in data and data["rotation"] != profile.rotation:
        raise ValueError(f"intrinsics rotation differs from {camera_id} profile")
    if "capture_width" in data and (data["capture_width"], data["capture_height"]) != (
        profile.capture_width, profile.capture_height
    ):
        raise ValueError(f"intrinsics capture mode differs from {camera_id} profile")
    if profile.pixel_format is not None and data.get("pixel_format") != profile.pixel_format:
        raise ValueError(f"intrinsics capture format differs from {camera_id} profile")
    if profile.target_fps is not None and data.get("target_fps") != profile.target_fps:
        raise ValueError(f"intrinsics capture rate differs from {camera_id} profile")
    if profile.backend != "auto" and data.get("backend") != profile.backend:
        raise ValueError(f"intrinsics backend differs from {camera_id} profile")
    if profile.device_vid_pid and data.get("device_vid_pid") != profile.device_vid_pid:
        raise ValueError(f"intrinsics hardware identity differs from {camera_id} profile")
    if profile.fixed_controls and data.get("fixed_controls") != profile.fixed_controls:
        raise ValueError(f"intrinsics fixed controls differ from {camera_id} profile")
    if (data.get("automatic_controls") or {}) != (profile.automatic_controls or {}):
        raise ValueError(f"intrinsics automatic controls differ from {camera_id} profile")
    matrix = np.asarray(data.get("camera_matrix"), dtype=float)
    distortion = np.asarray(data.get("distortion_coefficients"), dtype=float)
    if matrix.shape != (3, 3) or distortion.shape not in ((4,), (5,), (8,), (12,), (14,)):
        raise ValueError("invalid intrinsic matrix or distortion shape")
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("intrinsics contain nonfinite values")
    if profile.lens_model == "opencv_fisheye" and distortion.size != 4:
        raise ValueError("fisheye model requires four coefficients")
    return path, data


def load_mount(camera_id: str):
    intrinsics_path, intrinsics = load_intrinsics(camera_id)
    path = results_dir(camera_id) / "mount_pose.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("camera_id", "lingyan" if camera_id == "lingyan" else None) != camera_id:
        raise ValueError(f"mount pose belongs to another camera: {path}")
    if data.get("model", "opencv_pinhole" if camera_id == "lingyan" else None) != intrinsics["model"]:
        raise ValueError(f"mount pose lens model differs from {camera_id} intrinsics")
    expected_hash = data.get("intrinsics_sha256")
    if expected_hash != file_sha256(intrinsics_path):
        raise ValueError(f"mount pose uses different intrinsics: {path}")
    rotation = np.asarray(data.get("rotation_camera_to_body"), dtype=float)
    center = np.asarray(data.get("camera_position_body_mm"), dtype=float)
    if rotation.shape != (3, 3) or center.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(center).all():
        raise ValueError("invalid mount rotation or camera center")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("mount rotation is not a proper rotation matrix")
    if np.max(np.abs(center)) > 100_000:
        raise ValueError("mount center is physically implausible; check point correspondences")
    return path, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Register and list independent camera capture modes")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("camera_id")
    add.add_argument("--name", required=True)
    add.add_argument("--width", type=int, required=True, help="sensor capture width before rotation")
    add.add_argument("--height", type=int, required=True, help="sensor capture height before rotation")
    add.add_argument("--rotation", choices=ROTATION_CHOICES, default="none")
    add.add_argument("--model", choices=MODELS, default="opencv_pinhole")
    add.add_argument("--index", type=int, default=0)
    add.add_argument("--notes", default="", help="physical camera serial, focus/zoom and mode notes")
    add.add_argument("--fourcc", help="capture format, for example MJPG")
    add.add_argument("--fps", type=float, help="requested capture rate")
    add.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    add.add_argument("--device-vid-pid", help="bind to USB VID:PID, for example 0BDA:3035")
    add.add_argument("--control", action="append", default=[], help="fixed manual control, for example camera.exposure=-6")
    add.add_argument("--auto-control", action="append", default=[], help="automatic control with driver seed, for example video.white_balance=6500")
    args = parser.parse_args()
    if args.command == "add":
        fixed = {}
        for setting in args.control:
            name, value = setting.split("=", 1)
            fixed[name] = int(value)
        automatic = {}
        for setting in args.auto_control:
            name, value = setting.split("=", 1)
            automatic[name] = int(value)
        register_profile(args.camera_id, CameraProfile(
            args.name, args.width, args.height, args.rotation, args.model, args.index, args.notes,
            args.fourcc, args.fps, args.backend, args.device_vid_pid, fixed or None, automatic or None,
        ))
        print(f"registered {args.camera_id}; calibration data and results now use independent folders")
    for camera_id, profile in load_profiles().items():
        try:
            load_intrinsics(camera_id)
            intrinsic_status = "intrinsics: yes"
        except (FileNotFoundError, ValueError):
            intrinsic_status = "intrinsics: pending"
        try:
            load_mount(camera_id)
            mount_status = "mount: available (validate range separately)"
        except (FileNotFoundError, ValueError):
            mount_status = "mount: pending/invalid"
        print(f"{camera_id}: {profile.display_name}, capture {profile.capture_width}x{profile.capture_height}, "
              f"rotation {profile.rotation}, image {profile.image_size[0]}x{profile.image_size[1]}, "
              f"model {profile.lens_model}, format {profile.pixel_format or 'default'}, "
              f"fps {profile.target_fps or 'default'}, backend {profile.backend}, "
              f"USB {profile.device_vid_pid or 'unbound'}; {intrinsic_status}; {mount_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
