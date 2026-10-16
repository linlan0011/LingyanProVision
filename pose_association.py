"""Frame-bound input pose, corner measurements and independent visual pose feedback."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from corner_ranging import range_payload
from pose_io import normalize_heading_deg

PROJECT = Path(__file__).resolve().parent


def association_output_path(camera, requested=None, prior_path=None):
    root = (PROJECT / "output" / "association" / camera).resolve()
    output = Path(requested).resolve() if requested is not None else root / "latest.json"
    if not output.is_relative_to(root):
        raise ValueError(f"association output must remain under this camera's directory: {root}")
    if prior_path is not None and output == Path(prior_path).resolve():
        raise ValueError("association output cannot overwrite the simulated prior")
    return output


def association_payload(camera, frame_number, ranger, readings, prior, identities,
                        templates=(), scene="", prior_path=None, *, visual_pose=None):
    data = range_payload(camera, frame_number, ranger, readings)
    data.update({"schema": "camera.simulated_pose_association.v1",
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "created_at_unix": time.time(), "scene": scene,
                 "input_pose_kind": "simulated_UI_prior_not_sensor_measurement",
                 "input_pose_file": str(Path(prior_path).resolve()) if prior_path else None,
                 "simulated_prior": None if prior is None else {
                     "x_mm": prior.x_mm, "y_mm": prior.y_mm, "heading_deg": prior.heading_deg},
                 "pose_corrected": False,
                 "active_templates": [{"safety_id": t.safety_id, "side": t.side} for t in templates],
                 "confirmed_corner_ids": sorted({r.corner_id for r in identities if r.corner_id}),
                 "identities": [{"pixel": list(r.observation.point), "inferred": r.observation.inferred,
                                 "corner_id": r.corner_id, "reason": r.reason} for r in identities]})
    if visual_pose is not None:
        data['visual_pose'] = visual_pose
        data['pose_corrected'] = visual_pose['pose_corrected']
        data['corrected_pose'] = visual_pose['corrected_pose']
        data['input_pose_overwritten'] = False
    return data


def validated_corrected_pose(data):
    """Reject stale/inconsistent fields before drawing an independent pose arrow."""
    from pose_io import Pose2D
    report = data.get('visual_pose')
    if (not isinstance(report, dict) or report.get('schema') != 'safety.nearest_visual_pose.v1'
            or report.get('status') != 'corrected' or not report.get('pose_corrected')
            or not data.get('pose_corrected') or report.get('heading_source') != 'visual_ground_edges'
            or not report.get('corner_id')
            or set(data.get('confirmed_corner_ids', ())) != {report['corner_id']}):
        return None
    pose, root_pose = report.get('corrected_pose'), data.get('corrected_pose')
    if not isinstance(pose, dict) or not isinstance(root_pose, dict):
        return None
    try:
        values = [pose[name] for name in ('x_mm', 'y_mm', 'heading_deg')]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            return None
        if any(root_pose.get(name) != pose[name] for name in ('x_mm', 'y_mm', 'heading_deg')):
            return None
        return Pose2D(*values).validated()
    except (KeyError, TypeError, ValueError):
        return None


class AssociationWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.first = True
        self.last_busy_notice = float('-inf')

    def write(self, data):
        if self.first:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                backup = self.path.with_name(f'{self.path.stem}.backup_{stamp}{self.path.suffix}')
                shutil.copy2(self.path, backup)
            self.first = False
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            os.replace(temporary, self.path)
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33):
                raise
            # The UI's ordinary Windows read handle can briefly deny delete
            # sharing. Keep the last complete report and publish next frame.
            # Waiting here would stall the camera processing loop.
            now = time.monotonic()
            if now - self.last_busy_notice >= 2.0:
                print(f'association output busy; skipped report update: {self.path}', flush=True)
                self.last_busy_notice = now
            return False
        finally:
            temporary.unlink(missing_ok=True)
        return True


def report_matches_pose(data, prior):
    if not isinstance(data, dict) or prior is None or data.get("simulated_prior") is None:
        return False
    pose = data["simulated_prior"]
    try:
        return (abs(pose["x_mm"]-prior.x_mm) <= .002 and abs(pose["y_mm"]-prior.y_mm) <= .002
                and abs(normalize_heading_deg(pose["heading_deg"]-prior.heading_deg)) <= .002)
    except (KeyError, TypeError, ValueError):
        return False


def read_live_association(path, camera, started_at, prior, max_age_s=3.):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        timestamp = float(data["created_at_unix"])
        age = time.time()-timestamp
        if (data.get("schema") != "camera.simulated_pose_association.v1" or
                data.get("camera_id") != camera or not math.isfinite(timestamp) or
                timestamp < started_at or not -1 <= age <= max_age_s or
                not report_matches_pose(data, prior)):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None
