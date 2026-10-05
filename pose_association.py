"""Read-only scheme-one feedback: simulated prior -> corner IDs and ranges."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

from calibration.calibration_io import save_result_json
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
                        templates=(), scene="", prior_path=None):
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
    return data


class AssociationWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.first = True

    def write(self, data):
        if self.first:
            save_result_json(self.path, data)
            self.first = False
            return
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


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
