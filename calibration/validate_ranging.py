from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from calibration.calibration_io import camera_arrays, load_json, save_json
from calibration.ground_project import pixel_to_body_plane


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare plane-intersection ranging with surveyed points.")
    parser.add_argument("--intrinsics", type=Path, default=Path("calibration/results/lingyan/intrinsics.json"))
    parser.add_argument("--mount", type=Path, default=Path("calibration/results/lingyan/mount_pose.json"))
    parser.add_argument("--points", type=Path, default=Path("calibration/data/lingyan/range_validation.csv"))
    parser.add_argument("--output", type=Path, default=Path("calibration/results/lingyan/range_validation.json"))
    args = parser.parse_args()

    intrinsics = load_json(args.intrinsics)
    mount = load_json(args.mount)
    camera_matrix, distortion = camera_arrays(intrinsics)
    rotation_bc = np.asarray(mount["rotation_camera_to_body"], dtype=np.float64)
    position_b = np.asarray(mount["camera_position_body_mm"], dtype=np.float64)
    rows = []
    with args.points.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            expected = np.array([float(row[k]) for k in ("x_mm", "y_mm", "z_mm")])
            pixel = (float(row["u_px"]), float(row["v_px"]))
            predicted = pixel_to_body_plane(
                pixel, expected[2], camera_matrix, distortion, rotation_bc, position_b,
            )
            error_xy = predicted[:2] - expected[:2]
            rows.append({
                "row": index,
                "expected_body_mm": expected.tolist(),
                "pixel": list(pixel),
                "predicted_body_mm": predicted.tolist(),
                "x_error_mm": float(error_xy[0]),
                "y_error_mm": float(error_xy[1]),
                "horizontal_error_mm": float(np.linalg.norm(error_xy)),
            })
    if not rows:
        raise ValueError("validation CSV contains no points")
    errors = np.array([row["horizontal_error_mm"] for row in rows])
    result = {
        "schema": "lingyan.range_validation.v1",
        "mean_horizontal_error_mm": float(np.mean(errors)),
        "median_horizontal_error_mm": float(np.median(errors)),
        "max_horizontal_error_mm": float(np.max(errors)),
        "points": rows,
    }
    save_json(args.output, result)
    print(f"saved {args.output}")
    print(f"count={len(rows)}, mean={np.mean(errors):.1f} mm, "
          f"median={np.median(errors):.1f} mm, max={np.max(errors):.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
