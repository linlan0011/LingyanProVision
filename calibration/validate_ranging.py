from __future__ import annotations

import argparse
import csv
from pathlib import Path

from calibration.calibration_io import validate_point_sources
from calibration.ranging_report import apply_analysis_range, build_camera_report, render_report, save_reports
from camera_profiles import isolated_path, load_profiles, file_sha256


def read_records(paths, camera):
    records, sources = [], []
    if len(set(paths)) != len(paths):
        raise ValueError("the same input CSV was specified more than once")
    for path in paths:
        digest = file_sha256(path)
        validate_point_sources(path, camera)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"x_mm", "y_mm", "z_mm", "u_px", "v_px"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: CSV must contain {', '.join(sorted(required))}")
            count = 0
            for index, row in enumerate(reader, 1):
                records.append({**row, "source_file": str(path), "source_row": index})
                count += 1
        if not count:
            raise ValueError(f"{path}: validation CSV contains no points")
        if file_sha256(path) != digest:
            raise ValueError(f"{path}: CSV changed during reading; retry validation")
        sources.append({"file": str(path), "sha256": digest, "point_count": count})
    return records, sources


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Per-point and overall geometry ranging error report.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--points", type=Path, nargs="+", help="one or more CSVs in this camera's data directory")
    parser.add_argument("--output", type=Path, help="JSON report; matching TXT is also saved")
    parser.add_argument("--range-max-mm", type=float, default=2000.)
    parser.add_argument("--min-plane-angle-deg", type=float, default=3.)
    parser.add_argument("--max-error-mm", type=float, help="optional horizontal-position error pass threshold")
    parser.add_argument("--bin-mm", type=float, default=300., help="reference horizontal-distance grouping width")
    for axis in ("x","y","r"):
        for bound in ("min","max"):
            parser.add_argument(f"--{axis}-{bound}-mm", type=float,
                                help="inclusive analysis bound on surveyed coordinates; omitted=unlimited")
    args = parser.parse_args(argv)
    paths = [isolated_path(p, args.camera, result=False, default_name="range_validation.csv")
             for p in (args.points or [None])]
    output = isolated_path(args.output, args.camera, result=True, default_name="range_validation.json")
    records, sources = read_records(paths, args.camera)
    result = build_camera_report(args.camera, records, max_distance_mm=args.range_max_mm,
                                 min_plane_angle_deg=args.min_plane_angle_deg,
                                 error_limit_mm=args.max_error_mm, bin_mm=args.bin_mm)
    result["input_files"] = sources
    apply_analysis_range(result, **{f"{axis}_{bound}_mm": getattr(args,f"{axis}_{bound}_mm")
                                   for axis in ("x","y","r") for bound in ("min","max")})
    output, txt, backup, txt_backup = save_reports(output, result)
    print(render_report(result), end="")
    print(f"saved {output}\nsaved {txt}")
    for previous in (backup, txt_backup):
        if previous:
            print(f"previous report backed up to {previous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
