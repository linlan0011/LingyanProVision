"""Immutable, verified snapshot of one registered camera's raw and result files."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from camera_profiles import PROJECT, data_dir, load_profiles, results_dir


def archive_camera(camera_id: str, destination: Path | None = None) -> Path:
    profile = load_profiles()[camera_id]
    if destination is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = PROJECT / "calibration" / "archives" / f"{camera_id}_{stamp}.zip"
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"archive already exists: {destination}")
    sources = (data_dir(camera_id), results_dir(camera_id))
    files = sorted(
        path for root in sources if root.exists()
        for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise FileNotFoundError(f"nothing to archive for {camera_id}")
    if destination.is_relative_to(sources[0].resolve()) or destination.is_relative_to(sources[1].resolve()):
        raise ValueError("archive must stay outside the source folders")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    try:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                member = path.relative_to(PROJECT).as_posix()
                payload = path.read_bytes()
                archive.writestr(member, payload)
                records[member] = hashlib.sha256(payload).hexdigest()
            archive.writestr("manifest.json", json.dumps({
                "schema": "camera.archive.v1", "camera_id": camera_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "profile": profile.__dict__, "files_sha256": records,
                "note": "Snapshot only; existing source files remain in their original locations.",
            }, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("archive CRC verification failed")
            for member, expected in records.items():
                if hashlib.sha256(archive.read(member)).hexdigest() != expected:
                    raise RuntimeError(f"archive hash verification failed: {member}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive one camera's raw data and calibration results")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    args = parser.parse_args()
    archive = archive_camera(args.camera)
    print(f"verified archive: {archive}")
    with zipfile.ZipFile(archive) as packed:
        print(f"files={len(packed.namelist()) - 1}, bytes={archive.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
