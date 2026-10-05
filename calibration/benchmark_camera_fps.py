"""Enumerate DirectShow modes and time actual BGR frame delivery for each mode."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def native_modes(index: int):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        raise RuntimeError("PowerShell is required to read DirectShow capability tables")
    source = str(Path(__file__).with_name("DirectShowModes.cs")).replace("'", "''")
    command = f"Add-Type -Path '{source}'; [CameraDiagnostics.Modes]::Read({index}) | ConvertTo-Json -Depth 3 -Compress"
    process = subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True,
                             encoding="utf-8", errors="replace", timeout=30)
    if process.returncode != 0:
        raise RuntimeError(f"DirectShow enumeration failed: {process.stderr}")
    modes = json.loads(process.stdout)
    if isinstance(modes, dict):
        modes = [modes]
    unique = {}
    for item in modes:
        key = (item["Width"], item["Height"], item["FourCC"])
        if key not in unique or item["MaxFPS"] > unique[key]["MaxFPS"]:
            unique[key] = item
    return sorted(unique.values(), key=lambda item: (
        0 if item["FourCC"] == "MJPG" else 1, item["Width"] * item["Height"], item["Width"]))


def fourcc_name(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * i)) & 255) for i in range(4))


def measure(index: int, mode: dict, seconds: float):
    width, height, fourcc = mode["Width"], mode["Height"], mode["FourCC"]
    maximum = mode["MaxFPS"] or mode["DefaultFPS"]
    result = {"requested_width": width, "requested_height": height, "requested_fourcc": fourcc,
              "driver_max_fps": maximum, "driver_min_fps": mode["MinFPS"],
              "device_name": mode["DeviceName"], "device_index": index}
    cap = None
    for attempt in range(4):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        cap.release()
        time.sleep(.5)
    try:
        if not cap.isOpened():
            result["error"] = "camera cannot open"
            return result
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        result["fps_request_accepted"] = cap.set(cv2.CAP_PROP_FPS, maximum)
        # DSHOW resolution/FPS setters rebuild the media type; select FOURCC LAST.
        result["fourcc_request_accepted"] = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        ready = time.perf_counter()
        warm_frames = 0
        frame = None
        while warm_frames < 5 or time.perf_counter() - ready < 1.0:
            ok, frame = cap.read()
            if not ok or frame is None:
                result["error"] = "no frame during warmup"
                return result
            warm_frames += 1
        actual = (frame.shape[1], frame.shape[0])
        actual_fourcc = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
        result.update({"actual_width": actual[0], "actual_height": actual[1],
                       "actual_fourcc": actual_fourcc, "reported_fps": cap.get(cv2.CAP_PROP_FPS)})
        if actual != (width, height) or actual_fourcc != fourcc:
            result["error"] = "requested mode fell back to a different resolution/format"
            return result
        intervals = []
        hashes = set()
        start = previous = time.perf_counter()
        count = 0
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                result["error"] = "frame read failed during measurement"
                break
            if (frame.shape[1], frame.shape[0]) != actual:
                result["error"] = "resolution changed during measurement"
                break
            intervals.append(now - previous)
            previous = now
            count += 1
            hashes.add(hashlib.sha256(frame[::40, ::40].tobytes()).digest())
        elapsed = time.perf_counter() - start
        result.update({"frames": count, "elapsed_seconds": elapsed, "measured_fps": count / elapsed,
                       "distinct_sample_hashes": len(hashes),
                       "median_frame_interval_ms": float(np.median(intervals) * 1000) if intervals else None,
                       "p95_frame_interval_ms": float(np.percentile(intervals, 95) * 1000) if intervals else None,
                       "last_frame_sample_mean": float(frame[::32, ::32].mean()) if frame is not None else None})
        return result
    finally:
        cap.release()


def save_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = report["measurements"]
    text = ["# SP3220L 分辨率与最大帧率实测", "", f"采集时间（UTC）：{report['created_utc']}", "",
            "按驱动列出的最高帧率请求模式，先预热，再连续读取已解码彩色帧计时；不运行检测算法。",
            "表内实际值是当前电脑的接收速率，不将驱动报告 FPS 当作实际帧率。", ""]
    best = {}
    for row in rows:
        if row.get("error") or "measured_fps" not in row:
            continue
        key = row["requested_width"], row["requested_height"]
        if key not in best or row["measured_fps"] > best[key]["measured_fps"]:
            best[key] = row
    text.extend(["## 每档最高实测结果", "",
                 "| 分辨率 | 最快实测 FPS | 格式 | 连续读取帧数 |",
                 "| --- | ---: | --- | ---: |"])
    for key, row in sorted(best.items(), key=lambda item: item[0][0] * item[0][1]):
        text.append(f"| {key[0]}×{key[1]} | {row['measured_fps']:.2f} | {row['actual_fourcc']} | {row['frames']} |")
    failed = sum(bool(row.get("error")) for row in rows)
    if failed:
        text.extend(["", f"另有 {failed} 次对照测量未完成，详见下表；失败不能解释为该模式帧率为零或不支持。"])
    text.extend(["", "## 全部逐模式记录", "",
            "| 分辨率 | 格式 | 驱动上限 FPS | 实测 FPS | 帧数 / 秒 | 状态 |",
            "| --- | --- | ---: | ---: | --- | --- |"])
    for row in sorted(rows, key=lambda r: (r["requested_width"] * r["requested_height"], r["requested_fourcc"])):
        fps = row.get("measured_fps")
        text.append(f"| {row['requested_width']}×{row['requested_height']} | {row['requested_fourcc']} | "
                    f"{row['driver_max_fps']:.2f} | {fps:.2f}" if fps is not None else
                    f"| {row['requested_width']}×{row['requested_height']} | {row['requested_fourcc']} | {row['driver_max_fps']:.2f} | —")
        text[-1] += (f" | {row.get('frames', 0)} / {row.get('elapsed_seconds', 0):.2f} | "
                     f"{row.get('error', '正常')} |")
    text.extend(["", "DirectShow 设置顺序：分辨率 → FPS → FOURCC（最后设置 MJPG/YUY2）。",
                 "本机先设置 MJPG 再改分辨率或 FPS 时，会回到 YUY2；应检查实际 FOURCC。",
                 "光照、曝光、USB 连接、CPU 解码和识别运算均可能影响最终识别帧率。",
                 "驱动帧率范围读取方法参见 [Microsoft VIDEO_STREAM_CONFIG_CAPS](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ns-strmif-video_stream_config_caps)。",
                 "原始逐模式数据见同名 JSON。", ""])
    path.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure every driver-listed camera mode at its maximum rate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    if args.seconds < 3:
        raise ValueError("use at least 3 seconds per mode")
    modes = native_modes(args.index)
    # Some UVC drivers release enumeration handles asynchronously.
    time.sleep(1.0)
    output = PROJECT / "calibration/benchmarks" / f"sp3220l_fps_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    report = {"schema": "camera.fps_benchmark.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "seconds_per_mode": args.seconds, "native_modes": modes, "measurements": []}
    print(f"driver modes={len(modes)}, result={output}", flush=True)
    for i, mode in enumerate(modes, 1):
        print(f"[{i}/{len(modes)}] {mode['Width']}x{mode['Height']} {mode['FourCC']} "
              f"driver max={mode['MaxFPS']:.2f}", flush=True)
        row = measure(args.index, mode, args.seconds)
        report["measurements"].append(row)
        save_report(output, report)
        print(f"  observed={row.get('measured_fps', 0):.2f} fps; "
              f"actual={row.get('actual_width')}x{row.get('actual_height')} {row.get('actual_fourcc')}; "
              f"{row.get('error', 'OK')}", flush=True)
    print(f"saved {output} and {output.with_suffix('.md')}", flush=True)
    return 0 if any(row.get("measured_fps", 0) > 0 for row in report["measurements"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
