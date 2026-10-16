"""Multi-point geometry validation; never fit or modify camera calibration."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from calibration.calibration_io import save_result_json
from calibration.ground_project import pixel_to_body_plane
from camera_profiles import file_sha256, isolated_path, load_intrinsics, load_mount
from corner_ranging import CornerRanger
from pose_io import normalize_heading_deg


def percentile95(values):
    """Nearest-rank empirical P95, not a confidence interval."""
    return float(sorted(values)[math.ceil(.95*len(values))-1]) if values else None


def summarize(rows):
    valid = [r for r in rows if r["valid"]]
    def metric(key, operation):
        values = [r[key] for r in valid if r.get(key) is not None]
        return float(operation(values)) if values else None
    def rms(values):
        return np.sqrt(np.mean(np.square(values)))
    return {
        "count": len(rows), "valid_count": len(valid), "invalid_count": len(rows)-len(valid),
        "runtime_usable_count": sum(r["runtime_usable"] for r in rows),
        "unique_position_count": len({tuple(r["expected_body_mm"]) for r in rows}),
        "mean_x_error_mm": metric("x_error_mm", np.mean),
        "mean_y_error_mm": metric("y_error_mm", np.mean),
        "rmse_x_mm": metric("x_error_mm", rms), "rmse_y_mm": metric("y_error_mm", rms),
        "mean_horizontal_error_mm": metric("horizontal_error_mm", np.mean),
        "median_horizontal_error_mm": metric("horizontal_error_mm", np.median),
        "rmse_horizontal_mm": metric("horizontal_error_mm", rms),
        "p95_horizontal_error_mm": metric("horizontal_error_mm", percentile95),
        "max_horizontal_error_mm": metric("horizontal_error_mm", max),
        "mean_distance_error_mm": metric("distance_error_mm", np.mean),
        "mae_distance_mm": metric("distance_error_mm", lambda v: np.mean(np.abs(v))),
        "rmse_distance_mm": metric("distance_error_mm", rms),
        "max_abs_distance_error_mm": metric("distance_error_mm", lambda v: max(np.abs(v))),
        "mae_bearing_deg": metric("bearing_error_deg", lambda v: np.mean(np.abs(v))),
        "max_abs_bearing_error_deg": metric("bearing_error_deg", lambda v: max(np.abs(v))),
        "pass_count": sum(r.get("passed") is True for r in rows),
        "fail_count": sum(r.get("passed") is False for r in rows),
        "min_reference_distance_mm": min((r["reference_distance_mm"] for r in rows), default=None),
        "max_reference_distance_mm": max((r["reference_distance_mm"] for r in rows), default=None),
    }


def apply_analysis_range(result, *, x_min_mm=None, x_max_mm=None, y_min_mm=None,
                         y_max_mm=None, r_min_mm=None, r_max_mm=None):
    """Inclusive selection by surveyed coordinates, independent of runtime gates."""
    bounds = dict(x_min_mm=x_min_mm,x_max_mm=x_max_mm,y_min_mm=y_min_mm,
                  y_max_mm=y_max_mm,r_min_mm=r_min_mm,r_max_mm=r_max_mm)
    for key,value in bounds.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{key}: 范围边界必须是有限数字，留空表示不限")
    for axis in ("x","y","r"):
        lo,hi = bounds[f"{axis}_min_mm"],bounds[f"{axis}_max_mm"]
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{axis.upper()} 下限不能大于上限")
        if axis == "r" and any(v is not None and v < 0 for v in (lo,hi)):
            raise ValueError("水平距离 r 的范围边界不能小于 0")
    selected = []
    for row in result["points"]:
        x,y,_ = row["expected_body_mm"]
        values = dict(x=x,y=y,r=row["reference_distance_mm"])
        matches = all((bounds[f"{axis}_min_mm"] is None or value >= bounds[f"{axis}_min_mm"])
                      and (bounds[f"{axis}_max_mm"] is None or value <= bounds[f"{axis}_max_mm"])
                      for axis,value in values.items())
        row["selected_by_range"] = matches
        if matches:
            selected.append(row)
    result.update(report_version=4,
                  analysis_range={"active": any(v is not None for v in bounds.values()),
                                  "coordinate_source": "surveyed_body_coordinates",
                                  "boundaries_inclusive": True,**bounds,
                                  "point_ids": [r["point_id"] for r in selected]},
                  range_summary=summarize(selected),
                  range_runtime_summary=summarize([r for r in selected if r["runtime_usable"]]))
    return result


def analyse_records(records, intrinsics, mount, *, max_distance_mm=2000.,
                    min_plane_angle_deg=3., error_limit_mm=None, bin_mm=300.):
    if not math.isfinite(bin_mm) or bin_mm <= 0:
        raise ValueError("distance bin width must be positive and finite")
    if error_limit_mm is not None and (not math.isfinite(error_limit_mm) or error_limit_mm < 0):
        raise ValueError("horizontal error limit must be nonnegative and finite")
    CornerRanger(intrinsics, mount, 0., max_distance_mm, min_plane_angle_deg)
    rows, rangers = [], {}
    for index, record in enumerate(records, 1):
        source = f'{record.get("source_file", "current collection")} row {record.get("source_row", index)}'
        try:
            x,y,z,u,v = [float(record[k]) for k in ("x_mm","y_mm","z_mm","u_px","v_px")]
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{source}: missing or invalid X/Y/Z/u/v") from exc
        if not all(math.isfinite(a) for a in (x,y,z,u,v)):
            raise ValueError(f"{source}: X/Y/Z/u/v must be finite")
        reference_r = math.hypot(x,y)
        reference_a = math.degrees(math.atan2(y,x)) if reference_r > 1e-9 else None
        row = {"row": index, "point_id": f"P{index:03d}",
               "source_file": record.get("source_file"), "source_row": record.get("source_row",index),
               "frame_file": record.get("frame_file",""), "expected_body_mm": [x,y,z], "pixel": [u,v],
               "reference_distance_mm": reference_r, "reference_bearing_deg": reference_a,
               "valid": False, "reason": "", "runtime_usable": False, "runtime_reason": "",
               "predicted_body_mm": None, "predicted_distance_mm": None, "predicted_bearing_deg": None,
               "x_error_mm": None, "y_error_mm": None, "horizontal_error_mm": None,
               "distance_error_mm": None, "bearing_error_deg": None,
               "relative_distance_error_percent": None, "passed": None}
        if z not in rangers:
            rangers[z] = CornerRanger(intrinsics,mount,z,max_distance_mm,min_plane_angle_deg)
        ranger = rangers[z]
        runtime = ranger.project((u,v))
        row.update(runtime_usable=runtime.valid,runtime_reason=runtime.reason)
        if not (0 <= u < ranger.size[0] and 0 <= v < ranger.size[1]):
            row["reason"] = "pixel_outside_image"
        else:
            try:
                point = pixel_to_body_plane((u,v),z,ranger.matrix,ranger.distortion,
                                           ranger.rotation,ranger.center,ranger.model)
                if not np.isfinite(point).all():
                    raise ValueError("nonfinite intersection")
                dx,dy = float(point[0]-x),float(point[1]-y)
                predicted_r = math.hypot(point[0],point[1])
                predicted_a = math.degrees(math.atan2(point[1],point[0])) if predicted_r > 1e-9 else None
                e = math.hypot(dx,dy)
                row.update(valid=True,predicted_body_mm=point.tolist(),x_error_mm=dx,y_error_mm=dy,
                           horizontal_error_mm=e,predicted_distance_mm=predicted_r,
                           predicted_bearing_deg=predicted_a,distance_error_mm=predicted_r-reference_r,
                           bearing_error_deg=normalize_heading_deg(predicted_a-reference_a)
                           if predicted_a is not None and reference_a is not None else None,
                           relative_distance_error_percent=100*abs(predicted_r-reference_r)/reference_r
                           if reference_r > 1e-9 else None,
                           passed=e <= error_limit_mm if error_limit_mm is not None else None)
            except (ValueError,cv2.error) as exc:
                row["reason"] = str(exc)
        rows.append(row)
    if not rows:
        raise ValueError("validation contains no points")
    result = {"schema": "camera.range_validation.v2", "report_version": 3,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "coordinate_frame": "body_X_forward_Y_left_Z_up", "units": "millimetres",
              "limits": {"runtime_max_distance_mm": max_distance_mm,
                         "runtime_min_plane_angle_deg": min_plane_angle_deg,
                         "horizontal_error_limit_mm": error_limit_mm,"distance_bin_mm": bin_mm},
              "points": rows,"summary": summarize(rows),
              "runtime_summary": summarize([r for r in rows if r["runtime_usable"]]),
              "groups_by_x": [],"distance_groups": []}
    for x in sorted({r["expected_body_mm"][0] for r in rows}):
        result["groups_by_x"].append({"x_mm": x,**summarize([r for r in rows if r["expected_body_mm"][0]==x])})
    for bucket in sorted({math.floor(r["reference_distance_mm"]/bin_mm) for r in rows}):
        group = [r for r in rows if math.floor(r["reference_distance_mm"]/bin_mm)==bucket]
        result["distance_groups"].append({"start_mm": bucket*bin_mm,"end_mm": (bucket+1)*bin_mm,
                                          "interval": "[start,end)",**summarize(group)})
    for key in ("mean_horizontal_error_mm","median_horizontal_error_mm","max_horizontal_error_mm"):
        result[key] = result["summary"][key]
    return result


def build_camera_report(camera, records, **options):
    intrinsic_path,intrinsics = load_intrinsics(camera)
    mount_path,mount = load_mount(camera)
    intrinsic_hash,mount_hash = file_sha256(intrinsic_path),file_sha256(mount_path)
    if (mount.get("intrinsics_sha256") != intrinsic_hash or
            json.loads(intrinsic_path.read_text(encoding="utf-8")) != intrinsics or
            json.loads(mount_path.read_text(encoding="utf-8")) != mount):
        raise ValueError("calibration changed while loading; restart validation")
    result = analyse_records(records,intrinsics,mount,**options)
    if (file_sha256(intrinsic_path),file_sha256(mount_path)) != (intrinsic_hash,mount_hash):
        raise ValueError("calibration changed during validation; report not saved")
    result.update(camera_id=camera,intrinsics_file=str(intrinsic_path),mount_file=str(mount_path),
                  intrinsics_sha256=intrinsic_hash,mount_sha256=mount_hash,
                  calibration_snapshot={"intrinsics": intrinsics,"mount": mount})
    return result


def render_report(result):
    def num(value):
        return "--" if value is None else f"{value:.2f}"
    s = result["summary"]
    lines = ["几何测距多点误差报告","="*78,
             f'相机：{result.get("camera_id","synthetic")}；生成时间：{result["created_utc"]}',
             "坐标：车体 X 向前、Y 向左、Z 向上；长度 mm，角度 deg，像素 px。",
             f'总点数={s["count"]}，几何有效={s["valid_count"]}，无效={s["invalid_count"]}，'
             f'实时门限可用={s["runtime_usable_count"]}，不同 XYZ 位置={s["unique_position_count"]}',
             f'全部几何有效点的水平误差：mean={num(s["mean_horizontal_error_mm"])}，median={num(s["median_horizontal_error_mm"])}，'
             f'RMSE={num(s["rmse_horizontal_mm"])}，P95={num(s["p95_horizontal_error_mm"])}，max={num(s["max_horizontal_error_mm"])} mm',
             f'平均偏差：ΔX={num(s["mean_x_error_mm"])}，ΔY={num(s["mean_y_error_mm"])} mm',
             f'分轴 RMSE：X={num(s["rmse_x_mm"])}，Y={num(s["rmse_y_mm"])} mm',
             f'距离误差：平均带符号={num(s["mean_distance_error_mm"])}，MAE={num(s["mae_distance_mm"])}，'
             f'RMSE={num(s["rmse_distance_mm"])}，max|Δr|={num(s["max_abs_distance_error_mm"])} mm',
             f'方位角误差：MAE={num(s["mae_bearing_deg"])}，max|Δa|={num(s["max_abs_bearing_error_deg"])} deg',
             f'实测水平距离覆盖：{num(s["min_reference_distance_mm"])} 至 {num(s["max_reference_distance_mm"])} mm',
             f'实时门限：r≤{result["limits"]["runtime_max_distance_mm"]:g} mm，'
             f'射线与平面夹角≥{result["limits"]["runtime_min_plane_angle_deg"]:g} deg（另检查正向交点/车体前方）']
    limit = result["limits"]["horizontal_error_limit_mm"]
    lines.append("尚未设置合格门槛；未自动判定合格范围。" if limit is None else
                 f'eXY≤{limit:g} mm：通过={s["pass_count"]}，超限={s["fail_count"]}；无效点另计。')
    lines += ["","逐点坐标（原始像素，不是缩小预览坐标）：",
              "ID       X真       Y真    Z真      u       v       X算       Y算   几何/实时"]
    for r in result["points"]:
        x,y,z = r["expected_body_mm"]
        pred = r["predicted_body_mm"] or [None,None,None]
        status = "有效/可用" if r["runtime_usable"] else ("有效/拒绝" if r["valid"] else "无效/拒绝")
        lines.append(f'{r["point_id"]:<6} {x:9.2f} {y:9.2f} {z:6.2f} {r["pixel"][0]:7.2f} '
                     f'{r["pixel"][1]:7.2f} {num(pred[0]):>9} {num(pred[1]):>9}  {status}')
    lines += ["","逐点误差（Δ=计算-实测；百分比为 |Δr|/r真）：",
              "ID       r真      r算       ΔX       ΔY      eXY       Δr    a真/deg  a算/deg   Δa/deg  相对r/%  判定"]
    for r in result["points"]:
        verdict = "无效" if not r["valid"] else ("--" if r["passed"] is None else "通过" if r["passed"] else "超限")
        keys = ("reference_distance_mm","predicted_distance_mm","x_error_mm","y_error_mm",
                "horizontal_error_mm","distance_error_mm","reference_bearing_deg","predicted_bearing_deg",
                "bearing_error_deg","relative_distance_error_percent")
        lines.append(f'{r["point_id"]:<6} '+" ".join(f"{num(r[k]):>8}" for k in keys)+"  "+verdict)
        if not r["runtime_usable"]:
            lines.append(f'       原因：{r["reason"] or r["runtime_reason"]}')
    selection = result.get("analysis_range")
    if selection and selection["active"]:
        conditions = []
        for axis in ("x","y","r"):
            label = axis.upper() if axis != "r" else "r"
            lo,hi = selection[f"{axis}_min_mm"],selection[f"{axis}_max_mm"]
            if lo is not None:
                conditions.append(f"{label}≥{lo:g} mm")
            if hi is not None:
                conditions.append(f"{label}≤{hi:g} mm")
        chosen = set(selection["point_ids"])
        selected = [r for r in result["points"] if r["point_id"] in chosen]
        q = result["range_summary"]
        lines += ["","="*78,"可调范围统计（按实测坐标；边界包含；多条件同时满足）：",
                  " 且 ".join(conditions),
                  f'入选={q["count"]}/{s["count"]}，几何有效={q["valid_count"]}，无效={q["invalid_count"]}，'
                  f'实时门限可用={q["runtime_usable_count"]}',
                  f'范围内水平误差：mean={num(q["mean_horizontal_error_mm"])}，median={num(q["median_horizontal_error_mm"])}，'
                  f'RMSE={num(q["rmse_horizontal_mm"])}，P95={num(q["p95_horizontal_error_mm"])}，max={num(q["max_horizontal_error_mm"])} mm',
                  f'平均偏差：ΔX={num(q["mean_x_error_mm"])}，ΔY={num(q["mean_y_error_mm"])} mm；'
                  f'分轴 RMSE：X={num(q["rmse_x_mm"])}，Y={num(q["rmse_y_mm"])} mm',
                  f'距离误差：平均带符号={num(q["mean_distance_error_mm"])}，MAE={num(q["mae_distance_mm"])}，'
                  f'RMSE={num(q["rmse_distance_mm"])}，max|Δr|={num(q["max_abs_distance_error_mm"])} mm',
                  f'方位角误差：MAE={num(q["mae_bearing_deg"])}，max|Δa|={num(q["max_abs_bearing_error_deg"])} deg']
        if limit is not None:
            lines.append(f'范围内 eXY≤{limit:g} mm：通过={q["pass_count"]}，超限={q["fail_count"]}；无效点另计。')
        lines += ["范围内逐点误差（保留全报告原点号）：",
                  "ID       X真      Y真       ΔX       ΔY      eXY       Δr     Δa/deg  相对r/%"]
        for r in selected:
            keys = ("x_error_mm","y_error_mm","horizontal_error_mm","distance_error_mm",
                    "bearing_error_deg","relative_distance_error_percent")
            x,y,_ = r["expected_body_mm"]
            lines.append(f'{r["point_id"]:<6} {x:8.2f} {y:8.2f} '+" ".join(f"{num(r[k]):>8}" for k in keys))
            if not r["runtime_usable"]:
                lines.append(f'       原因：{r["reason"] or r["runtime_reason"]}')
        if not selected:
            lines.append("本范围没有采集点；各误差显示 --，不是零误差。")
        rt_range = result["range_runtime_summary"]
        lines += [f'范围内仅实时可用点：n={rt_range["count"]}，mean_e={num(rt_range["mean_horizontal_error_mm"])}，'
                  f'RMSE={num(rt_range["rmse_horizontal_mm"])}，P95={num(rt_range["p95_horizontal_error_mm"])}，'
                  f'max_e={num(rt_range["max_horizontal_error_mm"])} mm',"="*78]
    for title,groups in (("按前向 X 分组",result["groups_by_x"]),("按实测水平距离 r 分组",result["distance_groups"])):
        lines += ["",title+"（误差基于几何有效点，长度 mm）：",
                  "组别          有效/总数  实时可用   偏X     偏Y    mean_e    RMSE     P95     max_e"]
        for g in groups:
            label = f'X={g["x_mm"]:g}' if "x_mm" in g else f'[{g["start_mm"]:g},{g["end_mm"]:g})'
            metrics = ("mean_x_error_mm","mean_y_error_mm","mean_horizontal_error_mm",
                       "rmse_horizontal_mm","p95_horizontal_error_mm","max_horizontal_error_mm")
            lines.append(f'{label:<15} {g["valid_count"]:>3}/{g["count"]:<3} {g["runtime_usable_count"]:>6} '
                         +" ".join(f"{num(g[k]):>8}" for k in metrics))
    rt = result["runtime_summary"]
    lines += ["",f'仅实时门限可用点：n={rt["count"]}，mean_e={num(rt["mean_horizontal_error_mm"])}，'
              f'RMSE={num(rt["rmse_horizontal_mm"])}，P95={num(rt["p95_horizontal_error_mm"])}，max_e={num(rt["max_horizontal_error_mm"])} mm',
              "","来源索引（各 CSV 按指定顺序合并；未自动删除重复测量）："]
    for r in result["points"]:
        lines.append(f'{r["point_id"]}: {r["source_file"] or "当前采集"}，行 {r["source_row"]}；原图 {r["frame_file"] or "未记录"}')
    lines += ["","注意：P95 使用最近秩法，≤5 点时等于最大值；不是 95% 精度保证。",
              "无效点不按零误差参与统计；运行门限拒绝点若几何可解，仍保留其几何误差。",
              "重复位置按每条记录计权；不同位置的离散程度不等于同一点的时间抖动。",
              "Z 是求交平面的输入约束，不是本方法独立测出的高度。",
              "本报告不包含未采集/视野外位置，不能据此推算覆盖率或自动识别成功率。",
              "结论仅针对已采样的位置、光照及安装状态；未自动推断连续有效范围。"]
    return "\n".join(lines)+"\n"


def save_reports(output,result):
    output = isolated_path(output,result["camera_id"],result=True,default_name="range_validation.json")
    if output.suffix.lower() != ".json" or output.name.lower() in {"intrinsics.json","mount_pose.json","validation.json"}:
        raise ValueError("choose a separate .json report, not a calibration parameter file")
    text = render_report(result)
    json.dumps(result, ensure_ascii=False, allow_nan=False)  # Reject nonfinite statistics before any writes.
    backup = save_result_json(output,result)
    txt_path = output.with_suffix(".txt")
    text_backup = None
    if txt_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        text_backup = txt_path.with_name(f"{txt_path.stem}.backup_{stamp}.txt")
        shutil.copy2(txt_path,text_backup)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=txt_path.parent,delete=False) as stream:
        temp = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temp,txt_path)
    finally:
        temp.unlink(missing_ok=True)
    return output,txt_path,backup,text_backup
