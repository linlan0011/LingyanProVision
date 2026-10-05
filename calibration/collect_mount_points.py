"""Manual surveyed-point collection on unmodified LINGYAN camera frames."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import shutil
import tempfile
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk, scrolledtext

import cv2
import numpy as np

from calibration.calibration_io import load_json
from calibration.fit_mount_pose import validate_correspondences
from camera_orientation import rotate_frame
from camera_profiles import data_dir, file_sha256, load_intrinsics, load_profiles, verify_capture_format
from purple_camera_segmentation import open_camera

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = ("x_mm", "y_mm", "z_mm", "u_px", "v_px", "frame_file")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def image_pixel(x: float, y: float, zoom: float, width: int, height: int):
    # Canvas coordinates already include scrolling; display zoom is integer nearest-neighbour.
    u, v = math.floor(x / zoom), math.floor(y / zoom)
    return (u, v) if 0 <= u < width and 0 <= v < height else None


def measured_xyz(values) -> tuple[float, float, float]:
    try:
        xyz = tuple(float(value) for value in values)
    except (ValueError, TypeError) as exc:
        raise ValueError("请填写 X、Y、Z 三个实测坐标，单位为毫米。") from exc
    if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
        raise ValueError("坐标必须是有限数字，不能留空或填写 NaN / Inf。")
    return xyz


@dataclass(frozen=True)
class Point:
    x_mm: float
    y_mm: float
    z_mm: float
    u_px: int
    v_px: int
    frame_file: str


def validate_new_point(point: Point, existing: list[Point], size: tuple[int, int]) -> None:
    measured_xyz((point.x_mm, point.y_mm, point.z_mm))
    if not all(math.isfinite(v) for v in (point.u_px, point.v_px)):
        raise ValueError("像素坐标不是有效数字。")
    if not (0 <= point.u_px < size[0] and 0 <= point.v_px < size[1]):
        raise ValueError("选中的像素不在原图内。")
    for other in existing:
        if (point.u_px, point.v_px) == (other.u_px, other.v_px):
            raise ValueError("这个像素已经采集过，请勿重复添加。")
        if (point.x_mm, point.y_mm, point.z_mm) == (other.x_mm, other.y_mm, other.z_mm):
            raise ValueError("这个实测位置已经采集过。填错时请先删除旧行再重新添加。")


def points_csv(points: list[Point]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(asdict(point) for point in points)
    return stream.getvalue()


def export_points(path: Path, points: list[Point], purpose: str) -> Path | None:
    if purpose == "mount":
        validate_correspondences(
            np.array([(p.x_mm, p.y_mm, p.z_mm) for p in points]),
            np.array([(p.u_px, p.v_px) for p in points]),
        )
    elif not points:
        raise ValueError("请至少采集一个独立验证点。")
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup_{timestamp()}{path.suffix}")
        shutil.copy2(path, backup)
    atomic_text(path, points_csv(points))
    return backup


class CameraFeed:
    """Keep device IO off the Tk thread; only this worker owns/releases the capture."""

    def __init__(self, index: int, profile=None):
        self.profile = profile or load_profiles()["lingyan"]
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(index,), daemon=True)
        self.thread.start()

    def _run(self, index: int) -> None:
        cap = None
        try:
            profile = self.profile
            cap = open_camera(index, profile.capture_width, profile.capture_height,
                              fourcc=profile.pixel_format, fps=profile.target_fps, profile=profile)
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("相机未提供图像。请关闭其他占用相机的程序，并检查连接。")
                verify_capture_format(cap, profile)
                if (frame.shape[1], frame.shape[0]) != (profile.capture_width, profile.capture_height):
                    raise RuntimeError(f"实际帧尺寸不是 {profile.capture_width}×{profile.capture_height}："
                                       f"{frame.shape[1]}×{frame.shape[0]}")
                frame = rotate_frame(frame, profile.rotation)
                with self.lock:
                    self.frame = frame
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.error


class Collector:
    def __init__(self, root: tk.Tk, *, purpose="mount", camera="lingyan", index=None,
                 intrinsics=None, output=None, sessions=None):
        self.root = root
        self.camera = camera
        self.profile = load_profiles()[camera]
        self.purpose = purpose
        self.index = self.profile.default_index if index is None else index
        if intrinsics is None:
            self.intrinsics, params = load_intrinsics(camera)
        else:
            self.intrinsics = Path(intrinsics)
            params = load_json(self.intrinsics)
        self.size = (int(params["image_width"]), int(params["image_height"]))
        if self.size != self.profile.image_size or params.get("model") != self.profile.lens_model:
            raise ValueError(f"内参与 {camera} 相机配置的图像尺寸或镜头模型不符。")
        if params.get("camera_id", "lingyan" if camera == "lingyan" else None) != camera:
            raise ValueError("内参属于另一台相机。")
        self.intrinsics_hash = file_sha256(self.intrinsics)
        filename = "mount_points.csv" if purpose == "mount" else "range_validation.csv"
        self.output = Path(output) if output else data_dir(camera) / filename
        self.session = (Path(sessions) if sessions else data_dir(camera) / "point_sessions") / f"{purpose}_{timestamp()}"
        self.created = datetime.now(timezone.utc).isoformat()
        self.points: list[Point] = []
        self.frame = None
        self.frame_file = None
        self.source_image = None
        self.selected = None
        self.frozen = False
        self.feed = None
        self.closed = False
        self.after_id = None
        self.last_export = None
        self.dirty = False
        self.zoom = tk.DoubleVar(value=0.5 if self.size[0] > 1000 or self.size[1] > 750 else 1.0)
        self.xyz = [tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="0")]
        self.status = tk.StringVar(value="请打开相机或原图。保持车辆、相机及车体系原点不变。")
        self.pixel_label = tk.StringVar(value="未点选")
        self.phase = tk.StringVar(value="尚未打开画面")
        self.count = tk.StringVar(value="已采集 0 点")
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll()

    def _build(self):
        title = "装车外参" if self.purpose == "mount" else "独立测距验证"
        self.root.title(f"{self.profile.display_name} 标定点采集 — {title}")
        self.root.geometry("1160x820")
        self.root.minsize(940, 720)
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="打开相机", command=self.start_camera).pack(side="left")
        ttk.Button(top, text="打开原图…", command=self.open_image).pack(side="left", padx=6)
        self.freeze_button = ttk.Button(top, text="冻结画面", command=self.toggle_freeze)
        self.freeze_button.pack(side="left")
        ttk.Label(top, text="显示放大（不改原图）：").pack(side="left", padx=(16, 0))
        zoom_box = ttk.Combobox(top, textvariable=self.zoom, values=(0.5, 1, 2, 3), state="readonly", width=3)
        zoom_box.pack(side="left")
        zoom_box.bind("<<ComboboxSelected>>", lambda _: self.render())
        ttk.Label(top, textvariable=self.phase).pack(side="left", padx=12)
        ttk.Label(self.root, text=f"相机 {self.camera} · 图像 {self.size[0]}×{self.size[1]} · 方向 {self.profile.rotation} · 冻结后点击十字中心",
                  padding=(8, 0, 8, 6)).pack(anchor="w")
        body = ttk.Frame(self.root, padding=(8, 0))
        body.pack(fill="both", expand=True)
        image_box = ttk.Frame(body)
        image_box.pack(side="left", fill="both", expand=True)
        image_box.rowconfigure(0, weight=1)
        image_box.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(image_box, background="#202329", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xscroll = ttk.Scrollbar(image_box, orient="horizontal", command=self.canvas.xview)
        yscroll = ttk.Scrollbar(image_box, orient="vertical", command=self.canvas.yview)
        xscroll.grid(row=1, column=0, sticky="ew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.hover)
        side = ttk.Frame(body, padding=(12, 0, 0, 0), width=300)
        side.pack(side="right", fill="y")
        ttk.Label(side, text="1. 冻结画面并点击标记中心\n2. 输入这个点相对车体原点的坐标\n3. 点击添加，再采下一个点").pack(anchor="w")
        ttk.Label(side, textvariable=self.pixel_label, padding=(0, 10)).pack(anchor="w")
        entries = ttk.Frame(side)
        entries.pack(anchor="w")
        self.entries = []
        for row, label in enumerate(("X 向前（mm）", "Y 向左（mm）", "Z 向上（mm）")):
            ttk.Label(entries, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(entries, textvariable=self.xyz[row], width=14)
            entry.grid(row=row, column=1, padx=8)
            self.entries.append(entry)
        ttk.Label(side, text="地面 Z=0 仅在车体系原点位于地面时成立。\n右侧点 Y<0；坐标不是相对镜头测量。",
                  wraplength=290, padding=(0, 8)).pack(anchor="w")
        ttk.Button(side, text="添加这个点", command=self.add_point).pack(fill="x", pady=5)
        ttk.Label(side, text="光标附近 21×21 原图像素（辅助找中心）").pack(anchor="w", pady=(8, 2))
        self.loupe = tk.Canvas(side, width=210, height=210, highlightthickness=0, background="#202329")
        self.loupe.pack(anchor="w")
        ttk.Label(side, textvariable=self.count, padding=(0, 8)).pack(anchor="w")
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        columns = ("id", "X", "Y", "Z", "u", "v", "image")
        self.table = ttk.Treeview(bottom, columns=columns, show="headings", height=5, selectmode="extended")
        for name, label in zip(columns, ("编号", "X / mm", "Y / mm", "Z / mm", "u / px", "v / px", "原图")):
            self.table.heading(name, text=label)
            self.table.column(name, width=70 if name != "image" else 320, anchor="center")
        self.table.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(bottom, orient="vertical", command=self.table.yview)
        scrollbar.pack(side="left", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)
        buttons = ttk.Frame(bottom, padding=(8, 0))
        buttons.pack(side="right", fill="y")
        ttk.Button(buttons, text="删除选中行", command=self.delete_points).pack(fill="x", pady=4)
        ttk.Button(buttons, text="导出标定 CSV", command=self.export).pack(fill="x", pady=4)
        if self.purpose == "validation":
            ttk.Button(buttons, text="查看全部点误差", command=self.show_range_report).pack(fill="x", pady=4)
        ttk.Label(self.root, textvariable=self.status, wraplength=1100, padding=8).pack(fill="x")

    def _same_pose(self) -> bool:
        return not self.points or messagebox.askokcancel(
            "确认安装姿态不变", "继续本批采集前，请确认车辆、镜头、安装支架、车体系原点均未移动。\n"
            "只可移动地面标记，且要重新测量它的坐标。\n"
            "如果车辆或相机动过，请取消并重新启动工具，新建一批采集。", parent=self.root)

    def start_camera(self):
        if not self._same_pose():
            return
        if self.feed and self.feed.thread.is_alive() and self.feed.stop.is_set():
            messagebox.showinfo("相机正在释放", "稍等相机释放完成后，再点击打开相机。", parent=self.root)
            return
        if not self.feed or not self.feed.thread.is_alive():
            self.feed = CameraFeed(self.index, self.profile)
        self.frozen, self.selected, self.frame_file = False, None, None
        self.frame = None
        self.source_image = None
        self.phase.set("连接相机中…")
        self.pixel_label.set("未点选")
        self.freeze_button.configure(text="冻结画面")
        self.status.set("实时预览不能采点；清晰且静止时点击“冻结画面”。")

    def _poll(self):
        if self.closed:
            return
        if self.feed and not self.frozen and not self.feed.stop.is_set():
            frame, error = self.feed.latest()
            if error:
                self.feed.stop.set()
                self.frame = None
                self.canvas.delete("all")
                self.phase.set("相机错误")
                self.status.set(error)
            elif frame is not None:
                self.frame = frame
                self.phase.set("实时画面")
                self.render()
        self.after_id = self.root.after(60, self._poll)

    def open_image(self):
        if not self._same_pose():
            return
        filename = filedialog.askopenfilename(parent=self.root, title=f"选择 {self.camera} 的同模式原图（未裁切、未去畸变）",
                                             filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("全部", "*.*")])
        if filename:
            try:
                self.load_image(Path(filename))
            except Exception as exc:
                messagebox.showerror("无法打开图片", str(exc), parent=self.root)

    def load_image(self, path: Path):
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"需要 {self.size[0]}×{self.size[1]} 的同相机原图；不能用缩放图、截图或拼接预览图。")
        if self.feed:
            self.feed.stop.set()
        self.frame = frame
        self.source_image = str(path.resolve())
        self.frozen, self.selected, self.frame_file = True, None, None
        self.pixel_label.set("未点选")
        self.phase.set("已冻结原图")
        self.freeze_button.configure(text="重新打开相机")
        self.status.set(f"请确认这确实是 {self.camera}、同一安装姿态和镜头焦点拍摄的原始图片。")
        self.render()

    def toggle_freeze(self):
        if self.frozen:
            self.start_camera()
        elif self.frame is not None:
            self.frame = self.frame.copy()
            self.frozen, self.selected, self.frame_file = True, None, None
            self.phase.set("画面已冻结")
            self.freeze_button.configure(text="继续实时预览")
            self.status.set("点击十字中心，输入该点实测坐标；同一冻结画面可采多个点。")
            self.render()

    def render(self):
        if self.frame is None:
            return
        zoom = self.zoom.get()
        shown = cv2.resize(self.frame, None, fx=zoom, fy=zoom,
                           interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_NEAREST)
        self.photo = self._photo(shown)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, shown.shape[1], shown.shape[0]))
        for number, point in enumerate(self.points, 1):
            if point.frame_file == self.frame_file:
                self._cross(point.u_px, point.v_px, "#38ed7a", str(number))
        if self.selected:
            self._cross(*self.selected, "#ffd34e", "选中")

    @staticmethod
    def _photo(frame):
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("图像编码失败")
        return tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))

    def _cross(self, u, v, color, label):
        zoom = self.zoom.get()
        x, y = (u + 0.5) * zoom, (v + 0.5) * zoom
        self.canvas.create_line(x - 8, y, x + 8, y, fill=color, width=1)
        self.canvas.create_line(x, y - 8, x, y + 8, fill=color, width=1)
        self.canvas.create_text(x + 9, y - 8, text=label, fill=color, anchor="sw")

    def _event_pixel(self, event):
        return image_pixel(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
                           self.zoom.get(), *self.size)

    def click(self, event):
        if not self.frozen or self.frame is None:
            self.status.set("请先冻结画面，再点选标记中心。")
            return
        selected = self._event_pixel(event)
        if selected is not None:
            self.selected = selected
            self.pixel_label.set(f"已选原图像素 u={selected[0]}, v={selected[1]}")
            self.entries[0].focus_set()
            self.render()

    def hover(self, event):
        if self.frame is None or not self.frozen:
            return
        pixel = self._event_pixel(event)
        if pixel is None:
            return
        u, v = pixel
        padded = cv2.copyMakeBorder(self.frame, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        crop = padded[v:v + 21, u:u + 21]
        self.loupe_photo = self._photo(cv2.resize(crop, (210, 210), interpolation=cv2.INTER_NEAREST))
        self.loupe.delete("all")
        self.loupe.create_image(0, 0, image=self.loupe_photo, anchor="nw")
        self.loupe.create_rectangle(100, 100, 110, 110, outline="#ffd34e")

    def _ensure_snapshot(self):
        if self.frame_file is None:
            self.session.mkdir(parents=True, exist_ok=True)
            image_path = self.session / f"frame_{timestamp()}.png"
            ok, encoded = cv2.imencode(".png", self.frame)
            if not ok:
                raise RuntimeError("原图保存失败")
            image_path.write_bytes(encoded.tobytes())
            self.frame_file = str(image_path.resolve())

    def add_point(self):
        try:
            if not self.frozen or self.selected is None:
                raise ValueError("请先冻结画面并点选中心。")
            xyz = measured_xyz([value.get() for value in self.xyz])
            point = Point(*xyz, *self.selected, self.frame_file or "")
            validate_new_point(point, self.points, self.size)
            self._ensure_snapshot()
            point = Point(*xyz, *self.selected, self.frame_file)
            self.points.append(point)
            try:
                self._save_draft()
            except Exception:
                self.points.pop()
                raise
            self.dirty = True
            self.selected = None
            self.pixel_label.set("未点选：请继续点击下一个标记")
            self.xyz[0].set("")
            self.xyz[1].set("")
            self.refresh_table()
            self.render()
            self.status.set(f"已自动保存 {len(self.points)} 点草稿。采集完成后点击“导出标定 CSV”。")
        except Exception as exc:
            messagebox.showerror("未添加", str(exc), parent=self.root)

    def _save_draft(self):
        atomic_text(self.session / "points.csv", points_csv(self.points))
        metadata = {
            "schema": "camera.manual_point_session.v2", "purpose": self.purpose,
            "created_utc": self.created, "camera": self.camera, "image_size": list(self.size),
            "rotation": self.profile.rotation, "pixel_space": "registered_rotated_distorted", "units": "millimetres",
            "pixel_format": self.profile.pixel_format, "target_fps": self.profile.target_fps,
            "backend": self.profile.backend, "device_vid_pid": self.profile.device_vid_pid,
            "fixed_controls": self.profile.fixed_controls,
            "automatic_controls": self.profile.automatic_controls,
            "body_axes": "X forward, Y left, Z up", "intrinsics_file": str(self.intrinsics.resolve()),
            "intrinsics_sha256": self.intrinsics_hash, "last_opened_image": self.source_image,
            "point_count": len(self.points), "points": [asdict(point) for point in self.points],
            "note": "Manual picks; assumes a fixed vehicle/camera pose throughout the session.",
        }
        atomic_text(self.session / "session.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    def refresh_table(self):
        self.table.delete(*self.table.get_children())
        for i, p in enumerate(self.points):
            self.table.insert("", "end", iid=str(i), values=(
                i + 1, f"{p.x_mm:g}", f"{p.y_mm:g}", f"{p.z_mm:g}", p.u_px, p.v_px, Path(p.frame_file).name))
        self.count.set(f"已采集 {len(self.points)} 点" + ("（外参建议至少 12 点）" if self.purpose == "mount" else "（勿重复使用外参拟合点）"))

    def delete_points(self):
        selected = set(map(int, self.table.selection()))
        if not selected:
            return
        previous = self.points
        self.points = [p for i, p in enumerate(previous) if i not in selected]
        try:
            self._save_draft()
        except Exception as exc:
            self.points = previous
            messagebox.showerror("删除未保存", str(exc), parent=self.root)
            return
        self.dirty = True
        self.refresh_table()
        self.render()
        self.status.set("已删除选中行并更新草稿；此前导出的文件未变，如需更新请重新导出。")

    def export(self):
        if not self.points:
            messagebox.showerror("无法导出", "尚未采集标定点。", parent=self.root)
            return
        if self.purpose == "mount" and len(self.points) < 12:
            if not messagebox.askokcancel("点数提示", "外参最少 6 点，建议至少 12 点并覆盖近远和左右。\n继续检查并导出？", parent=self.root):
                return
        filename = filedialog.asksaveasfilename(parent=self.root, title="导出供标定程序读取的 CSV（旧文件会备份）",
                                              initialdir=self.output.parent, initialfile=self.output.name,
                                              defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not filename:
            return
        destination = Path(filename)
        camera_data = (PROJECT / "calibration" / "data").resolve()
        if destination.resolve().is_relative_to(camera_data) and not destination.resolve().is_relative_to(data_dir(self.camera).resolve()):
            messagebox.showerror("请选择本相机目录", f"不能把 {self.camera} 点位写入其他相机的数据目录。", parent=self.root)
            return
        # Session originals are evidence; only the draft writer may update session files.
        if destination.resolve().is_relative_to(self.session.resolve()):
            messagebox.showerror("请选择其他位置", "请勿覆盖本次会话的原图或草稿。", parent=self.root)
            return
        if destination.suffix.lower() != ".csv":
            messagebox.showerror("文件类型错误", "请选择 .csv 文件。", parent=self.root)
            return
        try:
            backup = export_points(destination, self.points, self.purpose)
        except Exception as exc:
            messagebox.showerror("未导出", str(exc), parent=self.root)
            return
        self.last_export, self.dirty = destination, False
        self.status.set(f"已导出：{destination}。下一步运行{'安装外参拟合' if self.purpose == 'mount' else '独立测距验证'}。")
        info = f"已保存 {len(self.points)} 个点：\n{destination}\n\n原图和会话记录：\n{self.session}"
        if backup:
            info += f"\n\n原文件已备份：\n{backup}"
        messagebox.showinfo("导出成功", info, parent=self.root)

    def show_range_report(self):
        if not self.points:
            messagebox.showerror("无法验证", "请先添加一个或多个实测点。", parent=self.root)
            return
        try:
            from calibration.ranging_report import build_camera_report, render_report, save_reports
            from camera_profiles import results_dir
            current_path, _ = load_intrinsics(self.camera)
            if file_sha256(current_path) != self.intrinsics_hash:
                raise ValueError("采集期间内参已改变，请重新启动采集工具后再验证。")
            records = [{**asdict(p), "source_file": str(self.session/"points.csv"), "source_row": i}
                       for i,p in enumerate(self.points,1)]
            report = build_camera_report(self.camera, records)
            report["collector_session"] = str(self.session)
        except Exception as exc:
            messagebox.showerror("无法生成误差报告", str(exc), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title(f"{self.camera} — 全部采集点的几何误差")
        window.geometry("1180x760")
        box = scrolledtext.ScrolledText(window, wrap="none", font=("Consolas",10))
        box.pack(fill="both", expand=True)
        box.insert("1.0", render_report(report))
        box.configure(state="disabled")
        xscroll = ttk.Scrollbar(window, orient="horizontal", command=box.xview)
        xscroll.pack(fill="x")
        box.configure(xscrollcommand=xscroll.set)
        def save():
            name = filedialog.asksaveasfilename(parent=window, title="保存 JSON 和纯文本误差报告",
                initialdir=results_dir(self.camera), initialfile=f"geometry_{self.session.name}.json",
                defaultextension=".json", filetypes=[("JSON report","*.json")])
            if not name:
                return
            try:
                output, txt, backup, txt_backup = save_reports(Path(name), report)
            except Exception as exc:
                messagebox.showerror("报告保存失败", str(exc), parent=window)
                return
            messagebox.showinfo("报告已保存", f"{output}\n{txt}\n"
                + (f"旧报告备份：{backup}\n{txt_backup}" if backup or txt_backup else ""), parent=window)
        ttk.Button(window, text="保存 JSON + TXT（不修改标定参数）", command=save).pack(pady=8)

    def close(self):
        if self.dirty and not messagebox.askokcancel(
            "尚未导出最新数据", f"最新修改尚未导出到标定 CSV；草稿已保存在：\n{self.session}\n仍要退出？", parent=self.root):
            return
        self.closed = True
        if self.feed:
            self.feed.stop.set()
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually collect surveyed points for a registered camera.")
    parser.add_argument("--camera", choices=load_profiles(), default="lingyan")
    parser.add_argument("--index", type=int)
    parser.add_argument("--purpose", choices=("mount", "validation"), default="mount")
    parser.add_argument("--image", type=Path, help="start with the selected camera's registered image instead of live capture")
    parser.add_argument("--output", type=Path, help="default export path; existing file is backed up on export")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        app = Collector(root, purpose=args.purpose, camera=args.camera, index=args.index,
                        output=args.output)
        if args.image:
            app.load_image(args.image)
        else:
            root.after(100, app.start_camera)
    except Exception as exc:
        messagebox.showerror("标定点采集启动失败", str(exc), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
