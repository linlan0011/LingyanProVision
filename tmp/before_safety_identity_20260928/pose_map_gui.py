"""Interactive field-map pose selector for the LINGYAN vision program."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np

from field_zones import FieldZone, read_zones, write_zones_atomic
from pose_io import Pose2D, normalize_heading_deg, write_pose_atomic

DEFAULT_MAP = Path(__file__).resolve().with_name("field_map.png")
# Pixel coordinates of the 3000 x 3000 mm field border in the supplied diagram.
DEFAULT_FIELD_PIXELS = (27.0, 22.0, 749.0, 743.0)


class CoordinateTransform:
    """Map between a world rectangle and a fitted canvas rectangle."""

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 3000.0,
        y_min: float = 0.0,
        y_max: float = 3000.0,
    ) -> None:
        self.set_world_bounds(x_min, x_max, y_min, y_max)
        self.set_canvas_bounds(0.0, 0.0, 1.0, 1.0)

    def set_world_bounds(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
            raise ValueError("坐标范围必须是有限数字")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("最大坐标必须大于最小坐标")
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

    def set_canvas_bounds(self, left: float, top: float, right: float, bottom: float) -> None:
        if right <= left or bottom <= top:
            raise ValueError("invalid canvas bounds")
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def contains_canvas(self, px: float, py: float) -> bool:
        return self.left <= px <= self.right and self.top <= py <= self.bottom

    def canvas_to_world(self, px: float, py: float, clamp: bool = False) -> tuple[float, float]:
        if clamp:
            px = min(max(px, self.left), self.right)
            py = min(max(py, self.top), self.bottom)
        x_ratio = (px - self.left) / (self.right - self.left)
        y_ratio = (py - self.top) / (self.bottom - self.top)
        return (
            self.x_min + x_ratio * (self.x_max - self.x_min),
            self.y_max - y_ratio * (self.y_max - self.y_min),
        )

    def world_to_canvas(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        x_ratio = (x_mm - self.x_min) / (self.x_max - self.x_min)
        y_ratio = (self.y_max - y_mm) / (self.y_max - self.y_min)
        return (
            self.left + x_ratio * (self.right - self.left),
            self.top + y_ratio * (self.bottom - self.top),
        )


def heading_from_points(origin: tuple[float, float], target: tuple[float, float]) -> float:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        raise ValueError("direction needs two distinct points")
    return normalize_heading_deg(math.degrees(math.atan2(dy, dx)))


def read_image_unicode(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


class PoseMapApp:
    MAP_MARGIN = 34

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("场地位姿与识别分区 - Lingyan Pro Vision")
        self.root.minsize(900, 640)
        self.output_path = Path(args.output).expanduser().resolve()
        self.zones_path = Path(args.zones).expanduser().resolve()
        self.transform = CoordinateTransform(args.x_min, args.x_max, args.y_min, args.y_max)
        self.pose = Pose2D(args.x, args.y, args.heading).validated()
        self.map_path: Optional[Path] = None
        self.map_image: Optional[np.ndarray] = None
        self.field_pixels: Optional[tuple[float, float, float, float]] = None
        self.tk_map_image: Optional[tk.PhotoImage] = None
        self.drag_origin_world: Optional[tuple[float, float]] = None
        self.drag_origin_canvas: Optional[tuple[float, float]] = None
        self.zone_drag_origin: Optional[tuple[float, float]] = None
        self.zone_preview_bounds: Optional[tuple[float, float, float, float]] = None
        self.zones: list[FieldZone] = []
        self.zone_mode = False
        self.zone_window: Optional[tk.Toplevel] = None
        self.zone_tree: Optional[ttk.Treeview] = None
        self.editing_zone_name: Optional[str] = None
        self._updating_entries = False
        self._resize_job: Optional[str] = None
        self._initial_draw_job: Optional[str] = None

        self.x_var = tk.StringVar(value=f"{self.pose.x_mm:.1f}")
        self.y_var = tk.StringVar(value=f"{self.pose.y_mm:.1f}")
        self.heading_var = tk.StringVar(value=f"{self.pose.heading_deg:.1f}")
        self.hover_var = tk.StringVar(value="光标坐标：—")
        self.status_var = tk.StringVar(value="尚未输出；点击地图选点，拖动可指定航向")
        self.output_var = tk.StringVar(value=str(self.output_path))
        self.zone_mode_var = tk.StringVar(value="模式：选择位姿")
        self.zone_vars = {
            key: tk.StringVar() for key in (
                "name", "x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm",
                "heading_deg", "heading_tolerance_deg",
            )
        }
        self.zone_vars["heading_deg"].set(f"{self.pose.heading_deg:.1f}")
        self.zone_vars["heading_tolerance_deg"].set("45")
        self.bounds_vars = {
            "x_min": tk.StringVar(value=f"{args.x_min:g}"),
            "x_max": tk.StringVar(value=f"{args.x_max:g}"),
            "y_min": tk.StringVar(value=f"{args.y_min:g}"),
            "y_max": tk.StringVar(value=f"{args.y_max:g}"),
        }
        self.pixel_vars = {
            key: tk.StringVar() for key in ("left", "top", "right", "bottom")
        }

        self._build_ui()
        for variable in (self.x_var, self.y_var, self.heading_var):
            variable.trace_add("write", self._manual_pose_changed)
        for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm"):
            self.zone_vars[key].trace_add("write", self._zone_form_changed)
        self.zone_vars["heading_deg"].trace_add("write", self._zone_form_changed)
        if self.zones_path.is_file():
            try:
                self.zones = read_zones(self.zones_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                messagebox.showerror("分区文件读取失败", f"{self.zones_path}\n{exc}")
                self.status_var.set("分区文件读取失败；请修复文件后重新启动")
                self.zones_path = None
        map_path = Path(args.map) if args.map else DEFAULT_MAP
        if map_path.is_file():
            self.load_map(map_path)
        self._initial_draw_job = self.root.after_idle(self.redraw)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
            self._resize_job = None
        if self._initial_draw_job is not None:
            self.root.after_cancel(self._initial_draw_job)
            self._initial_draw_job = None
        self.root.destroy()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main = ttk.Frame(self.root, padding=10)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, background="#20242b", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _: self.hover_var.set("光标坐标：—"))
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        side = ttk.Frame(main, width=270)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(side, text="选定位姿", font=("Microsoft YaHei UI", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        row += 1
        for label, variable, unit in (
            ("X", self.x_var, "mm"),
            ("Y", self.y_var, "mm"),
            ("航向角", self.heading_var, "°"),
        ):
            ttk.Label(side, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(side, textvariable=variable, width=15)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            ttk.Label(side, text=unit).grid(row=row, column=2, sticky="w", padx=(4, 0))
            row += 1
        ttk.Label(side, text="0° 向右（+X），90° 向上（+Y）", foreground="#555555").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 10)
        )
        row += 1
        ttk.Button(side, text="确认并输出给识别程序", command=self.publish_pose).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 12)
        )
        row += 1
        ttk.Button(side, text="识别情境分区管理…", command=self.open_zone_manager).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4)
        )
        row += 1
        ttk.Label(side, textvariable=self.zone_mode_var, foreground="#147d8b").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1
        ttk.Separator(side).grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1
        ttk.Label(side, text="地图与坐标标定", font=("Microsoft YaHei UI", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(6, 5)
        )
        row += 1
        ttk.Button(side, text="选择地图图片…", command=self.choose_map).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 5)
        )
        row += 1
        for key, label in (("x_min", "X 最小"), ("x_max", "X 最大"), ("y_min", "Y 最小"), ("y_max", "Y 最大")):
            ttk.Label(side, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(side, textvariable=self.bounds_vars[key], width=15).grid(
                row=row, column=1, columnspan=2, sticky="ew", pady=2
            )
            row += 1
        ttk.Button(side, text="应用坐标范围", command=self.apply_bounds).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(5, 10)
        )
        row += 1
        ttk.Label(side, text="场地外框在图片中的像素位置").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 3)
        )
        row += 1
        for key, label in (("left", "左"), ("top", "上"), ("right", "右"), ("bottom", "下")):
            ttk.Label(side, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(side, textvariable=self.pixel_vars[key], width=15).grid(
                row=row, column=1, columnspan=2, sticky="ew", pady=2
            )
            row += 1
        ttk.Button(side, text="应用图片外框", command=self.apply_field_pixels).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(5, 10)
        )
        row += 1
        ttk.Label(side, text="输出文件", font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1
        ttk.Label(side, textvariable=self.output_var, wraplength=255, foreground="#555555").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 10)
        )
        row += 1
        ttk.Label(
            side,
            text="操作：单击选位置；按住左键拖向目标方向可同时设定航向。手动输入会立即更新地图标记。",
            wraplength=255,
            justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        status = ttk.Frame(main)
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.hover_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.status_var, anchor="e").grid(row=0, column=1, sticky="ew")

    def _on_resize(self, _: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self.redraw)

    def _fit_map_rectangle(self) -> tuple[float, float, float, float]:
        width = max(2, self.canvas.winfo_width() - 2 * self.MAP_MARGIN)
        height = max(2, self.canvas.winfo_height() - 2 * self.MAP_MARGIN)
        if self.map_image is not None:
            image_h, image_w = self.map_image.shape[:2]
            aspect = image_w / image_h
        else:
            aspect = (self.transform.x_max - self.transform.x_min) / (
                self.transform.y_max - self.transform.y_min
            )
        fitted_w = min(width, height * aspect)
        fitted_h = fitted_w / aspect
        if fitted_h > height:
            fitted_h = height
            fitted_w = fitted_h * aspect
        left = (self.canvas.winfo_width() - fitted_w) / 2
        top = (self.canvas.winfo_height() - fitted_h) / 2
        return left, top, left + fitted_w, top + fitted_h

    def redraw(self) -> None:
        self.canvas.delete("all")
        left, top, right, bottom = self._fit_map_rectangle()
        if self.map_image is not None:
            image_h, image_w = self.map_image.shape[:2]
            field_left, field_top, field_right, field_bottom = self.field_pixels or (
                0.0, 0.0, float(image_w), float(image_h)
            )
            scale_x = (right - left) / image_w
            scale_y = (bottom - top) / image_h
            self.transform.set_canvas_bounds(
                left + field_left * scale_x,
                top + field_top * scale_y,
                left + field_right * scale_x,
                top + field_bottom * scale_y,
            )
            width, height = max(1, int(right - left)), max(1, int(bottom - top))
            resized = cv2.resize(self.map_image, (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ok, ppm = cv2.imencode(".ppm", rgb)
            if ok:
                self.tk_map_image = tk.PhotoImage(data=ppm.tobytes(), format="PPM")
                self.canvas.create_image(left, top, image=self.tk_map_image, anchor="nw")
        else:
            self.transform.set_canvas_bounds(left, top, right, bottom)
            self.canvas.create_rectangle(left, top, right, bottom, fill="#f5f5f2", outline="")
            for index in range(11):
                ratio = index / 10
                px = left + ratio * (right - left)
                py = top + ratio * (bottom - top)
                color = "#cbd0d6" if index % 5 else "#8d969f"
                self.canvas.create_line(px, top, px, bottom, fill=color)
                self.canvas.create_line(left, py, right, py, fill=color)
            self.canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="尚未加载场地地图\n点击右侧“选择地图图片”",
                fill="#737b84",
                justify="center",
                font=("Microsoft YaHei UI", 15),
            )
        self.canvas.create_rectangle(
            self.transform.left, self.transform.top,
            self.transform.right, self.transform.bottom,
            outline="#e4572e", width=2,
        )
        self._draw_zones()
        self._draw_axes()
        self._draw_pose()

    def _draw_zones(self) -> None:
        for zone in self.zones:
            x1, y1 = self.transform.world_to_canvas(zone.x_min_mm, zone.y_max_mm)
            x2, y2 = self.transform.world_to_canvas(zone.x_max_mm, zone.y_min_mm)
            selected = zone.name == self.editing_zone_name
            color = "#f5a623" if selected else "#008a99"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if selected else 2)
            center_x, center_y = self.transform.world_to_canvas(
                (zone.x_min_mm + zone.x_max_mm) / 2,
                (zone.y_min_mm + zone.y_max_mm) / 2,
            )
            angle = math.radians(zone.heading_deg)
            self.canvas.create_line(
                center_x, center_y,
                center_x + 23 * math.cos(angle), center_y - 23 * math.sin(angle),
                fill=color, width=3, arrow=tk.LAST,
            )
            self.canvas.create_text(
                x1 + 3, y1 + 3,
                text=f"{zone.name}  {zone.heading_deg:g}°±{zone.heading_tolerance_deg:g}°",
                anchor="nw", fill=color, font=("Microsoft YaHei UI", 10, "bold"),
            )
        if self.zone_preview_bounds is not None:
            x_min, x_max, y_min, y_max = self.zone_preview_bounds
            x1, y1 = self.transform.world_to_canvas(x_min, y_max)
            x2, y2 = self.transform.world_to_canvas(x_max, y_min)
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#ff7b00", width=3, dash=(6, 4)
            )
            try:
                angle = math.radians(float(self.zone_vars["heading_deg"].get()))
                if math.isfinite(angle):
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    self.canvas.create_line(
                        center_x, center_y,
                        center_x + 23 * math.cos(angle), center_y - 23 * math.sin(angle),
                        fill="#ff7b00", width=3, arrow=tk.LAST,
                    )
            except ValueError:
                pass

    def _draw_axes(self) -> None:
        left, top, right, bottom = (
            self.transform.left,
            self.transform.top,
            self.transform.right,
            self.transform.bottom,
        )
        self.canvas.create_text(left, bottom + 16, text=f"({self.transform.x_min:g}, {self.transform.y_min:g})", fill="#d9dde2", anchor="w")
        self.canvas.create_text(right, bottom + 16, text=f"({self.transform.x_max:g}, {self.transform.y_min:g})", fill="#d9dde2", anchor="e")
        self.canvas.create_text(left, top - 16, text=f"Y={self.transform.y_max:g}", fill="#d9dde2", anchor="w")

    def _draw_pose(self) -> None:
        px, py = self.transform.world_to_canvas(self.pose.x_mm, self.pose.y_mm)
        angle = math.radians(self.pose.heading_deg)
        length = max(36.0, min(self.transform.right - self.transform.left, self.transform.bottom - self.transform.top) * 0.10)
        end_x = px + length * math.cos(angle)
        end_y = py - length * math.sin(angle)
        self.canvas.create_oval(px - 7, py - 7, px + 7, py + 7, fill="#ff3b30", outline="white", width=2, tags="pose")
        self.canvas.create_line(px, py, end_x, end_y, fill="#ff3b30", width=4, arrow=tk.LAST, arrowshape=(13, 16, 6), tags="pose")
        self.canvas.create_text(px + 10, py + 12, text=f"({self.pose.x_mm:.1f}, {self.pose.y_mm:.1f}, {self.pose.heading_deg:.1f}°)", fill="#ff3b30", anchor="nw", font=("Microsoft YaHei UI", 10, "bold"), tags="pose")

    def _manual_pose_changed(self, *_: object) -> None:
        if self._updating_entries:
            return
        try:
            candidate = Pose2D(float(self.x_var.get()), float(self.y_var.get()), float(self.heading_var.get())).validated()
        except ValueError:
            self.status_var.set("输入尚未构成有效数字")
            return
        self.pose = candidate
        self.status_var.set("手动输入已更新地图；点击“确认并输出”提交")
        self.redraw()

    def _set_pose(self, pose: Pose2D, message: str) -> None:
        self.pose = pose.validated()
        self._updating_entries = True
        try:
            self.x_var.set(f"{self.pose.x_mm:.1f}")
            self.y_var.set(f"{self.pose.y_mm:.1f}")
            self.heading_var.set(f"{self.pose.heading_deg:.1f}")
        finally:
            self._updating_entries = False
        self.status_var.set(message)
        self.redraw()

    def _on_motion(self, event: tk.Event) -> None:
        if not self.transform.contains_canvas(event.x, event.y):
            self.hover_var.set("光标坐标：—")
            return
        x_mm, y_mm = self.transform.canvas_to_world(event.x, event.y)
        self.hover_var.set(f"光标坐标：X={x_mm:.1f} mm，Y={y_mm:.1f} mm")

    def _on_press(self, event: tk.Event) -> None:
        if not self.transform.contains_canvas(event.x, event.y):
            return
        x_mm, y_mm = self.transform.canvas_to_world(event.x, event.y, clamp=True)
        if self.zone_mode:
            self.zone_drag_origin = (x_mm, y_mm)
            self.zone_preview_bounds = None
            self.status_var.set("正在框选分区；拖动到矩形的另一角")
            return
        self.drag_origin_world = (x_mm, y_mm)
        self.drag_origin_canvas = (event.x, event.y)
        self._set_pose(Pose2D(x_mm, y_mm, self.pose.heading_deg), "已选位置；拖动可设置航向")

    def _on_drag(self, event: tk.Event) -> None:
        if self.zone_mode and self.zone_drag_origin is not None:
            target = self.transform.canvas_to_world(event.x, event.y, clamp=True)
            x0, y0 = self.zone_drag_origin
            self.zone_preview_bounds = (
                min(x0, target[0]), max(x0, target[0]),
                min(y0, target[1]), max(y0, target[1]),
            )
            self.redraw()
            return
        if self.drag_origin_world is None or self.drag_origin_canvas is None:
            return
        if math.hypot(event.x - self.drag_origin_canvas[0], event.y - self.drag_origin_canvas[1]) < 5:
            return
        target = self.transform.canvas_to_world(event.x, event.y, clamp=True)
        try:
            heading = heading_from_points(self.drag_origin_world, target)
        except ValueError:
            return
        self._set_pose(Pose2D(*self.drag_origin_world, heading), "位置和航向已更新；松开后可确认输出")

    def _on_release(self, _: tk.Event) -> None:
        if self.zone_mode and self.zone_drag_origin is not None:
            self.zone_drag_origin = None
            bounds = self.zone_preview_bounds
            if bounds is None or bounds[1] - bounds[0] < 1 or bounds[3] - bounds[2] < 1:
                self.zone_preview_bounds = None
                self.status_var.set("分区太小；请拖出一个矩形范围")
                self.redraw()
            else:
                for key, value in zip(
                    ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm"), bounds
                ):
                    self.zone_vars[key].set(f"{value:.1f}")
                self.status_var.set("分区范围已框选；请填写名称和航向角后保存")
                if self.zone_window is not None:
                    self.zone_window.lift()
            return
        self.drag_origin_world = None
        self.drag_origin_canvas = None

    def choose_map(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择场地地图",
            filetypes=(("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")),
        )
        if selected:
            self.load_map(Path(selected))

    def load_map(self, path: Path) -> None:
        try:
            image = read_image_unicode(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("地图加载失败", str(exc))
            return
        self.map_image = image
        self.map_path = path.resolve()
        image_h, image_w = image.shape[:2]
        self.field_pixels = (
            DEFAULT_FIELD_PIXELS if self.map_path == DEFAULT_MAP
            else (0.0, 0.0, float(image_w), float(image_h))
        )
        for key, value in zip(self.pixel_vars, self.field_pixels):
            self.pixel_vars[key].set(f"{value:g}")
        self.status_var.set(f"已加载地图：{self.map_path.name}")
        self.redraw()

    def apply_field_pixels(self) -> None:
        if self.map_image is None:
            messagebox.showerror("无法标定", "请先加载地图图片")
            return
        try:
            left, top, right, bottom = (
                float(self.pixel_vars[key].get())
                for key in ("left", "top", "right", "bottom")
            )
            image_h, image_w = self.map_image.shape[:2]
            if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                raise ValueError("图片外框必须是有限数字")
            if not (0 <= left < right <= image_w and 0 <= top < bottom <= image_h):
                raise ValueError(f"图片外框必须位于 {image_w} × {image_h} 像素图片内")
        except ValueError as exc:
            messagebox.showerror("图片外框无效", str(exc))
            return
        self.field_pixels = (left, top, right, bottom)
        self.status_var.set("场地图片外框已应用")
        self.redraw()

    def apply_bounds(self) -> None:
        try:
            self.transform.set_world_bounds(
                float(self.bounds_vars["x_min"].get()),
                float(self.bounds_vars["x_max"].get()),
                float(self.bounds_vars["y_min"].get()),
                float(self.bounds_vars["y_max"].get()),
            )
        except ValueError as exc:
            messagebox.showerror("坐标范围无效", str(exc))
            return
        self.status_var.set("坐标标定范围已应用")
        self.redraw()

    def open_zone_manager(self) -> None:
        if self.zone_window is not None and self.zone_window.winfo_exists():
            self.zone_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.zone_window = window
        window.title("识别情境分区管理")
        window.geometry("480x650")
        window.minsize(440, 570)
        window.protocol("WM_DELETE_WINDOW", self.close_zone_manager)
        panel = ttk.Frame(window, padding=12)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(2, weight=1)

        ttk.Label(
            panel, text="框选矩形，填写名称与参考航向角",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(panel, text="开始在地图上框选", command=self.start_zone_selection).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )

        tree = ttk.Treeview(
            panel, columns=("name", "xy", "heading"), show="headings", height=7,
            selectmode="browse",
        )
        self.zone_tree = tree
        for key, title, width in (
            ("name", "名称", 110),
            ("xy", "范围 X / Y (mm)", 220),
            ("heading", "航向", 100),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        tree.grid(row=2, column=0, columnspan=3, sticky="nsew")
        tree.bind("<<TreeviewSelect>>", self._on_zone_selected)
        self._refresh_zone_tree()

        row = 3
        for key, title in (
            ("name", "分区名称"),
            ("x_min_mm", "X 最小 (mm)"),
            ("x_max_mm", "X 最大 (mm)"),
            ("y_min_mm", "Y 最小 (mm)"),
            ("y_max_mm", "Y 最大 (mm)"),
            ("heading_deg", "参考航向角 (°)"),
            ("heading_tolerance_deg", "航向容差 (°)"),
        ):
            ttk.Label(panel, text=title).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(panel, textvariable=self.zone_vars[key]).grid(
                row=row, column=1, columnspan=2, sticky="ew", pady=3
            )
            row += 1
        ttk.Label(
            panel,
            text="参考航向角：0° 向右，90° 向上；容差决定允许偏离的角度。",
            wraplength=440,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(5, 8))
        row += 1
        ttk.Button(panel, text="新建", command=self.new_zone).grid(row=row, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(panel, text="保存分区", command=self.save_zone).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(panel, text="删除选中分区", command=self.delete_zone).grid(
            row=row, column=2, sticky="ew", padx=(4, 0)
        )
        row += 1
        ttk.Label(panel, text=f"保存文件：{self.zones_path}", wraplength=440).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

    def close_zone_manager(self) -> None:
        self.zone_mode = False
        self.zone_mode_var.set("模式：选择位姿")
        self.zone_drag_origin = None
        self.zone_preview_bounds = None
        if self.zone_window is not None:
            self.zone_window.destroy()
        self.zone_window = None
        self.zone_tree = None
        self.redraw()

    def start_zone_selection(self) -> None:
        self.new_zone()
        self.zone_mode = True
        self.zone_mode_var.set("模式：框选分区（在地图上拖动）")
        self.zone_drag_origin = None
        self.zone_preview_bounds = None
        self.editing_zone_name = None
        self.status_var.set("在地图内按住左键，拖出矩形分区")
        self.root.lift()
        self.redraw()

    def _zone_form_changed(self, *_: object) -> None:
        try:
            x_min, x_max, y_min, y_max = (
                float(self.zone_vars[key].get())
                for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm")
            )
            if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
                return
            if x_min >= x_max or y_min >= y_max:
                return
        except ValueError:
            return
        self.zone_preview_bounds = (x_min, x_max, y_min, y_max)
        self.redraw()

    def _refresh_zone_tree(self, selected_name: Optional[str] = None) -> None:
        if self.zone_tree is None:
            return
        for item in self.zone_tree.get_children():
            self.zone_tree.delete(item)
        for index, zone in enumerate(self.zones):
            self.zone_tree.insert(
                "", "end", iid=str(index),
                values=(
                    zone.name,
                    f"{zone.x_min_mm:g}–{zone.x_max_mm:g} / {zone.y_min_mm:g}–{zone.y_max_mm:g}",
                    f"{zone.heading_deg:g}° ±{zone.heading_tolerance_deg:g}°",
                ),
            )
            if zone.name == selected_name:
                self.zone_tree.selection_set(str(index))
                self.zone_tree.see(str(index))

    def _on_zone_selected(self, _: tk.Event) -> None:
        if self.zone_tree is None or not self.zone_tree.selection():
            return
        zone = self.zones[int(self.zone_tree.selection()[0])]
        self.editing_zone_name = zone.name
        for key in self.zone_vars:
            value = getattr(zone, key)
            self.zone_vars[key].set(str(value))
        self.zone_preview_bounds = None
        self.redraw()

    def new_zone(self) -> None:
        self.editing_zone_name = None
        if self.zone_tree is not None:
            self.zone_tree.selection_remove(*self.zone_tree.selection())
        self.zone_vars["name"].set("")
        self.zone_vars["heading_deg"].set(f"{self.pose.heading_deg:.1f}")
        self.zone_vars["heading_tolerance_deg"].set("45")
        for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm"):
            self.zone_vars[key].set("")
        self.zone_preview_bounds = None
        self.redraw()

    def save_zone(self) -> None:
        if self.zones_path is None:
            messagebox.showerror("无法保存", "当前分区文件读取失败，请修复文件后重新启动")
            return
        try:
            zone = FieldZone(
                name=self.zone_vars["name"].get(),
                x_min_mm=float(self.zone_vars["x_min_mm"].get()),
                x_max_mm=float(self.zone_vars["x_max_mm"].get()),
                y_min_mm=float(self.zone_vars["y_min_mm"].get()),
                y_max_mm=float(self.zone_vars["y_max_mm"].get()),
                heading_deg=float(self.zone_vars["heading_deg"].get()),
                heading_tolerance_deg=float(self.zone_vars["heading_tolerance_deg"].get()),
            ).validated()
            if not (
                self.transform.x_min <= zone.x_min_mm < zone.x_max_mm <= self.transform.x_max
                and self.transform.y_min <= zone.y_min_mm < zone.y_max_mm <= self.transform.y_max
            ):
                raise ValueError("分区范围必须位于当前场地坐标范围内")
            updated = [
                existing for existing in self.zones
                if existing.name != self.editing_zone_name
            ]
            if any(existing.name.casefold() == zone.name.casefold() for existing in updated):
                raise ValueError(f"分区名称已存在：{zone.name}")
            updated.append(zone)
            write_zones_atomic(self.zones_path, updated)
        except (OSError, ValueError) as exc:
            messagebox.showerror("分区保存失败", str(exc))
            return
        self.zones = updated
        self.editing_zone_name = zone.name
        self.zone_mode = False
        self.zone_mode_var.set("模式：选择位姿")
        self.zone_preview_bounds = None
        self._refresh_zone_tree(zone.name)
        self.status_var.set(f"分区“{zone.name}”已保存")
        self.redraw()

    def delete_zone(self) -> None:
        if self.zones_path is None or self.editing_zone_name is None:
            return
        if not messagebox.askyesno("删除分区", f"确定删除“{self.editing_zone_name}”吗？"):
            return
        try:
            updated = [zone for zone in self.zones if zone.name != self.editing_zone_name]
            write_zones_atomic(self.zones_path, updated)
        except (OSError, ValueError) as exc:
            messagebox.showerror("分区删除失败", str(exc))
            return
        deleted_name = self.editing_zone_name
        self.zones = updated
        self.editing_zone_name = None
        self._refresh_zone_tree()
        self.new_zone()
        self.status_var.set(f"分区“{deleted_name}”已删除")
        self.redraw()

    def publish_pose(self) -> None:
        if not (
            self.transform.x_min <= self.pose.x_mm <= self.transform.x_max
            and self.transform.y_min <= self.pose.y_mm <= self.transform.y_max
        ):
            messagebox.showerror("无法输出", "所选 X/Y 超出当前地图坐标范围")
            return
        try:
            write_pose_atomic(self.output_path, self.pose)
        except OSError as exc:
            messagebox.showerror("输出失败", str(exc))
            return
        self.status_var.set(
            f"已输出 X={self.pose.x_mm:.1f}, Y={self.pose.y_mm:.1f}, 航向={self.pose.heading_deg:.1f}°"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在场地地图上选择机器人初始位姿")
    parser.add_argument("--map", help="场地地图 PNG/JPG/BMP 路径；默认使用项目内 field_map.png")
    parser.add_argument("--output", default="selected_pose.json", help="给识别程序读取的 JSON 文件")
    parser.add_argument("--zones", default="field_zones.json", help="命名分区 JSON 文件")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=3000.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=3000.0)
    parser.add_argument("--x", type=float, default=1500.0)
    parser.add_argument("--y", type=float, default=1500.0)
    parser.add_argument("--heading", type=float, default=45.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = tk.Tk()
    PoseMapApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
