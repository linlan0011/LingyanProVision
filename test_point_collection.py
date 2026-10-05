from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from calibration.collect_mount_points import (
    CameraFeed, Collector, Point, export_points, image_pixel, measured_xyz, validate_new_point,
)
from calibration.fit_mount_pose import read_correspondences, validate_correspondences


class PointCollectionTests(unittest.TestCase):
    def points(self):
        return [Point(500 + 500 * row, -300 + 300 * col, 0,
                      100 + 150 * col, 100 + 80 * row, "原图.png")
                for row in range(4) for col in range(3)]

    def test_pixel_zoom_and_bounds(self):
        self.assertEqual(image_pixel(961, 721, 3, 640, 480), (320, 240))
        self.assertEqual(image_pixel(1279, 959, 2, 640, 480), (639, 479))
        self.assertIsNone(image_pixel(1280, 0, 2, 640, 480))
        self.assertIsNone(image_pixel(-0.1, 0, 1, 640, 480))

    def test_numeric_input_and_duplicate_guard(self):
        self.assertEqual(measured_xyz(["1000", "-300", "0"]), (1000, -300, 0))
        for values in (["", "0", "0"], ["NaN", "0", "0"], ["0", "inf", "0"]):
            with self.assertRaises(ValueError):
                measured_xyz(values)
        point = self.points()[0]
        validate_new_point(point, [], (640, 480))
        with self.assertRaises(ValueError):
            validate_new_point(point, [point], (640, 480))
        with self.assertRaises(ValueError):
            validate_new_point(Point(20, 30, 0, 640, 480, "x"), [], (640, 480))

    def test_csv_backup_and_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "点位.csv"
            path.write_text("previous user data", encoding="utf-8")
            backup = export_points(path, self.points(), "mount")
            self.assertEqual(backup.read_text(encoding="utf-8"), "previous user data")
            xyz, uv = read_correspondences(path)
            self.assertEqual(xyz.shape, (12, 3))
            np.testing.assert_array_equal(uv[0], [100, 100])
            self.assertEqual(len(list(Path(temp).glob("*.backup_*.csv"))), 1)

    def test_invalid_export_leaves_original_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "points.csv"
            path.write_text("untouched", encoding="utf-8")
            with self.assertRaises(ValueError):
                export_points(path, self.points()[:2], "mount")
            self.assertEqual(path.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(len(list(Path(temp).iterdir())), 1)
            export_points(path, self.points()[:1], "validation")

    def test_degenerate_correspondences_rejected(self):
        xyz = np.array([(p.x_mm, p.y_mm, p.z_mm) for p in self.points()])
        uv = np.array([(p.u_px, p.v_px) for p in self.points()])
        validate_correspondences(xyz, uv)  # Ground-plane points are valid.
        for bad_xyz, bad_uv in (
            (xyz, np.zeros_like(uv)),
            (np.zeros_like(xyz), uv),
            (np.column_stack([np.arange(12), np.zeros((12, 2))]), uv),
            (xyz, np.column_stack([np.arange(12), np.arange(12)])),
            (xyz, np.full((12, 2), np.nan)),
        ):
            with self.assertRaises(ValueError):
                validate_correspondences(bad_xyz, bad_uv)

    def test_camera_worker_capture_and_release(self):
        released = threading.Event()

        class FakeCamera:
            def read(self):
                time.sleep(0.005)
                return True, np.zeros((480, 640, 3), np.uint8)

            def release(self):
                released.set()

        with patch("calibration.collect_mount_points.open_camera", return_value=FakeCamera()):
            feed = CameraFeed(0)
            deadline = time.monotonic() + 2
            while feed.latest()[0] is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(feed.latest()[0].shape, (480, 640, 3))
            feed.stop.set()
            feed.thread.join(timeout=2)
            self.assertTrue(released.is_set())

    def test_camera_rejects_wrong_resolution(self):
        class WrongCamera:
            def read(self):
                return True, np.zeros((1024, 768, 3), np.uint8)

            def release(self):
                pass

        with patch("calibration.collect_mount_points.open_camera", return_value=WrongCamera()):
            feed = CameraFeed(0)
            feed.thread.join(timeout=2)
            frame, error = feed.latest()
            self.assertIsNone(frame)
            self.assertIn("640×480", error)

    def test_gui_offline_add_delete_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            intrinsics = folder / "intrinsics.json"
            intrinsics.write_text(json.dumps({"image_width": 640, "image_height": 480,
                                              "model": "opencv_pinhole"}), encoding="utf-8")
            raw = np.full((480, 640, 3), 235, np.uint8)
            cv2.line(raw, (300, 240), (340, 240), (0, 0, 0), 2)
            cv2.line(raw, (320, 220), (320, 260), (0, 0, 0), 2)
            raw_path = folder / "原图.png"
            raw_path.write_bytes(cv2.imencode(".png", raw)[1].tobytes())
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            output = folder / "range_validation.csv"
            app = Collector(root, purpose="validation", intrinsics=intrinsics,
                            output=output, sessions=folder / "sessions")
            try:
                app.load_image(raw_path)
                app.zoom.set(2)
                app.render()
                event = SimpleNamespace(x=640, y=480)
                app.click(event)
                app.hover(event)
                self.assertEqual(app.selected, (320, 240))
                for variable, value in zip(app.xyz, ("1000", "-200", "0")):
                    variable.set(value)
                with patch("calibration.collect_mount_points.messagebox.showerror") as errors:
                    app.add_point()
                    errors.assert_not_called()
                self.assertEqual(len(app.points), 1)
                self.assertTrue(app.dirty)
                self.assertEqual(len(app.table.get_children()), 1)
                saved = cv2.imdecode(np.frombuffer(Path(app.frame_file).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(saved, raw)  # Never save the overlay or zoom.
                with (app.session / "points.csv").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(rows[0]["u_px"], "320")
                with patch("calibration.collect_mount_points.filedialog.asksaveasfilename", return_value=str(output)), \
                     patch("calibration.collect_mount_points.messagebox.showinfo"):
                    app.export()
                self.assertTrue(output.exists())
                self.assertFalse(app.dirty)
                app.table.selection_set("0")
                app.delete_points()
                self.assertEqual(len(app.points), 0)
                self.assertTrue(output.exists())  # Deleting a draft row does not rewrite the export.
            finally:
                app.dirty = False
                app.close()

    def test_live_freeze_uses_an_unmodified_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            intrinsics = folder / "intrinsics.json"
            intrinsics.write_text(json.dumps({"image_width": 640, "image_height": 480,
                                              "model": "opencv_pinhole"}), encoding="utf-8")
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            app = Collector(root, intrinsics=intrinsics, sessions=folder / "sessions")
            try:
                frame = np.zeros((480, 640, 3), np.uint8)
                app.frame = frame
                app.click(SimpleNamespace(x=100, y=100))
                self.assertIsNone(app.selected)  # Live clicks cannot become measurements.
                app.toggle_freeze()
                self.assertTrue(app.frozen)
                frame[:] = 255
                self.assertTrue(np.all(app.frame == 0))
                app.zoom.set(3)
                app.render()
                root.update_idletasks()
                app.canvas.xview_moveto(0.25)
                app.canvas.yview_moveto(0.25)
                event = SimpleNamespace(x=50, y=70)
                expected = image_pixel(app.canvas.canvasx(50), app.canvas.canvasy(70), 3, 640, 480)
                app.click(event)
                self.assertEqual(app.selected, expected)
                bad_image = folder / "bad.png"
                bad_image.write_bytes(cv2.imencode(".png", np.zeros((768, 1024, 3), np.uint8))[1].tobytes())
                with self.assertRaises(ValueError):
                    app.load_image(bad_image)
            finally:
                app.dirty = False
                app.close()


if __name__ == "__main__":
    unittest.main()
