from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

from field_zones import FieldZone, ZoneFileReader, matching_zones, read_zones, write_zones_atomic
from pose_io import Pose2D, PoseFileReader, normalize_heading_deg, read_pose, write_pose_atomic
from pose_map_gui import CoordinateTransform, heading_from_points, PoseMapApp, build_parser


class CoordinateTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = CoordinateTransform(0, 3000, 0, 3000)
        self.transform.set_canvas_bounds(100, 50, 700, 650)

    def test_corners_use_bottom_left_world_origin(self) -> None:
        self.assertEqual(self.transform.canvas_to_world(100, 650), (0.0, 0.0))
        self.assertEqual(self.transform.canvas_to_world(700, 50), (3000.0, 3000.0))

    def test_round_trip(self) -> None:
        canvas = self.transform.world_to_canvas(725.5, 2100.25)
        world = self.transform.canvas_to_world(*canvas)
        self.assertAlmostEqual(world[0], 725.5)
        self.assertAlmostEqual(world[1], 2100.25)

    def test_heading_convention(self) -> None:
        self.assertEqual(heading_from_points((0, 0), (1, 0)), 0.0)
        self.assertEqual(heading_from_points((0, 0), (0, 1)), 90.0)
        self.assertEqual(heading_from_points((0, 0), (0, -1)), -90.0)
        self.assertEqual(normalize_heading_deg(270), -90.0)


class PoseIoTests(unittest.TestCase):
    def test_same_timestamp_pose_update_is_not_missed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            write_pose_atomic(path, Pose2D(2200, 200, 180))
            reader = PoseFileReader(path)
            reader.poll()
            stamp = path.stat().st_mtime_ns
            write_pose_atomic(path, Pose2D(2200, 200, 170))
            os.utime(path, ns=(stamp, stamp))
            self.assertEqual(reader.poll().heading_deg, 170)

    def test_atomic_json_round_trip_and_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_pose.json"
            write_pose_atomic(path, Pose2D(123.4567, 987.6543, 270))
            pose = read_pose(path)
            self.assertEqual(pose, Pose2D(123.457, 987.654, -90.0))
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["coordinate_frame"], "field_bottom_left_mm")
            reader = PoseFileReader(path)
            self.assertEqual(reader.poll(), pose)
            self.assertIsNone(reader.error)
            path.unlink()
            self.assertIsNone(reader.poll())
            self.assertEqual(reader.error, "waiting for pose file")


class LivePoseUiTests(unittest.TestCase):
    def test_click_drag_numeric_input_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'pose.json'
            root = tk.Tk()
            root.withdraw()
            app = PoseMapApp(root, build_parser().parse_args(['--output',str(output)]))
            root.update()
            def wait(ms):
                root.after(ms,root.quit)
                root.mainloop()
            try:
                with patch.object(app,'redraw'):
                    app.transform.set_canvas_bounds(0,0,600,600)
                    app._on_press(SimpleNamespace(x=440,y=560))
                    wait(90)
                    reader=PoseFileReader(output)
                    self.assertEqual((reader.poll().x_mm,reader.pose.y_mm),(2200,200))
                    app._on_drag(SimpleNamespace(x=400,y=560))
                    wait(90)  # Publishes before mouse release.
                    self.assertEqual(reader.poll().heading_deg,-180)
                    app._on_drag(SimpleNamespace(x=440,y=520))
                    app._on_release(None)
                    self.assertEqual(reader.poll().heading_deg,90)
                    app.x_var.set('2100')
                    app.x_var.set('-')
                    wait(240)
                    self.assertEqual(reader.poll().x_mm,2200)
                    app.x_var.set('2100')
                    wait(240)
                    self.assertEqual(reader.poll().x_mm,2100)
                    app.x_var.set('2050')
                    app.live_output_var.set(False)
                    app._live_output_changed()
                    wait(240)
                    self.assertEqual(reader.poll().x_mm,2100)
                    app.publish_pose()
                    self.assertEqual(reader.poll().x_mm,2050)
            finally:
                app.close()


class FieldZoneTests(unittest.TestCase):
    def test_region_and_wrapped_heading_match(self) -> None:
        zone = FieldZone("前方视角", 100, 500, 200, 600, 179, 5)
        self.assertEqual(matching_zones(Pose2D(300, 400, -179), [zone]), [zone])
        self.assertEqual(matching_zones(Pose2D(700, 400, -179), [zone]), [])
        self.assertEqual(matching_zones(Pose2D(300, 400, 160), [zone]), [])

    def test_zone_file_round_trip_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            zone = FieldZone("侧向识别", 100, 500, 200, 600, 270, 30)
            write_zones_atomic(path, [zone])
            self.assertEqual(read_zones(path)[0].heading_deg, -90)
            reader = ZoneFileReader(path)
            self.assertEqual(reader.poll()[0].name, "侧向识别")
            with self.assertRaises(ValueError):
                write_zones_atomic(path, [zone, zone])
            self.assertEqual(len(read_zones(path)), 1)


if __name__ == "__main__":
    unittest.main()
