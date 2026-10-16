"""The pose sidebar remains accessible as visual feedback grows."""

from __future__ import annotations

from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest

from pose_map_gui import PoseMapApp, build_parser
from pose_io import Pose2D


class PoseSidebarTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        directory = Path(self.directory.name)
        self.output = directory / "selected_pose.json"
        self.output.write_bytes(b'{"test": "unchanged"}')
        self.original_output = self.output.read_bytes()
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.geometry("1180x880")
        self.app = PoseMapApp(self.root, build_parser().parse_args([
            "--output", str(self.output),
            "--zones", str(directory / "no_zones.json"),
            "--map", str(directory / "no_map.png"),
        ]))
        self.app.live_output_var.set(False)
        self.addCleanup(self.app.close)
        self.root.update_idletasks()

    def test_long_feedback_does_not_increase_window_request_height(self):
        requested_height = self.root.winfo_reqheight()
        original_content_height = self.app.sidebar_content.winfo_reqheight()
        self.app.live_identity_var.set("\n".join([
            "视觉校正：X=1500.0 Y=2100.0 航向=90.0°",
            "惯导输入：X=1490.0 Y=2090.0 航向=89.0°",
        ] * 8))
        self.root.update_idletasks()

        self.assertGreater(self.app.sidebar_content.winfo_reqheight(),
                           original_content_height)
        self.assertEqual(self.root.winfo_reqheight(), requested_height)
        self.assertLessEqual(requested_height, 640)
        bounds = self.app.sidebar_canvas.bbox("all")
        self.assertEqual(bounds[3], self.app.sidebar_content.winfo_reqheight())
        self.app.sidebar_canvas.yview_moveto(1)
        self.assertAlmostEqual(self.app.sidebar_canvas.yview()[1], 1.0)
        self.app.sidebar_canvas.yview_moveto(0)
        self.assertAlmostEqual(self.app.sidebar_canvas.yview()[0], 0.0)
        self.assertEqual(self.output.read_bytes(), self.original_output)

    def test_wheel_is_local_to_sidebar_and_reaches_both_ends(self):
        viewport = self.app.sidebar_canvas
        tag = self.app._sidebar_scroll_tag
        self.assertIn(tag, viewport.bindtags())
        self.assertNotIn(tag, self.app.canvas.bindtags())
        entry = next(widget for widget in self.app.sidebar_content.winfo_children()
                     if widget.winfo_class() == "TEntry")
        self.assertIn(tag, entry.bindtags())
        viewport.yview_moveto(0)
        self.assertEqual(self.app._scroll_sidebar(SimpleNamespace(delta=-120)), "break")
        self.assertGreater(viewport.yview()[0], 0.0)
        self.app._scroll_sidebar(SimpleNamespace(delta=120))
        self.assertAlmostEqual(viewport.yview()[0], 0.0)
        self.app._scroll_sidebar(SimpleNamespace(num=5))
        self.assertGreater(viewport.yview()[0], 0.0)
        self.app._scroll_sidebar(SimpleNamespace(num=4))
        self.assertAlmostEqual(viewport.yview()[0], 0.0)
        for key in ("Prior", "Next", "Home", "End"):
            self.assertTrue(viewport.bind(f"<{key}>"))
        self.assertEqual(self.output.read_bytes(), self.original_output)

    def test_sidebar_content_follows_viewport_width(self):
        self.app._resize_sidebar_content(SimpleNamespace(width=300))
        self.root.update_idletasks()
        self.assertEqual(float(self.app.sidebar_canvas.itemcget(
            self.app._sidebar_window, "width")), 300.0)

    def test_candidate_highlight_uses_frame_view_instead_of_half_field(self):
        prior = Pose2D(1500., 1490., 90.)
        self.app._set_pose(prior, "test input")
        self.assertFalse(self.app.canvas.find_withtag("candidate_safety_zone"))
        self.assertIn("等待", self.app.identity_var.get())
        original_pose_items = self.app.canvas.find_withtag("pose")
        corrected = dict(x_mm=1510., y_mm=1495., heading_deg=91.)
        data = dict(
            simulated_prior=dict(x_mm=prior.x_mm, y_mm=prior.y_mm,
                                 heading_deg=prior.heading_deg),
            confirmed_corner_ids=["1.3"], pose_corrected=True,
            corrected_pose=corrected.copy(),
            visual_pose=dict(
                schema="safety.nearest_visual_pose.v1", status="corrected",
                pose_corrected=True, heading_source="visual_ground_edges",
                corner_id="1.3", corrected_pose=corrected.copy(),
                candidate_view=dict(possible_safety_ids=[1]),
            ),
        )
        self.app.set_live_association(data)
        candidates = self.app.canvas.find_withtag("candidate_safety_zone")
        self.assertTrue(candidates)
        self.assertTrue(all("safety_zone_1" in self.app.canvas.gettags(item)
                            for item in candidates))
        self.assertEqual(self.app.identity_var.get(), "视野候选（非视觉确认）：安全区1")
        self.assertEqual(self.app.canvas.find_withtag("pose"), original_pose_items)
        self.assertTrue(self.app.canvas.find_withtag("corrected_pose"))
        self.assertEqual(self.app.corrected_pose, Pose2D(1510., 1495., 91.))
        self.assertEqual(self.app.pose, prior)
        self.assertEqual(self.output.read_bytes(), self.original_output)

        self.app._set_pose(Pose2D(1500., 1490., 80.), "changed input")
        self.assertFalse(self.app.canvas.find_withtag("candidate_safety_zone"))
        self.assertFalse(self.app.canvas.find_withtag("corrected_pose"))
        self.assertIn("等待", self.app.identity_var.get())
        self.assertEqual(self.output.read_bytes(), self.original_output)


if __name__ == "__main__":
    unittest.main()
