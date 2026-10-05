from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import pose_association as feedback
from pose_io import Pose2D, write_pose_atomic
from safety_identity import (ActiveTemplate, CornerObservation, CornerIdentity,
                             associate_corners, load_templates, map_corners)
from test_corner_ranging import fixture


def report(prior=None):
    prior = prior or Pose2D(1500, 2100, 90)
    ranger = fixture()
    identity = CornerIdentity(CornerObservation((320., 360.), False, (180., 135.)),
                              {"1.3"}, "1.3", "unique template match")
    readings = ranger.measure([(None, [((320., 360.), 90.)], [])], [identity])
    return feedback.association_payload("testcam", 1, ranger, readings, prior, [identity])


class AssociationTests(unittest.TestCase):
    def test_identity_and_body_range_remain_separate_from_prior(self):
        prior = Pose2D(1500, 2100, 90)
        data = report(prior)
        self.assertEqual(data["confirmed_corner_ids"], ["1.3"])
        self.assertEqual(data["corners"][0]["corner_id"], "1.3")
        self.assertAlmostEqual(data["corners"][0]["distance_mm"], 1500.)
        self.assertEqual(data["simulated_prior"], dict(x_mm=1500, y_mm=2100, heading_deg=90))
        self.assertFalse(data["pose_corrected"])
        self.assertEqual(prior, Pose2D(1500, 2100, 90))
        unknown = fixture().measure([(None, [((320., 360.), 90.)], [])])[0]
        self.assertTrue(unknown.valid)
        self.assertIsNone(unknown.corner_id)

    def test_read_rejects_old_wrong_camera_changed_pose_and_malformed_reports(self):
        prior = Pose2D(1500, 2100, 90)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"latest.json"
            base = report(prior)
            def read(data, camera="testcam", start=0, pose=prior):
                path.write_text(json.dumps(data), encoding="utf-8")
                return feedback.read_live_association(path, camera, start, pose)
            self.assertIsNotNone(read(base))
            self.assertIsNone(read(base, camera="other"))
            self.assertIsNone(read(base, start=base["created_at_unix"]+1))
            self.assertIsNone(read(base, pose=replace(prior, heading_deg=91)))
            self.assertIsNone(read(base | {"created_at_unix": time.time()-5}))
            self.assertIsNone(read(base | {"created_at_unix": float("nan")}))
            self.assertIsNone(read(base | {"simulated_prior": "invalid"}))
            self.assertIsNone(read([]))
            path.write_text("{partial", encoding="utf-8")
            self.assertIsNone(feedback.read_live_association(path, "testcam", 0, prior))
        wrapped = report(Pose2D(1500, 2100, 180))
        self.assertTrue(feedback.report_matches_pose(wrapped, Pose2D(1500, 2100, -180)))

    def test_output_isolated_and_atomic_with_initial_backup(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(feedback, "PROJECT", Path(directory)):
            prior = Path(directory)/"selected_pose.json"
            write_pose_atomic(prior, Pose2D(1500, 2100, 90))
            before = prior.read_bytes()
            output = feedback.association_output_path("testcam", prior_path=prior)
            output.parent.mkdir(parents=True)
            output.write_text('{"old":true}', encoding="utf-8")
            writer = feedback.AssociationWriter(output)
            writer.write(report())
            writer.write(report() | {"frame_number": 2})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["frame_number"], 2)
            backups = list(output.parent.glob("latest.backup_*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(prior.read_bytes(), before)
            with self.assertRaises(ValueError):
                feedback.association_output_path("othercam", output)
            with self.assertRaises(ValueError):
                feedback.association_output_path("testcam", prior)
            with self.assertRaises(ValueError):
                feedback.association_output_path("testcam", output, output)

    def test_calibrated_identity_projection_does_not_use_legacy_fov(self):
        from calibration.lens_models import project_points
        ranger = fixture()
        prior = Pose2D(1500, 1800, 90)
        model = load_templates()
        model["diagonal_pose_projection"]["image_bearing_tolerance_deg"] = .1
        model["diagonal_pose_projection"]["horizontal_fov_deg"] = 20
        template = ActiveTemplate(1, "southwest", "general", (), True, "pose_projection", (1,2,3,4))
        x, y = map_corners(model["safety_zones"][0])[2]
        theta = math.radians(prior.heading_deg)
        dx, dy = x-prior.x_mm, y-prior.y_mm
        body = np.array([[math.cos(theta)*dx+math.sin(theta)*dy,
                          -math.sin(theta)*dx+math.cos(theta)*dy, 0.]])
        rvec, _ = cv2.Rodrigues(ranger.rotation.T)
        pixel = project_points(body, rvec, -ranger.rotation.T@ranger.center,
                               ranger.matrix, ranger.distortion, ranger.model)[0]
        observation = CornerObservation(tuple(pixel), False, ())
        with patch("safety_identity.projected_corner_candidates", side_effect=AssertionError("legacy projection")):
            result = associate_corners([observation], [template], model, prior, 640, ranger)[0]
        self.assertEqual(result.corner_id, "1.3")
        np.testing.assert_allclose([ranger.project(pixel).x_mm, ranger.project(pixel).y_mm], body[0,:2])

    def test_map_feedback_is_read_only_and_clears_on_pose_change(self):
        from pose_map_gui import PoseMapApp, build_parser
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pose.json"
            root = tk.Tk()
            root.withdraw()
            app = PoseMapApp(root, build_parser().parse_args([
                "--output", str(path), "--x", "1500", "--y", "2100", "--heading", "90"]))
            try:
                root.update()
                app.publish_pose()
                before = path.read_bytes()
                prior = app.pose
                with patch.object(app, "publish_pose", side_effect=AssertionError("feedback must not publish")):
                    app.set_live_association(report(prior))
                self.assertEqual(app.pose, prior)
                self.assertEqual(path.read_bytes(), before)
                self.assertIn("1.3", app.live_identity_var.get())
                self.assertTrue(app.canvas.find_withtag("live_identity"))
                app.live_output_var.set(False)
                app._set_pose(replace(prior, heading_deg=80), "test changed prior")
                self.assertFalse(app.canvas.find_withtag("live_identity"))
                self.assertIn("等待", app.live_identity_var.get())
                self.assertEqual(path.read_bytes(), before)
            finally:
                app.close()

    def test_live_pipeline_updates_feedback_for_each_simulated_pose(self):
        import purple_camera_segmentation as vision
        from test_safety_identity import synthetic_frame
        frame = synthetic_frame()
        class Camera:
            def read(self): return True, frame.copy()
            def get(self, _): return 30
            def release(self): pass
        trackbars, records = {}, []
        with tempfile.TemporaryDirectory() as directory, patch.object(feedback, "PROJECT", Path(directory)):
            pose = Path(directory)/"pose.json"
            poses = [Pose2D(1500,2100,90), Pose2D(2200,200,180)]
            write_pose_atomic(pose, poses[0])
            output = feedback.association_output_path("lingyan", prior_path=pose)
            def displayed(window, image):
                records.append(json.loads(output.read_text(encoding="utf-8")))
                if len(records) == 1:
                    write_pose_atomic(pose, poses[1])  # Simulate a user changing the UI input.
            argv = ["vision", "--scene", "auto", "--ranging", "--target-z-mm", "0",
                    "--pose-file", str(pose), "--association-output", str(output)]
            with patch("sys.argv", argv), patch.object(vision, "open_camera", return_value=Camera()), \
                    patch("corner_ranging.CornerRanger.from_camera", return_value=fixture()), \
                    patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                    patch.object(cv2, "createTrackbar", side_effect=lambda n,w,v,*_:trackbars.update({n:v})), \
                    patch.object(cv2, "getTrackbarPos", side_effect=lambda n,w:trackbars[n]), \
                    patch.object(cv2, "setTrackbarPos", side_effect=lambda n,w,v:trackbars.update({n:v})), \
                    patch.object(cv2, "getWindowProperty", return_value=1), \
                    patch.object(cv2, "imshow", side_effect=displayed), \
                    patch.object(cv2, "waitKey", side_effect=[0,ord("q")]), patch.object(cv2, "destroyAllWindows"):
                self.assertEqual(vision.main(), 0)
            self.assertEqual(len(records), 2)
            for data, expected in zip(records, poses):
                self.assertTrue(feedback.report_matches_pose(data, expected))
                self.assertFalse(data["pose_corrected"])
            self.assertEqual([r["scene"] for r in records], ["front-facing", "start-facing"])


if __name__ == "__main__":
    unittest.main()
