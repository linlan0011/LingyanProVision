from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from corner_ranging import CornerRanger, append_range_panel, range_payload, save_range_image
from run_app import build_parser, vision_command


def fixture(**kwargs):
    intrinsic = {"camera_matrix": [[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]],
                 "distortion_coefficients": [0., 0., 0., 0., 0.],
                 "image_width": 640, "image_height": 480, "model": "opencv_pinhole"}
    mount = {"rotation_camera_to_body": [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
             "camera_position_body_mm": [0., 0., 300.]}
    return CornerRanger(intrinsic, mount, kwargs.pop("target_z_mm", 0.), **kwargs)


class RangingTests(unittest.TestCase):
    def test_metric_coordinates_and_bearing(self):
        r = fixture().project((400, 360))
        self.assertTrue(r.valid)
        np.testing.assert_allclose([r.x_mm, r.y_mm, r.z_mm], [1500., -200., 0.], atol=1e-8)
        self.assertAlmostEqual(r.distance_mm, math.hypot(1500, 200))
        self.assertAlmostEqual(r.bearing_deg, math.degrees(math.atan2(-200, 1500)))

    def test_nonzero_height_and_inferred_opt_in(self):
        ranger = fixture(target_z_mm=100.)
        r = ranger.project((320, 360))
        self.assertAlmostEqual(r.x_mm, 1000.)
        self.assertAlmostEqual(r.z_mm, 100.)
        self.assertEqual(ranger.project((320, 360), inferred=True).reason, "inferred_not_measured")
        estimated = fixture(include_inferred=True).project((320, 360), inferred=True)
        self.assertTrue(estimated.valid)
        self.assertEqual(estimated.reason, "estimated")

    def test_fisheye_pixels_use_fisheye_undistortion(self):
        from calibration.lens_models import project_points
        ranger = fixture()
        ranger.model = "opencv_fisheye"
        ranger.distortion = np.zeros(4)
        expected = np.array([[1500., 200., 0.]])
        rvec, _ = cv2.Rodrigues(ranger.rotation.T)
        tvec = -ranger.rotation.T @ ranger.center
        pixel = project_points(expected, rvec, tvec, ranger.matrix, ranger.distortion, ranger.model)[0]
        result = ranger.project(pixel)
        self.assertTrue(result.valid)
        np.testing.assert_allclose([result.x_mm, result.y_mm, result.z_mm], expected[0], atol=1e-6)

    def test_invalid_rays_and_limits_never_contain_distances(self):
        ranger = fixture()
        cases = [((320, 240), "near_horizon"), ((320, 239), "near_horizon"),
                 ((320, 100), "no_forward_plane_intersection"),
                 ((-1, 360), "pixel_outside_image"), ((640, 360), "pixel_outside_image"),
                 ((float("nan"), 360), "invalid_pixel"), ((320, 320), "beyond_distance_limit")]
        for pixel, reason in cases:
            with self.subTest(pixel=pixel):
                r = ranger.project(pixel)
                self.assertFalse(r.valid)
                self.assertEqual(r.reason, reason)
                self.assertIsNone(r.distance_mm)
        with self.assertRaises(ValueError):
            fixture(target_z_mm=float("nan"))
        with self.assertRaises(ValueError):
            fixture(max_distance_mm=0.)

    def test_calibration_failure_is_not_silently_ignored(self):
        with patch("corner_ranging.load_intrinsics", side_effect=ValueError("camera mismatch")):
            with self.assertRaisesRegex(ValueError, "camera mismatch"):
                CornerRanger.from_camera("bad", target_z_mm=0.)

    def test_deduplication_and_changed_calibration(self):
        ranger = fixture()
        recognitions = [(None, [((320, 360), 90.)], [((320, 360), 90.), ((400, 360), 90.)])]
        readings = ranger.measure(recognitions)
        self.assertEqual(len(readings), 2)
        self.assertTrue(readings[0].valid)
        self.assertFalse(readings[1].valid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"intrinsics.json"
            path.write_text("old", encoding="utf-8")
            from camera_profiles import file_sha256
            ranger.sources = [(path, file_sha256(path)), (path, file_sha256(path))]
            path.write_text("replacement", encoding="utf-8")
            stale = ranger.measure(recognitions)
            self.assertTrue(all(not r.valid and r.reason == "calibration_changed" for r in stale))
            self.assertFalse(range_payload("cam", 1, ranger, stale)["calibration_valid"])

    def test_display_resize_does_not_change_raw_pixel_ranging(self):
        ranger = fixture()
        readings = [ranger.project((400, 360), label="R1")]
        before = asdict(readings[0])
        panel = append_range_panel(np.zeros((1080, 3840, 3), np.uint8), ranger, readings)
        self.assertEqual(panel.shape, (584, 1280, 3))
        self.assertEqual(asdict(readings[0]), before)

    def test_snapshot_image_supports_unicode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"照片"
            folder.mkdir()
            path = folder/"snapshot.raw.png"
            source = np.full((12, 20, 3), 127, np.uint8)
            save_range_image(path, source)
            decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(source, decoded)

    def test_launcher_forwards_explicit_camera_and_height(self):
        args = build_parser().parse_args(["--camera", "usb0bda3035_1080p", "--ranging",
                                        "--target-z-mm", "0", "--range-max-mm", "1000"])
        command = vision_command(args, Path("stop"))
        self.assertIn("--ranging", command)
        self.assertEqual(command[command.index("--camera")+1], "usb0bda3035_1080p")
        self.assertEqual(command[command.index("--target-z-mm")+1], "0.0")
        self.assertEqual(command[command.index("--range-max-mm")+1], "1000.0")
        args.target_z_mm = None
        with self.assertRaises(ValueError):
            vision_command(args, Path("stop"))

    def test_full_app_defaults_to_calibrated_ground_ranging_and_can_disable(self):
        args = build_parser().parse_args([])
        command = vision_command(args,Path("stop"))
        self.assertEqual(args.camera,"usb0bda3035_1080p")
        self.assertIn("--ranging",command)
        self.assertEqual(command[command.index("--target-z-mm")+1],"0.0")
        disabled = vision_command(build_parser().parse_args(["--camera","lingyan","--no-ranging"]),Path("stop"))
        self.assertNotIn("--ranging",disabled)
        self.assertNotIn("--association-output",disabled)

    def test_labels_are_drawn_at_display_scale_without_changing_input_image(self):
        ranger = fixture()
        reading = ranger.project((400,360),label="R1")
        preview = np.zeros((480,1280,3),np.uint8)
        with patch.object(cv2,"putText",wraps=cv2.putText) as draw:
            result = append_range_panel(preview,ranger,[reading])
        calls = [c for c in draw.call_args_list if c.args[1].startswith("R1 r=")]
        self.assertTrue(calls)
        self.assertIn("mm",calls[0].args[1])
        self.assertEqual(calls[0].args[4],.60)
        self.assertFalse(preview.any())
        self.assertTrue(result.any())

    def test_live_pipeline_calls_ranging_without_modifying_detection(self):
        import purple_camera_segmentation as vision
        from pose_io import Pose2D, write_pose_atomic
        from test_safety_identity import synthetic_frame
        frame = synthetic_frame()
        class Camera:
            def read(self): return True, frame.copy()
            def get(self, _): return 30
            def release(self): pass
        ranger = fixture()
        trackbars = {}
        displays = []
        with tempfile.TemporaryDirectory() as directory:
            pose = Path(directory)/"pose.json"
            write_pose_atomic(pose, Pose2D(1500, 2100, 90))
            argv = ["vision", "--scene", "front-facing", "--ranging", "--target-z-mm", "0",
                    "--pose-file", str(pose)]
            with patch("sys.argv", argv), patch.object(vision, "open_camera", return_value=Camera()), \
                    patch.object(vision, "preview_dimensions", return_value=(1440,792)), \
                    patch("corner_ranging.CornerRanger.from_camera", return_value=ranger), \
                    patch.object(ranger, "measure", wraps=ranger.measure) as measure, \
                    patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                    patch.object(cv2, "createTrackbar", side_effect=lambda n,w,v,*_:trackbars.update({n:v})), \
                    patch.object(cv2, "getTrackbarPos", side_effect=lambda n,w:trackbars[n]), \
                    patch.object(cv2, "getWindowProperty", return_value=1), \
                    patch.object(cv2, "imshow", side_effect=lambda w,img:displays.append(img.copy())), \
                    patch.object(cv2, "waitKey", return_value=ord("q")), patch.object(cv2, "destroyAllWindows"):
                self.assertEqual(vision.main(), 0)
        self.assertEqual(measure.call_count, 1)
        self.assertTrue(any(r.valid for r in ranger.measure(measure.call_args.args[0])))
        self.assertEqual(displays[0].shape, (792,1440,3))


if __name__ == "__main__":
    unittest.main()
