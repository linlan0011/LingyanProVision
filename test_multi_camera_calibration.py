from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import camera_profiles as profiles
from calibration.archive_camera import archive_camera
from calibration.fit_mount_pose import main as fit_main
from calibration.ground_project import pixel_to_body_plane
from calibration.lens_models import calibrate, project_points
from calibration.validate_ranging import main as ranging_main
from run_app import build_parser


class MultiCameraCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for target, name, value in (
            (profiles, "PROJECT", self.root),
            (profiles, "REGISTRY", self.root / "camera_profiles.json"),
        ):
            temporary_patch = patch.object(target, name, value)
            temporary_patch.start()
            self.addCleanup(temporary_patch.stop)
        temporary_patch = patch("calibration.archive_camera.PROJECT", self.root)
        temporary_patch.start()
        self.addCleanup(temporary_patch.stop)
        self.profile = profiles.CameraProfile("HD camera", 1280, 1024, "none", default_index=1)
        profiles.register_profile("hd_cam", self.profile)

    def test_registration_and_isolation(self):
        self.assertEqual(profiles.load_profiles()["hd_cam"].image_size, (1280, 1024))
        self.assertEqual(build_parser().parse_args(["--camera", "hd_cam"]).camera, "hd_cam")
        self.assertNotEqual(profiles.data_dir("hd_cam"), profiles.data_dir("lingyan"))
        with self.assertRaises(ValueError):
            profiles.isolated_path(profiles.data_dir("lingyan") / "train", "hd_cam",
                                   result=False, default_name="train")
        with self.assertRaises(ValueError):
            profiles.register_profile("hd_cam", self.profile)
        with self.assertRaises(ValueError):
            profiles.register_profile("../bad", self.profile)
        from calibration.collect_mount_points import image_pixel
        self.assertEqual(image_pixel(320, 256, .5, 1280, 1024), (640, 512))

    def test_capture_configuration_retains_mjpg(self):
        from purple_camera_segmentation import open_camera

        class SimulatedCapture:
            def __init__(self):
                self.format = cv2.VideoWriter_fourcc(*"YUY2")

            def isOpened(self):
                return True

            def set(self, prop, value):
                # Model the measured driver behaviour: mode setters reset the format.
                self.format = value if prop == cv2.CAP_PROP_FOURCC else cv2.VideoWriter_fourcc(*"YUY2")
                return True

            def get(self, prop):
                return self.format if prop == cv2.CAP_PROP_FOURCC else 0

            def release(self):
                pass

        device = SimulatedCapture()
        with patch("purple_camera_segmentation.cv2.VideoCapture", return_value=device):
            cap = open_camera(0, 1920, 1080, fourcc="MJPG", fps=25)
            profile = profiles.CameraProfile("SP", 1920, 1080, "none", pixel_format="MJPG", target_fps=25)
            profiles.verify_capture_format(cap, profile)
            self.assertEqual(device.format, cv2.VideoWriter_fourcc(*"MJPG"))

    def test_capture_retry_requires_a_fresh_frame(self):
        from calibration.capture_chessboard import read_frame
        from unittest.mock import Mock
        frame = np.zeros((10, 10, 3), np.uint8)
        cap = Mock()
        cap.read.side_effect = [(False, None), (False, None), (True, frame)]
        with patch("calibration.capture_chessboard.time.sleep"):
            ok, actual = read_frame(cap)
        self.assertTrue(ok)
        self.assertIs(actual, frame)
        cap.read.side_effect = [(False, None)] * 8
        with patch("calibration.capture_chessboard.time.sleep"):
            self.assertEqual(read_frame(cap), (False, None))

    def test_automatic_control_verifies_mode_not_dynamic_value(self):
        from camera_controls import configure_controls
        from types import SimpleNamespace
        response = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "Controls": [{"Kind": "video", "Name": "white_balance", "Value": 5300, "Flags": 1}]
        }))
        with patch("camera_controls.control_shell", return_value="powershell"), \
                patch("camera_controls.subprocess.run", return_value=response) as run:
            configure_controls(0, "0BDA:3035", automatic={"video.white_balance": 6500})
            self.assertIn('"Flags": 1', run.call_args.args[0][-1])
            self.assertIn("foreach ($setting in $settings)", run.call_args.args[0][-1])
            with self.assertRaisesRegex(RuntimeError, "manual control"):
                configure_controls(0, "0BDA:3035", {"video.white_balance": 6500})
            with self.assertRaisesRegex(ValueError, "both manual and automatic"):
                configure_controls(0, "0BDA:3035", {"video.white_balance": 6500},
                                   {"video.white_balance": 6500})

    def test_camera_controls_prefer_windows_host_over_store_alias(self):
        from camera_controls import control_shell
        with patch.dict("camera_controls.os.environ", {"SystemRoot": r"C:\Windows"}), \
                patch("camera_controls.Path.is_file", return_value=True), \
                patch("camera_controls.shutil.which", return_value=r"C:\WindowsApps\pwsh.exe") as lookup:
            self.assertEqual(control_shell(), r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
            lookup.assert_not_called()

    def test_camera_controls_host_fallback_prefers_windows_powershell(self):
        from camera_controls import control_shell
        with patch("camera_controls.Path.is_file", return_value=False), \
                patch("camera_controls.shutil.which", return_value="powershell") as lookup:
            self.assertEqual(control_shell(), "powershell")
            lookup.assert_called_once_with("powershell")

    def test_intrinsics_reject_changed_automatic_mode(self):
        from dataclasses import replace
        path, _, _ = self.make_intrinsics()
        bound = replace(self.profile, device_vid_pid="0BDA:3035",
                        automatic_controls={"video.white_balance": 6500})
        data = json.loads(path.read_text())
        data["device_vid_pid"] = "0BDA:3035"
        path.write_text(json.dumps(data), encoding="utf-8")
        with patch("camera_profiles.load_profiles", return_value={"hd_cam": bound}):
            with self.assertRaisesRegex(ValueError, "automatic controls"):
                profiles.load_intrinsics("hd_cam")

    def make_intrinsics(self):
        result = profiles.results_dir("hd_cam")
        result.mkdir(parents=True)
        matrix = [[800., 0, 640.], [0, 805., 512.], [0, 0, 1.]]
        intrinsics = result / "intrinsics.json"
        intrinsics.write_text(json.dumps({
            "camera_id": "hd_cam", "model": "opencv_pinhole", "image_width": 1280,
            "image_height": 1024, "capture_width": 1280, "capture_height": 1024,
            "rotation": "none", "camera_matrix": matrix,
            "distortion_coefficients": [0, 0, 0, 0, 0],
        }), encoding="utf-8")
        return intrinsics, np.asarray(matrix), np.zeros(5)

    def test_mount_and_ranging_use_selected_camera(self):
        intrinsics, matrix, distortion = self.make_intrinsics()
        ground = np.array([[x, y, 0] for x in (500, 850, 1200, 1600)
                           for y in (-300, 0, 300)], np.float64)
        rotation_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float64)
        center = np.array([40., 5., 280.])
        rvec, _ = cv2.Rodrigues(rotation_bc.T)
        tvec = -rotation_bc.T @ center
        pixels = project_points(ground, rvec, tvec, matrix, distortion, "opencv_pinhole")
        data = profiles.data_dir("hd_cam")
        data.mkdir(parents=True)
        with (data / "mount_points.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x_mm", "y_mm", "z_mm", "u_px", "v_px"])
            writer.writerows([*point, *pixel] for point, pixel in zip(ground, pixels))
        with patch("sys.argv", ["fit_mount_pose.py", "--camera", "hd_cam"]):
            self.assertEqual(fit_main(), 0)
        self.assertFalse((profiles.results_dir("lingyan") / "mount_pose.json").exists())
        _, mount = profiles.load_mount("hd_cam")
        np.testing.assert_allclose(mount["camera_position_body_mm"], center, atol=1e-3)
        predicted = pixel_to_body_plane(tuple(pixels[4]), 0, matrix, distortion,
                                        np.asarray(mount["rotation_camera_to_body"]),
                                        np.asarray(mount["camera_position_body_mm"]))
        np.testing.assert_allclose(predicted, ground[4], atol=1e-3)
        with (data / "range_validation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x_mm", "y_mm", "z_mm", "u_px", "v_px"])
            writer.writerow([*ground[4], *pixels[4]])
        with patch("sys.argv", ["validate_ranging.py", "--camera", "hd_cam"]):
            self.assertEqual(ranging_main(), 0)
        saved = json.loads((profiles.results_dir("hd_cam") / "range_validation.json").read_text())
        self.assertLess(saved["max_horizontal_error_mm"], 1e-3)
        intrinsics.write_bytes(intrinsics.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "different intrinsics"):
            profiles.load_mount("hd_cam")

    def test_archive_keeps_sources_and_refuses_overwrite(self):
        intrinsics, _, _ = self.make_intrinsics()
        data = profiles.data_dir("hd_cam")
        data.mkdir(parents=True)
        raw = data / "sample.png"
        raw.write_bytes(b"sample image bytes")
        archive = archive_camera("hd_cam", self.root / "saved.zip")
        self.assertTrue(archive.exists())
        self.assertTrue(intrinsics.exists())
        self.assertEqual(raw.read_bytes(), b"sample image bytes")
        with self.assertRaises(FileExistsError):
            archive_camera("hd_cam", archive)

    def test_fisheye_model_is_distinct(self):
        matrix = np.array([[500., 0, 320.], [0, 505., 240.], [0, 0, 1.]])
        distortion = np.array([-0.02, 0.01, 0., 0.])
        obj = np.zeros((54, 3), np.float64)
        obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 19.12
        objects, images = [], []
        for i in range(14):
            rvec = np.array([0.12 * np.sin(i), 0.18 * np.cos(i * 1.7), 0.07 * np.sin(i * .7)])
            tvec = np.array([-80 + 10 * np.sin(i * 1.3), -50 + 8 * np.cos(i), 330 + 10 * i])
            pixels = project_points(obj, rvec, tvec, matrix, distortion, "opencv_fisheye")
            objects.append(obj.astype(np.float32))
            images.append(pixels.reshape(-1, 1, 2).astype(np.float32))
        rms, fit_matrix, fit_distortion, *_ = calibrate(objects, images, (640, 480), "opencv_fisheye")
        self.assertLess(rms, 1e-3)
        self.assertEqual(fit_distortion.size, 4)
        self.assertAlmostEqual(fit_matrix[0, 0], 500, delta=.1)


if __name__ == "__main__":
    unittest.main()
