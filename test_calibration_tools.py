import unittest

import cv2
import numpy as np

from calibration.fit_mount_pose import estimate_mount
from calibration.ground_project import pixel_to_body_plane


class CalibrationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros(5, dtype=np.float64)
        # Camera axes mapped to body: camera Z -> body X, camera X -> body -Y,
        # camera Y -> body -Z (level, forward-looking camera).
        self.rotation_bc = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        self.position_b = np.array([50.0, 0.0, 300.0], dtype=np.float64)

    def test_ground_projection(self) -> None:
        point = pixel_to_body_plane(
            (320.0, 360.0), 0.0, self.camera_matrix, self.distortion,
            self.rotation_bc, self.position_b,
        )
        np.testing.assert_allclose(point, [1550.0, 0.0, 0.0], atol=1e-8)

    def test_mount_fit_recovers_synthetic_pose(self) -> None:
        body_points = np.array([
            [500, -300, 0], [500, 0, 0], [500, 300, 0],
            [900, -350, 0], [900, 100, 0], [900, 350, 0],
            [1300, -400, 0], [1300, 0, 0], [1300, 400, 0],
            [700, -250, 200], [1000, 250, 250], [1400, -200, 300],
        ], dtype=np.float64)
        rotation_cb = self.rotation_bc.T
        translation_cb = -rotation_cb @ self.position_b.reshape(3, 1)
        rvec_cb, _ = cv2.Rodrigues(rotation_cb)
        pixels, _ = cv2.projectPoints(
            body_points, rvec_cb, translation_cb, self.camera_matrix, self.distortion,
        )
        fitted_r, fitted_c, inliers, errors = estimate_mount(
            body_points, pixels.reshape(-1, 2), self.camera_matrix, self.distortion,
        )
        np.testing.assert_allclose(fitted_r, self.rotation_bc, atol=1e-5)
        np.testing.assert_allclose(fitted_c, self.position_b, atol=1e-4)
        self.assertEqual(len(inliers), len(body_points))
        self.assertLess(float(np.max(errors)), 1e-3)

    def test_mount_fit_accepts_ground_plane_only(self) -> None:
        body_points = np.array([
            [500, -400, 0], [500, 0, 0], [500, 400, 0],
            [900, -400, 0], [900, 0, 0], [900, 400, 0],
            [1300, -400, 0], [1300, 0, 0], [1300, 400, 0],
            [1700, -400, 0], [1700, 0, 0], [1700, 400, 0],
        ], dtype=np.float64)
        rotation_cb = self.rotation_bc.T
        translation_cb = -rotation_cb @ self.position_b.reshape(3, 1)
        rvec_cb, _ = cv2.Rodrigues(rotation_cb)
        pixels, _ = cv2.projectPoints(
            body_points, rvec_cb, translation_cb, self.camera_matrix, self.distortion,
        )
        fitted_r, fitted_c, inliers, errors = estimate_mount(
            body_points, pixels.reshape(-1, 2), self.camera_matrix, self.distortion,
        )
        np.testing.assert_allclose(fitted_r, self.rotation_bc, atol=1e-5)
        np.testing.assert_allclose(fitted_c, self.position_b, atol=1e-4)
        self.assertEqual(len(inliers), len(body_points))
        self.assertLess(float(np.max(errors)), 1e-3)


if __name__ == "__main__":
    unittest.main()
