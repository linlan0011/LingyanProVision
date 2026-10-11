"""Geometry requirements exercised with independently drawn ground truth."""
import unittest

import cv2
import numpy as np

from boundary_geometry import BoundaryConfig, PointGeometry, detect_boundaries


def detect(purple, white=None, config=None):
    if white is None:
        white = cv2.bitwise_not(purple)
    contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return detect_boundaries(contours, white, purple, config)


def rectangle():
    purple = np.zeros((480, 640), np.uint8)
    cv2.rectangle(purple, (100, 100), (500, 400), 255, -1)
    return purple


class BoundaryGeometryTests(unittest.TestCase):
    def assert_corners(self, result, expected, inferred=False, tolerance=3.):
        values = result.inferred if inferred else result.visible
        self.assertEqual(len(values), len(expected), values)
        for point in expected:
            self.assertLessEqual(min(np.linalg.norm(p - point) for p, _ in values), tolerance)

    def test_outer_ring_excludes_inner_white_boundary(self):
        purple = rectangle()
        cv2.rectangle(purple, (130, 130), (470, 370), 0, -1)
        result = detect(purple)
        self.assertEqual(len(result.lines), 4)
        self.assert_corners(result, [(100, 100), (500, 100), (500, 400), (100, 400)])
        self.assertFalse(result.inferred)
        for piece in result.pieces:
            self.assertTrue(np.all((piece.boundary_path[:, 0] <= 101)
                                   | (piece.boundary_path[:, 0] >= 499)
                                   | (piece.boundary_path[:, 1] <= 101)
                                   | (piece.boundary_path[:, 1] >= 399)))

    def test_black_occluder_boundary_never_becomes_green(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        # A 20 px dark strip separates purple from the white field on the right.
        cv2.rectangle(white, (501, 90), (520, 410), 0, -1)
        result = detect(purple, white)
        right = result.boundaries[0]
        middle = (right.points[:, 0] == 500) & (right.points[:, 1] > 120) & (right.points[:, 1] < 380)
        self.assertFalse(right.valid[middle].any())
        self.assertEqual(len(result.lines), 3)
        self.assertFalse(result.inferred)

    def test_multiple_gaps_merge_into_one_bottom_edge(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        for x in (180, 280, 380):
            cv2.rectangle(purple, (x, 370), (x + 25, 420), 0, -1)
            cv2.rectangle(white, (x, 370), (x + 25, 420), 0, -1)
        result = detect(purple, white)
        self.assertEqual(len(result.lines), 4)
        bottom = min(result.lines, key=lambda line: abs(line.fitted_origin[1] - 400))
        self.assertEqual(len(bottom.members), 4)
        self.assert_corners(result, [(100, 100), (500, 100), (500, 400), (100, 400)])
        self.assertFalse(result.inferred)
        for boundary in result.boundaries:
            for x in (180, 280, 380):
                hidden = ((boundary.points[:, 0] > x + 3) & (boundary.points[:, 0] < x + 22)
                          & (boundary.points[:, 1] >= 370))
                self.assertFalse(boundary.valid[hidden].any())

    def test_parallel_edges_and_white_side_remain_separate(self):
        purple = np.zeros((200, 640), np.uint8)
        cv2.rectangle(purple, (80, 80), (560, 90), 255, -1)
        result = detect(purple, config=BoundaryConfig(minimum_edge=5, fit_tolerance_px=6))
        self.assertEqual(len(result.lines), 4)
        long_lines = [line for line in result.lines if line.length > 400]
        self.assertEqual(len(long_lines), 2)
        self.assertLess(np.dot(long_lines[0].white_normal, long_lines[1].white_normal), -.9)

    def test_closed_objects_do_not_share_a_line_or_hidden_corner(self):
        purple = np.zeros((480, 640), np.uint8)
        cv2.rectangle(purple, (80, 100), (200, 250), 255, -1)
        cv2.rectangle(purple, (230, 100), (350, 250), 255, -1)
        result = detect(purple)
        self.assertEqual(len(result.lines), 8)
        self.assertEqual(len(result.visible), 8)
        self.assertFalse(result.inferred)

    def test_one_hidden_corner_and_mirror_use_same_rules(self):
        for mirrored in (False, True):
            purple = rectangle()
            white = cv2.bitwise_not(purple)
            cv2.rectangle(purple, (460, 350), (540, 440), 0, -1)
            cv2.rectangle(white, (460, 350), (540, 440), 0, -1)
            if mirrored:
                purple, white = cv2.flip(purple, 1), cv2.flip(white, 1)
            result = detect(purple, white)
            expected = (139, 400) if mirrored else (500, 400)
            self.assert_corners(result, [expected], inferred=True)
            self.assertEqual(len(result.visible), 3)
            self.assertTrue(all(np.linalg.norm(p - expected) > 12 for p, _ in result.visible))
            self.assertIn(tuple(result.inferred[0][0]), result.corner_supports)

    def test_hidden_corner_with_fragmented_support_edges(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        for a, b in [((460, 350), (540, 440)), ((260, 370), (290, 440)),
                     ((480, 200), (540, 225))]:
            cv2.rectangle(purple, a, b, 0, -1)
            cv2.rectangle(white, a, b, 0, -1)
        result = detect(purple, white)
        self.assertEqual(len(result.lines), 4)
        self.assert_corners(result, [(500, 400)], inferred=True)
        self.assertEqual(len(result.visible), 3)

    def test_split_components_recover_the_same_line(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        cv2.rectangle(purple, (280, 70), (310, 440), 0, -1)
        cv2.rectangle(white, (280, 70), (310, 440), 0, -1)
        result = detect(purple, white)
        self.assertEqual(len(result.lines), 4)
        self.assert_corners(result, [(100, 100), (500, 100), (500, 400), (100, 400)])
        self.assertFalse(result.inferred)

    def test_stepped_parallel_fragments_are_not_one_edge(self):
        purple = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(purple, [np.array([(100, 100), (300, 100), (300, 125),
                                      (500, 125), (500, 400), (100, 400)])], 255)
        result = detect(purple)
        tops = [line for line in result.lines if abs(line.fitted_direction[1]) < .05
                and line.fitted_origin[1] < 150]
        self.assertEqual(len(tops), 2)
        self.assertGreater(abs(tops[0].fitted_origin[1] - tops[1].fitted_origin[1]), 20)

    def test_clipping_is_not_occlusion(self):
        purple = np.zeros((480, 640), np.uint8)
        cv2.rectangle(purple, (-100, 100), (200, 300), 255, -1)
        result = detect(purple)
        self.assert_corners(result, [(200, 100), (200, 300)])
        self.assertFalse(result.inferred)

    def test_calibrated_fit_returns_raw_pixels_and_keeps_green_on_raw_contour(self):
        matrix = np.array([[450., 0, 320.], [0, 450., 240.], [0, 0, 1.]])
        geometry = PointGeometry(matrix, np.array([.2, -.05, 0, 0, 0]), 'opencv_pinhole')
        corners = np.array([(120, 120), (500, 120), (500, 350), (120, 350)], float)
        ideal = np.vstack([np.linspace(corners[i], corners[(i+1) % 4], 150)
                           for i in range(4)])
        observed = geometry.to_pixels(ideal)
        self.assertLess(np.max(np.linalg.norm(geometry.to_geometry(observed)-ideal, axis=1)), .1)
        purple = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(purple, [np.rint(observed).astype(np.int32)], 255)
        result = detect(purple, config=BoundaryConfig(point_geometry=geometry))
        self.assertEqual(len(result.lines), 4)
        self.assert_corners(result, geometry.to_pixels(corners), tolerance=2.)
        for boundary in result.boundaries:
            samples = boundary.raw_points[boundary.valid].astype(int)
            self.assertTrue(np.all(purple[samples[:, 1], samples[:, 0]] > 0))
        for pair in result.corner_supports.values():
            self.assertEqual(len(pair), 2)

    def test_general_component_selection_recovers_vertical_split(self):
        import purple_camera_segmentation as vision
        purple = rectangle()
        cv2.rectangle(purple, (80, 230), (540, 260), 0, -1)
        detections = vision.find_largest_purple(purple, 80)
        ranks = vision.select_safety_component_ranks(detections, vision.GENERAL_RULES, purple.shape)
        self.assertEqual(ranks, {1, 2})

    def test_clean_short_turn_is_not_rejected_by_a_large_error_allowance(self):
        purple = np.zeros((480, 640), np.uint8)
        points = np.array([(100, 300), (100, 100), (400, 100),
                           (432, 115), (464, 144), (500, 300)])
        cv2.fillPoly(purple, [points], 255)
        target = np.array((432, 115))
        for tolerance in (2., 6.75):
            result = detect(purple, config=BoundaryConfig(
                approximation_px=1., fit_tolerance_px=tolerance))
            self.assertTrue(any(np.linalg.norm(p - target) <= 3.
                                for p, _ in result.visible), (tolerance, result.visible))


if __name__ == '__main__':
    unittest.main()
