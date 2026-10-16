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

    def test_thin_white_halo_does_not_turn_occluder_into_green(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        cv2.rectangle(purple, (460, 350), (540, 440), 0, -1)
        cv2.rectangle(white, (460, 350), (540, 440), 0, -1)
        # Only a 3 px pale band at the contact; the object behind it is coloured.
        cv2.rectangle(white, (460, 350), (462, 400), 255, -1)
        cv2.rectangle(white, (460, 350), (500, 352), 255, -1)
        result = detect(purple, white)
        self.assertEqual(len(result.visible), 3)
        self.assert_corners(result, [(500, 400)], inferred=True)
        for boundary in result.boundaries:
            points = boundary.raw_points
            contact = (((points[:, 0] == 459) & (points[:, 1] > 360))
                       | ((points[:, 1] == 349) & (points[:, 0] > 470)))
            self.assertFalse(boundary.valid[contact].any())

    def test_single_white_ray_is_not_an_adjacent_white_region(self):
        purple = rectangle()
        white = np.zeros_like(purple)
        white[250, 501:520] = 255
        result = detect(purple, white)
        self.assertFalse(any(b.valid.any() for b in result.boundaries))

    def test_white_beyond_coloured_gap_is_not_evidence(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        white[90:411, 501:504] = 0  # White resumes inside the wider search band.
        result = detect(purple, white)
        boundary = result.boundaries[0]
        right = (boundary.raw_points[:, 0] == 500) & (boundary.raw_points[:, 1] > 120) & (boundary.raw_points[:, 1] < 380)
        self.assertFalse(boundary.valid[right].any())

    def test_near_white_cannot_bridge_two_pixel_coloured_gap(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        white[90:411, 503:505] = 0  # 1/2 px white, 3/4 px coloured, then white.
        result = detect(purple, white)
        boundary = result.boundaries[0]
        middle = (boundary.raw_points[:, 0] == 500) & (boundary.raw_points[:, 1] > 120) & (boundary.raw_points[:, 1] < 380)
        self.assertFalse(boundary.valid[middle].any())

    def test_isolated_sample_noise_does_not_erase_a_white_strip(self):
        purple = rectangle()
        white = cv2.bitwise_not(purple)
        white[90:411, 503] = 0  # One isolated failed sample, surrounded by white.
        result = detect(purple, white)
        boundary = result.boundaries[0]
        middle = (boundary.raw_points[:, 0] == 500) & (boundary.raw_points[:, 1] > 120) & (boundary.raw_points[:, 1] < 380)
        self.assertTrue(boundary.valid[middle].all())

    def test_pale_blue_face_requires_neutral_core_and_recovers_hidden_corner(self):
        import purple_camera_segmentation as vision
        for mirrored in (False, True):
            for scene in ('start-facing', 'front-facing', 'general'):
                frame = np.full((480, 640, 3), 235, np.uint8)
                cv2.rectangle(frame, (100, 100), (500, 400), (140, 30, 140), -1)
                blue = cv2.cvtColor(np.uint8([[[100, 75, 218]]]), cv2.COLOR_HSV2BGR)[0, 0]
                cv2.rectangle(frame, (460, 350), (540, 440), tuple(map(int, blue)), -1)
                if mirrored:
                    frame = cv2.flip(frame, 1)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                rules = vision.USB_0BDA_3035_SCENE_RULES[scene]
                purple = cv2.inRange(hsv, (120, 55, 25), (150, 255, 255))
                white = vision.make_white_field_mask(hsv, rules.white_s_max, 100)
                core = vision.make_white_core_mask(hsv, rules.white_s_max, 100)
                self.assertGreater(int(white[380, 159 if mirrored else 480]), 0)
                self.assertEqual(int(core[380, 159 if mirrored else 480]), 0)
                contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                visible, inferred = vision.find_effective_boundary_corners(
                    contours, white, purple, rules, white_core_mask=core)
                self.assertEqual(len(visible), 3, (scene, mirrored, visible))
                self.assertEqual(len(inferred), 1)
                target = np.array((139, 400) if mirrored else (500, 400))
                self.assertLessEqual(np.linalg.norm(inferred[0][0] - target), 3.)

    def test_dim_cool_white_floor_keeps_all_visible_corners(self):
        import purple_camera_segmentation as vision
        colour = cv2.cvtColor(np.uint8([[[100, 35, 90]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame = np.full((480, 640, 3), colour, np.uint8)
        cv2.rectangle(frame, (100, 100), (500, 400), (140, 30, 140), -1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        purple = cv2.inRange(hsv, (120, 55, 25), (150, 255, 255))
        white = vision.make_white_field_mask(hsv, 80, 100)
        core = vision.make_white_core_mask(hsv, 80, 100)
        contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = detect_boundaries(contours, white, purple, white_core_mask=core)
        self.assert_corners(result, [(100, 100), (500, 100), (500, 400), (100, 400)])


class SafetyComponentSelectionTests(unittest.TestCase):
    def frame(self):
        purple = np.zeros((480, 640), np.uint8)
        cv2.rectangle(purple, (80, 300), (560, 440), 255, -1)
        cv2.rectangle(purple, (100, 200), (500, 212), 255, -1)
        return purple

    def select(self, purple, scene='front-facing', **kwargs):
        import purple_camera_segmentation as vision
        detections = vision.find_largest_purple(purple, 80)
        ranks = vision.select_safety_component_ranks(
            detections, vision.SCENE_RULES[scene], purple.shape, **kwargs)
        return detections, ranks

    def test_small_far_side_survives_perspective_mirror_and_rotation(self):
        purple = self.frame()
        for frame in (purple, cv2.flip(purple, 1),
                      cv2.rotate(purple, cv2.ROTATE_90_CLOCKWISE)):
            for scene in ('start-facing', 'front-facing', 'general'):
                detections, ranks = self.select(frame, scene)
                self.assertLess(detections[1][0], .25 * detections[0][0])
                self.assertEqual(ranks, {1, 2}, scene)

    def test_below_floor_noise_and_distant_fragments_are_excluded(self):
        purple = self.frame()
        cv2.rectangle(purple, (100, 250), (112, 262), 255, -1)  # Near, area 144.
        cv2.rectangle(purple, (100, 10), (500, 22), 255, -1)  # Large but distant.
        detections, ranks = self.select(purple)
        selected_boxes = {row[1:5] for index, row in enumerate(detections, 1) if index in ranks}
        self.assertEqual(selected_boxes, {(80, 300, 481, 141), (100, 200, 401, 13)})

    def test_small_fragments_can_connect_through_an_intermediate_fragment(self):
        purple = self.frame()
        purple[200:213, 100:501] = 0
        cv2.rectangle(purple, (100, 40), (500, 52), 255, -1)
        self.assertEqual(self.select(purple)[1], {1})
        cv2.rectangle(purple, (100, 170), (500, 182), 255, -1)
        self.assertEqual(self.select(purple)[1], {1, 2, 3})

    def test_current_parameter_area_floor_is_used(self):
        purple = self.frame()
        self.assertEqual(self.select(purple, minimum_area=400)[1], {1, 2})
        self.assertEqual(self.select(purple, minimum_area=8000)[1], {1})

    def test_side_mode_accepts_nearby_split_components(self):
        self.assertEqual(self.select(self.frame(), 'start-facing')[1], {1, 2})

    def test_selected_fragment_without_white_support_produces_no_corner(self):
        import purple_camera_segmentation as vision
        purple = self.frame()
        white = cv2.bitwise_not(purple)
        cv2.rectangle(white, (95, 195), (505, 217), 0, -1)
        detections, ranks = self.select(purple)
        self.assertEqual(ranks, {1, 2})
        contours = [row[-1] for index, row in enumerate(detections, 1) if index in ranks]
        state = {}
        visible, inferred = vision.find_effective_boundary_corners(
            contours, white, purple, vision.FRONT_RULES,
            white_core_mask=white, boundary_result=state)
        self.assertEqual(len(visible), 4)
        self.assertFalse(inferred)
        self.assertFalse(state['result'].boundaries[1].valid.any())


if __name__ == '__main__':
    unittest.main()
