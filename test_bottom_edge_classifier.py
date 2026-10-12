from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottom_edge_classifier import (BottomEdgeConfig, classify_bottom_edges,
                                    direct_corner_connections)
from corner_ranging import CornerRange


def reading(pixel, xy, *, inferred=False, valid=True, z=0.):
    x, y = xy
    return CornerRange('', tuple(pixel), inferred, valid,
                       '' if valid else 'near_horizon', x, y, z,
                       math.hypot(x, y), math.degrees(math.atan2(y, x)))


def graph(*chains):
    supports, lines = {}, []
    for line_id, points in enumerate(chains, 1):
        start, end = np.asarray(points[0], float), np.asarray(points[-1], float)
        direction = end - start
        direction /= np.linalg.norm(direction)
        lines.append(SimpleNamespace(line_id=line_id, fitted_origin=start,
                                     fitted_direction=direction))
        for point in points:
            supports.setdefault(tuple(point), []).append(SimpleNamespace(line_id=line_id))
    return SimpleNamespace(config=SimpleNamespace(point_geometry=None),
                           lines=lines, corner_supports=supports)


class BottomEdgeTests(unittest.TestCase):
    far, upper, seed = (100, 100), (200, 100), (100, 300)

    def basic(self):
        return [reading(self.far, (1200, 0), inferred=True),
                reading(self.upper, (1100, 100)), reading(self.seed, (300, 0))]

    def test_upper_neighbors_then_nearest_seed_and_both_bottom_dimensions(self):
        points = self.basic() + [reading((200, 300), (960, 0)),
                                 reading((300, 300), (300, 360)),
                                 reading((400, 300), (960, 360))]
        before = [asdict(r) for r in points]
        result = classify_bottom_edges(points, graph((self.far, self.upper)))
        self.assertEqual(result.status, 'classified')
        self.assertEqual(result.upper_seed, self.far)
        self.assertEqual(result.lower_seed, self.seed)
        self.assertEqual(result.upper_count, 2)
        self.assertEqual(result.lower_count, 3)
        self.assertEqual(result.by_pixel[(200, 300)].matched_edge, 'length')
        self.assertEqual(result.by_pixel[(300, 300)].matched_edge, 'width')
        # The opposite rectangle corner is a diagonal, not a length/width match.
        self.assertEqual(result.by_pixel[(400, 300)].layer, 'unknown')
        self.assertAlmostEqual(result.by_pixel[(400, 300)].distance_to_seed_mm,
                               math.hypot(660, 360))
        self.assertEqual([asdict(r) for r in points], before)
        self.assertEqual(len(result.corners), len(points))
        self.assertTrue(result.by_pixel[self.far].inferred)
        json.dumps(result.payload(), allow_nan=False)

    def test_two_candidates_use_nearest_without_upper_filter(self):
        result = classify_bottom_edges(self.basic()[:2], None)
        self.assertEqual(result.status, 'nearest_fallback')
        self.assertEqual(result.lower_seed, self.upper)
        self.assertEqual(result.upper_count, 0)
        self.assertEqual(result.lower_count, 1)
        self.assertEqual(result.by_pixel[self.far].layer, 'unknown')

    def test_one_inferred_candidate_is_the_fallback_seed(self):
        result = classify_bottom_edges(self.basic()[:1], None)
        self.assertEqual(result.status, 'nearest_fallback')
        self.assertEqual(result.lower_seed, self.far)
        self.assertTrue(result.by_pixel[self.far].inferred)

    def test_no_candidate_produces_no_reference(self):
        result = classify_bottom_edges([], None)
        self.assertEqual(result.status, 'no_candidates')
        self.assertIsNone(result.lower_seed)

    def test_missing_one_range_does_not_rank_a_partial_set(self):
        points = self.basic() + [reading((300, 200), (2000, 0), valid=False)]
        result = classify_bottom_edges(points, None)
        self.assertEqual(result.status, 'unmeasurable_or_wrong_plane')
        self.assertIsNone(result.upper_seed)
        self.assertEqual(result.by_pixel[(300, 200)].reason, 'near_horizon')

    def test_mixed_height_ranges_are_not_compared(self):
        points = self.basic()
        points[-1] = reading(self.seed, (300, 0), z=20.)
        self.assertEqual(classify_bottom_edges(points, None).status,
                         'unmeasurable_or_wrong_plane')

    def test_upper_propagation_is_only_one_hop_and_only_adjacent(self):
        third = (300, 100)
        points = self.basic() + [reading(third, (900, 0))]
        result = classify_bottom_edges(points, graph((self.far, self.upper, third)))
        self.assertEqual(result.by_pixel[self.upper].layer, 'upper')
        self.assertNotEqual(result.by_pixel[third].layer, 'upper')
        self.assertNotIn((self.far, third, 1), result.connections)

    def test_nearby_points_on_different_fitted_lines_are_not_connected(self):
        other = (101, 100)
        result = classify_bottom_edges(self.basic() + [reading(other, (1000, 0))],
                                       graph((self.far, self.upper), (other, self.seed)))
        self.assertNotEqual(result.by_pixel[other].layer, 'upper')

    def test_all_upper_neighbors_can_leave_no_bottom_candidate(self):
        points = self.basic()
        result = classify_bottom_edges(points, graph((self.far, self.upper),
                                                      (self.far, self.seed)))
        self.assertEqual(result.status, 'no_remaining_candidates')
        self.assertEqual(result.upper_count, 3)
        self.assertIsNone(result.lower_seed)

    def test_length_tolerance_is_inclusive_and_checks_error_not_just_upper_bound(self):
        for target, label in ((660, 'length'), (360, 'width')):
            for error in (-31, -30, 0, 30, 31):
                with self.subTest(target=target, error=error):
                    point = (400, 300)
                    result = classify_bottom_edges(
                        self.basic() + [reading(point, (300 + target + error, 0))],
                        graph((self.far, self.upper)))
                    corner = result.by_pixel[point]
                    self.assertEqual(corner.layer == 'lower', abs(error) <= 30)
                    if corner.layer == 'lower':
                        self.assertEqual(corner.matched_edge, label)

    def test_upper_points_are_never_reaccepted_by_the_dimension_check(self):
        points = [reading(self.far, (1500, 0)),
                  reading(self.upper, (960, 0)), reading(self.seed, (300, 0))]
        result = classify_bottom_edges(points, graph((self.far, self.upper)))
        self.assertEqual(result.by_pixel[self.upper].layer, 'upper')
        self.assertIsNone(result.by_pixel[self.upper].matched_edge)

    def test_body_rotation_and_mirror_preserve_distance_classification(self):
        original = self.basic() + [reading((400, 300), (300, 360))]
        for degrees in (0, 43, 90):
            angle = math.radians(degrees)
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            for mirror in (1, -1):
                points = [reading(r.pixel, rotation @ [r.x_mm, mirror*r.y_mm],
                                  inferred=r.inferred) for r in original]
                result = classify_bottom_edges(points, graph((self.far, self.upper)))
                self.assertEqual(result.upper_seed, self.far)
                self.assertEqual(result.lower_seed, self.seed)
                self.assertEqual(result.by_pixel[(400, 300)].layer, 'lower')

    def test_duplicate_visible_and_inferred_pixel_is_one_candidate(self):
        points = self.basic()[:2] + [reading(self.far, (1200, 0), inferred=False)]
        result = classify_bottom_edges(points, None)
        self.assertEqual(result.candidate_count, 2)
        self.assertFalse(result.by_pixel[self.far].inferred)

    def test_fallback_does_not_hide_a_failed_range(self):
        result = classify_bottom_edges([self.basic()[0],
                                         reading(self.seed,(300,0),valid=False)], None)
        self.assertEqual(result.status, 'unmeasurable_or_wrong_plane')
        self.assertIsNone(result.lower_seed)

    def test_invalid_configuration_rejected(self):
        for kwargs in ({'length_mm':0}, {'width_mm':float('nan')}, {'tolerance_mm':-1}):
            with self.assertRaises(ValueError):
                BottomEdgeConfig(**kwargs)

    def test_calibrated_support_ids_connect_the_same_corner_pair(self):
        from boundary_geometry import BoundaryConfig, PointGeometry, detect_boundaries
        purple = np.zeros((480, 640), np.uint8)
        cv2.rectangle(purple, (100, 100), (500, 400), 255, -1)
        contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        geometry = PointGeometry(np.array([[450.,0,320.],[0,450.,240.],[0,0,1.]]),
                                 np.zeros(5), 'opencv_pinhole')
        result = detect_boundaries(contours, cv2.bitwise_not(purple), purple,
                                   BoundaryConfig(point_geometry=geometry))
        pixels = [tuple(point) for point, _ in result.visible]
        links = direct_corner_connections(pixels, result)
        self.assertEqual(len(links), 4)
        self.assertTrue(all(line_id > 0 for _, _, line_id in links))
        self.assertFalse(any({a,b} == {(100,100),(500,400)} for a,b,_ in links))

    def test_projected_upper_and_lower_geometry_with_actual_height_offset(self):
        from calibration.lens_models import project_points
        from corner_ranging import CornerRanger
        intrinsic = {'camera_matrix':[[400.,0,320.],[0,400.,240.],[0,0,1.]],
                     'distortion_coefficients':[0.,0.,0.,0.,0.],
                     'image_width':640,'image_height':480,'model':'opencv_pinhole'}
        mount = {'rotation_camera_to_body':[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],
                 'camera_position_body_mm':[0.,0.,174.]}
        ranger = CornerRanger(intrinsic,mount,0.,include_inferred=True)
        # Bottom: X 600..960, Y +/-330. Top is inset 30 and raised 20.
        body = np.array([[930.,300.,20.],[930.,-300.,20.],
                         [600.,330.,0.],[600.,-330.,0.]])
        rvec,_=cv2.Rodrigues(ranger.rotation.T)
        pixels=project_points(body,rvec,-ranger.rotation.T@ranger.center,
                              ranger.matrix,ranger.distortion,ranger.model)
        ranges=[ranger.project(pixel,inferred=(i==0)) for i,pixel in enumerate(pixels)]
        keys=[tuple(map(int,np.rint(p)))for p in pixels]
        result=classify_bottom_edges(ranges,graph(keys[:2],keys[2:]))
        self.assertEqual(result.upper_count,2)
        self.assertEqual(result.lower_count,2)
        self.assertTrue(all(result.by_pixel[p].layer=='upper' for p in keys[:2]))
        self.assertTrue(all(result.by_pixel[p].layer in ('lower_seed','lower') for p in keys[2:]))
        matched=next(c for c in result.corners if c.layer=='lower')
        self.assertAlmostEqual(matched.distance_to_seed_mm,660.)


class BottomEdgePipelineTests(unittest.TestCase):
    def test_inferred_ranges_and_all_corners_survive_in_the_live_report(self):
        import purple_camera_segmentation as vision
        from test_corner_ranging import fixture
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,340),(500,440),(140,30,140),-1)
        class Camera:
            def read(self): return True,frame.copy()
            def get(self,_): return 30
            def release(self): pass
        ranger = fixture()
        original_detector = vision.find_effective_boundary_corners
        observed_counts = []
        def detector(*args, **kwargs):
            visible, inferred = original_detector(*args, **kwargs)
            # Exercise a hidden corner at the integration boundary, with its
            # already validated support pair retained in the boundary result.
            inferred = list(inferred) + visible[:1]
            visible = visible[1:]
            observed_counts.append(len(visible)+len(inferred))
            return visible,inferred
        trackbars={}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'output/association/lingyan/latest.json'
            with patch('sys.argv',['vision','--parameters','--scene','front-facing',
                    '--ranging','--target-z-mm','0','--association-output',str(output),
                    '--pose-file',str(Path(directory)/'no_pose.json')]), \
                 patch.object(vision,'open_camera',return_value=Camera()), \
                 patch.object(vision,'find_effective_boundary_corners',side_effect=detector), \
                 patch('pose_association.PROJECT',Path(directory)), \
                 patch('corner_ranging.CornerRanger.from_camera',return_value=ranger), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda n,w,v,*_:trackbars.update({n:v})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda n,w:trackbars[n]), \
                 patch.object(cv2,'setTrackbarPos',side_effect=lambda n,w,v:trackbars.update({n:v})), \
                 patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'imshow'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
            data=json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(observed_counts,[4])
        self.assertEqual(len(data['corners']),4)
        self.assertTrue(any(c['inferred'] and c['valid'] for c in data['corners']))
        self.assertEqual(data['bottom_edge_test']['candidate_count'],4)
        self.assertFalse(data['pose_corrected'])
        self.assertFalse(ranger.include_inferred)


if __name__ == '__main__':
    unittest.main()
