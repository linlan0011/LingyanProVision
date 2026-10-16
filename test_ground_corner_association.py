"""Regression tests for the explicitly retained candidate-safety view prefilter."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ground_corner_association import CalibratedGroundView, candidate_safety_view, circle_templates, heading_rotation
from field_zones import read_zones
from pose_io import Pose2D
from safety_identity import load_templates, map_corners
from test_nearest_corner_pose import camera, measured

class GroundViewTests(unittest.TestCase):
    def setUp(self):
        self.ranger=camera()
        self.view=CalibratedGroundView(self.ranger)
        self.model=load_templates()

    def test_cached_view_is_reused_and_candidate_report_contains_all_eight_corners(self):
        cached = self.view.body_polygon.copy()
        with patch.object(self.view, '_body_polygon', side_effect=AssertionError('uncached view')):
            for prior in (Pose2D(1500,2100,90), Pose2D(1500,900,-90), Pose2D(1500,1490,90)):
                for radius in (0.,200.,500.):
                    report = candidate_safety_view(prior,self.model,self.view,radius)
                    expected = cached@heading_rotation(prior.heading_deg).T+[prior.x_mm,prior.y_mm]
                    np.testing.assert_allclose(report['central_footprint_map_mm'],expected)
                    self.assertEqual(len(report['landmarks']),8)
                    self.assertEqual(report['landmarks'],self.view.landmarks(prior,radius,self.model))
        np.testing.assert_array_equal(self.view.body_polygon,cached)

    def test_one_possible_corner_keeps_the_whole_zone_without_half_field_cut(self):
        report = candidate_safety_view(Pose2D(1500,1490,90),self.model,self.view,200.)
        self.assertEqual(report['possible_safety_ids'],[1])
        self.assertEqual(report['status'],'ready')


    def test_half_field_center_does_not_override_the_disk_view(self):
            prior=Pose2D(1500,1490,90)
            landmarks=self.view.landmarks(prior,200,self.model)
            self.assertTrue(any(p['safety_id']==1 and p['possibly_visible'] for p in landmarks))
            self.assertFalse(any(p['safety_id']==2 and p['possibly_visible'] for p in landmarks))
            templates=circle_templates(prior,read_zones('field_zones.json'),self.model,{1},200)
            self.assertTrue(templates)
            self.assertEqual({t.safety_id for t in templates},{1})

    def test_visibility_union_keeps_corner_only_visible_from_disk_edge(self):
            prior=Pose2D(1500,1490,90)
            self.assertFalse(self.view.contains_body((1150,-330)))
            landmarks=self.view.landmarks(prior,200,self.model)
            self.assertTrue(next(p for p in landmarks if p['corner_id']=='1.3')['possibly_visible'])
            far=self.view.landmarks(prior,0,self.model)
            self.assertFalse(next(p for p in far if p['corner_id']=='1.3')['possibly_visible'])

    def test_calibrated_vertical_field_and_mount_are_used(self):
            self.assertFalse(self.view.contains_body((100,0)))
            self.assertTrue(self.view.contains_body((500,0)))
            self.assertFalse(self.view.contains_body((500,900)))
            self.assertTrue(self.view.contains_body((1000,0)))
            self.assertFalse(self.view.contains_body((1001,0)))
            self.assertGreater(len(self.view.body_polygon),3)

    def test_unlimited_view_keeps_valid_ground_points_beyond_one_meter(self):
            self.ranger.max_distance_mm=None
            view=CalibratedGroundView(self.ranger)
            self.assertTrue(view.contains_body((1001.,0.)))
            self.assertTrue(view.contains_body((2000.,0.)))
            self.assertTrue(measured(self.ranger,(2000.,0.)).valid)
            # With zero prior radius, 1.3 was outside the old 1000 mm disk.
            landmarks=view.landmarks(Pose2D(1500,1490,90),0.,self.model)
            self.assertTrue(next(p for p in landmarks if p['corner_id']=='1.3')['possibly_visible'])

    def test_unlimited_landmark_view_dynamically_covers_map_beyond_display_window(self):
            self.ranger.max_distance_mm=None
            view=CalibratedGroundView(self.ranger)
            prior=Pose2D(1500.,-30000.,90.)
            rotation=heading_rotation(prior.heading_deg)
            center=np.array([prior.x_mm,prior.y_mm])
            # Every corner is forward and projects inside the calibrated image,
            # even though all are beyond the default finite display footprint.
            corners=[np.asarray(point)for zone in self.model['safety_zones']
                     for point in map_corners(zone)]
            self.assertTrue(all(np.linalg.norm(point-center)>10000. for point in corners))
            self.assertTrue(all(view.contains_body(rotation.T@(point-center))for point in corners))
            self.assertGreater(view.map_extent(prior,0.,self.model),30000.)
            landmarks=view.landmarks(prior,0.,self.model)
            self.assertEqual(len(landmarks),8)
            self.assertTrue(all(p['possibly_visible']for p in landmarks))

    def test_distorted_border_footprint_contains_valid_ground_rays(self):
            self.ranger.distortion=np.array([.15,-.2,.002,-.002,.08])
            view=CalibratedGroundView(self.ranger)
            contour=view.body_polygon.astype(np.float32).reshape(-1,1,2)
            for u in np.linspace(0,639,29):
                for v in np.linspace(241,479,17):
                    reading=self.ranger.project((u,v))
                    if reading.valid:
                        self.assertGreaterEqual(cv2.pointPolygonTest(contour,(reading.x_mm,reading.y_mm),True),-.01)
