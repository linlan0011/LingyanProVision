from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math
import unittest

import cv2
import numpy as np

from calibration.lens_models import project_points
from corner_ranging import CornerRanger
from ground_corner_association import (CalibratedGroundView, candidate_safety_view, heading_rotation)
from nearest_corner_pose import VisualPoseConfig, solve_nearest_corner_pose, apply_visual_identities, pixel_key
from pose_io import Pose2D
from safety_identity import load_templates, map_corners


def camera():
    intrinsics = dict(camera_matrix=[[300.,0,320.],[0,300.,240.],[0,0,1.]],
        distortion_coefficients=[0.]*5, image_width=640,image_height=480,model='opencv_pinhole')
    mount = dict(rotation_camera_to_body=[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],
                 camera_position_body_mm=[0.,0.,174.])
    return CornerRanger(intrinsics,mount,0.,max_distance_mm=1000.,include_inferred=True)


def project_body(ranger, points):
    points = np.asarray(points, float).reshape(-1,2)
    rvec,_ = cv2.Rodrigues(ranger.rotation.T)
    return project_points(np.c_[points,np.zeros(len(points))],rvec,-ranger.rotation.T@ranger.center,
                          ranger.matrix,ranger.distortion,ranger.model)


def measured(ranger, xy, inferred=False, *, geometry=False):
    pixel = project_body(ranger, [xy])[0]
    return ranger.project(pixel,inferred=inferred,enforce_distance_limit=not geometry)


def scene(ranger, zone_id=1, corner_index=2, heading=90., body=(500.,0.)):
    model = load_templates()
    corners = np.asarray(map_corners(model['safety_zones'][zone_id-1]))
    corner = corners[corner_index]
    rotation = heading_rotation(heading)
    position = corner-rotation@body
    prior = Pose2D(*position,heading)
    reading = measured(ranger,body)
    supports = []
    for line_id, neighbor in enumerate((corners[(corner_index-1)%4],corners[(corner_index+1)%4]),1):
        unit = rotation.T@(neighbor-corner)/np.linalg.norm(neighbor-corner)
        pixels = project_body(ranger,np.asarray(body)+np.linspace(10.,120.,17)[:,None]*unit)
        supports.append(SimpleNamespace(line_id=line_id,boundary_path=pixels,
                                       fitted_start=pixels[0],fitted_end=pixels[-1]))
    return prior, reading, {pixel_key(reading.pixel):tuple(supports)}


class NearestVisualPoseTests(unittest.TestCase):
    def setUp(self):
        self.ranger = camera()
        self.model = load_templates()
        self.view = CalibratedGroundView(self.ranger)

    def solve(self, prior, readings, supports, *, geometry=None, radius=200., candidate=None,
              boundary_result=None, **config):
        geometry = geometry if geometry is not None else [replace(r,geometry_only=True) for r in readings]
        candidate = candidate if candidate is not None else candidate_safety_view(prior,self.model,self.view,radius)
        return solve_nearest_corner_pose(readings,geometry,supports,prior,self.model,self.ranger,candidate,
                                        boundary_result=boundary_result,config=VisualPoseConfig(**config))

    def test_all_eight_ids_and_all_absolute_pose_components(self):
        for zone_id in (1,2):
            for index in range(4):
                heading = 0. if index in (0,3) else 180.
                prior,reading,supports = scene(self.ranger,zone_id,index,heading)
                with self.subTest(zone=zone_id,index=index):
                    identities,report = self.solve(prior,[reading],supports)
                    self.assertEqual(report['status'],'corrected')
                    self.assertEqual(identities[0].corner_id,f'{zone_id}.{index+1}')
                    pose = report['corrected_pose']
                    np.testing.assert_allclose([pose['x_mm'],pose['y_mm']],[prior.x_mm,prior.y_mm],atol=1e-6)
                    self.assertAlmostEqual((pose['heading_deg']-heading+180)%360-180,0.)

    def test_gyro_perturbation_selects_identity_but_does_not_change_visual_pose(self):
        actual,reading,supports = scene(self.ranger,heading=90.)
        results = []
        for gyro in (80.,86.,90.,96.,100.):
            _,report = self.solve(replace(actual,heading_deg=gyro),[reading],supports)
            self.assertEqual(report['status'],'corrected')
            results.append(list(report['corrected_pose'].values()))
        np.testing.assert_allclose(results,np.tile([actual.x_mm,actual.y_mm,90.],(5,1)),atol=1e-6)

    def test_position_input_is_only_a_zone_view_prefilter_not_a_corner_error_circle(self):
        _,reading,supports = scene(self.ranger,heading=90.)
        prior = Pose2D(1500.,1800.,90.)
        _,report = self.solve(prior,[reading],supports)
        self.assertEqual(report['status'],'corrected')
        corrected = report['corrected_pose']
        self.assertGreater(math.hypot(corrected['x_mm']-prior.x_mm,corrected['y_mm']-prior.y_mm),200.)

    def test_nearest_is_selected_before_any_edge_or_identity_filter(self):
        prior,nearest,supports = scene(self.ranger)
        farther = measured(self.ranger,(650.,100.))
        farthest = measured(self.ranger,(800.,0.))
        identities,report = self.solve(prior,[farthest,nearest,farther],supports)
        self.assertEqual(report['selected_corner']['pixel'],list(pixel_key(nearest.pixel)))
        self.assertEqual([r.corner_id for r in identities],[None,'1.3',None])

    def test_bad_nearest_does_not_fall_back_to_farther_point(self):
        prior,farther,supports = scene(self.ranger,body=(600.,0.))
        nearest = measured(self.ranger,(400.,0.))
        identities,report = self.solve(prior,[farther,nearest],supports)
        self.assertEqual(report['selected_corner']['pixel'],list(pixel_key(nearest.pixel)))
        self.assertEqual(report['status'],'two_support_edges_required')
        self.assertTrue(all(i.corner_id is None for i in identities))
        self.assertIsNone(report['corrected_pose'])

    def test_geometry_ranking_precedes_one_metre_limit(self):
        prior,nearest,supports = scene(self.ranger)
        geometry_far = measured(self.ranger,(1400.,0.),True,geometry=True)
        far = self.ranger.project(geometry_far.pixel,inferred=True)
        self.assertFalse(far.valid)
        _,report = self.solve(prior,[far,nearest],supports,
                              geometry=[geometry_far,replace(nearest,geometry_only=True)])
        self.assertEqual(report['status'],'corrected')
        _,report = self.solve(prior,[far],{},geometry=[geometry_far])
        self.assertEqual(report['status'],'nearest_out_of_range')

    def test_inferred_point_is_supported_and_inputs_are_not_mutated(self):
        prior,reading,supports = scene(self.ranger)
        reading = replace(reading,inferred=True)
        before = deepcopy(reading)
        identities,report = self.solve(prior,[reading],supports)
        self.assertEqual(report['status'],'corrected')
        self.assertTrue(report['selected_corner']['inferred'])
        self.assertEqual(reading,before)
        self.assertIsNone(reading.corner_id)
        tagged = apply_visual_identities([reading],identities)
        self.assertEqual(tagged[0].corner_id,'1.3')

    def test_two_candidate_zones_remain_ambiguous(self):
        prior,reading,supports = scene(self.ranger)
        candidate = candidate_safety_view(prior,self.model,self.view,3000.)
        self.assertEqual(candidate['possible_safety_ids'],[1,2])
        identities,report = self.solve(prior,[reading],supports,candidate=candidate)
        self.assertEqual(report['status'],'ambiguous_safety_zone')
        self.assertEqual(identities[0].candidates,{'1.3','2.3'})
        self.assertFalse(report['pose_corrected'])
        self.assertIsNone(report['corrected_pose'])

    def test_heading_wrap_and_support_endpoint_order_do_not_flip_rays(self):
        actual,reading,supports = scene(self.ranger,corner_index=1,heading=179.)
        for support in supports[pixel_key(reading.pixel)]:
            support.fitted_start,support.fitted_end = support.fitted_end,support.fitted_start
            support.boundary_path = support.boundary_path[::-1]
        _,report = self.solve(replace(actual,heading_deg=-179.),[reading],supports)
        self.assertEqual(report['status'],'corrected')
        self.assertAlmostEqual(report['corrected_pose']['heading_deg'],179.)

    def test_outside_identity_tolerance_never_uses_gyro_as_pose_fallback(self):
        prior,reading,supports = scene(self.ranger)
        _,report = self.solve(replace(prior,heading_deg=115.),[reading],supports)
        self.assertEqual(report['status'],'identity_no_match')
        self.assertIsNone(report['corrected_pose'])

    def test_short_edge_and_nonorthogonal_pair_reject_update(self):
        prior,reading,supports = scene(self.ranger)
        pair = supports[pixel_key(reading.pixel)]
        pair[0].boundary_path = project_body(self.ranger,[[500.,v] for v in np.linspace(1.,5.,5)])
        _,report = self.solve(prior,[reading],supports)
        self.assertEqual(report['status'],'edge_too_short')
        prior,reading,supports = scene(self.ranger)
        pair = supports[pixel_key(reading.pixel)]
        pair[0].boundary_path = project_body(self.ranger,[[500.+v,v] for v in np.linspace(10.,120.,17)])
        _,report = self.solve(prior,[reading],supports)
        self.assertEqual(report['status'],'edges_not_orthogonal')

    def test_two_visual_headings_must_agree_before_pose_update(self):
        prior,reading,supports = scene(self.ranger)
        pair = supports[pixel_key(reading.pixel)]
        unit = np.array([math.cos(math.radians(10)),math.sin(math.radians(10))])
        pair[0].boundary_path = project_body(self.ranger,np.array([500.,0.])+np.linspace(10.,120.,17)[:,None]*unit)
        _,report = self.solve(prior,[reading],supports,heading_consistency_deg=5.)
        self.assertEqual(report['status'],'visual_heading_inconsistent')
        self.assertIsNone(report['corrected_pose'])

    def test_rectified_boundary_samples_are_converted_back_to_raw_pixels(self):
        from boundary_geometry import PointGeometry
        self.ranger.distortion = np.array([.1,-.05,.001,-.001,0.])
        self.view = CalibratedGroundView(self.ranger)
        prior,reading,supports = scene(self.ranger)
        geometry = PointGeometry(self.ranger.matrix,self.ranger.distortion,self.ranger.model)
        lines = []
        for support in supports[pixel_key(reading.pixel)]:
            path = geometry.to_geometry(support.boundary_path)
            lines.append(SimpleNamespace(line_id=support.line_id,members=[SimpleNamespace(boundary_path=path)]))
        boundary = SimpleNamespace(lines=lines,config=SimpleNamespace(point_geometry=geometry))
        _,report = self.solve(prior,[reading],supports,boundary_result=boundary)
        self.assertEqual(report['status'],'corrected')
        self.assertAlmostEqual(report['corrected_pose']['heading_deg'],90.,places=4)

    def test_missing_prior_map_or_calibration_cannot_publish_a_pose(self):
        prior,reading,supports = scene(self.ranger)
        for missing in ('prior','model','ranger'):
            args = dict(readings=[reading],geometry_readings=[replace(reading,geometry_only=True)],
                        supports=supports,prior=prior,model=self.model,ranger=self.ranger,
                        candidate_view=candidate_safety_view(prior,self.model,self.view))
            args[missing] = None
            _,report = solve_nearest_corner_pose(**args)
            self.assertFalse(report['pose_corrected'])
            self.assertIsNone(report['corrected_pose'])

    def test_bad_parameters_are_rejected(self):
        for config in ({'identity_tolerance_deg':45.},{'heading_consistency_deg':-1.},
                       {'minimum_edge_span_mm':0.},{'maximum_line_residual_mm':float('nan')}):
            with self.assertRaises(ValueError):
                VisualPoseConfig(**config)


if __name__ == '__main__':
    unittest.main()
