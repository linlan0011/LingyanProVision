from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottom_edge_classifier import BottomEdgeConfig, BottomEdgeResult, LayerCorner, classify_bottom_edges
from calibration.lens_models import project_points
from corner_ranging import CornerRanger
from field_zones import read_zones
from ground_corner_association import (CalibratedGroundView, GroundAssociationConfig,
    associate_ground_corners, apply_ground_identities, circle_templates,
    disk_intersection_witness, heading_rotation)
from pose_io import Pose2D
from safety_identity import load_templates, map_corners


def camera():
    intrinsics = dict(camera_matrix=[[300.,0,320.],[0,300.,240.],[0,0,1.]],
        distortion_coefficients=[0.]*5, image_width=640,image_height=480,model='opencv_pinhole')
    mount = dict(rotation_camera_to_body=[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],
                 camera_position_body_mm=[0.,0.,174.])
    # Keep FOV/association fixtures explicitly bounded independently of defaults.
    return CornerRanger(intrinsics,mount,0.,max_distance_mm=1000.,include_inferred=True)


def measured(ranger, xy, inferred=False):
    rvec, _ = cv2.Rodrigues(ranger.rotation.T)
    pixel=project_points(np.array([[*xy,0.]]),rvec,-ranger.rotation.T@ranger.center,
                         ranger.matrix,ranger.distortion,ranger.model)[0]
    result=ranger.project(pixel,inferred=inferred)
    if not result.valid:
        raise AssertionError(f'invalid fixture {xy}: {result.reason}')
    return result


def lower_layers(readings):
    result=BottomEdgeResult(BottomEdgeConfig(),'nearest_fallback',len(readings),eligible_count=len(readings))
    result.corners=[LayerCorner(tuple(map(int,np.rint(r.pixel))),r.inferred,r.distance_mm,
                               'lower_seed' if i==0 else 'lower',
                               localization_eligible=True)for i,r in enumerate(readings)]
    return result


class GroundAssociationTests(unittest.TestCase):
    def setUp(self):
        self.ranger=camera()
        self.view=CalibratedGroundView(self.ranger)
        self.model=load_templates()

    def associate(self, readings, prior, **config):
        return associate_ground_corners(readings,lower_layers(readings),prior,self.model,
                                        self.view,GroundAssociationConfig(**config))

    def test_single_lower_corner_identifies_13_and_reflects_to_21(self):
        for prior,xy,expected in ((Pose2D(1500,2100,90),(540,-330),'1.3'),
                                  (Pose2D(1500,900,-90),(540,-330),'2.1')):
            with self.subTest(expected=expected):
                reading=measured(self.ranger,xy)
                identities,report=self.associate([reading],prior)
                self.assertEqual(identities[0].corner_id,expected)
                np.testing.assert_allclose(report['vision_vehicle_position_mm'],[prior.x_mm,prior.y_mm])
                self.assertFalse(report['pose_corrected'])
                json.dumps(report,allow_nan=False)

    def test_all_eight_map_ids_use_the_same_rotation_and_matching_rules(self):
        for zone in self.model['safety_zones']:
            for index,corner in enumerate(map_corners(zone),1):
                heading=0 if index in (1,4) else 180
                world_y=150 if index in (1,2) else -150
                body_y=world_y if heading==0 else -world_y
                relative=np.array([540,body_y])
                position=np.array(corner)-heading_rotation(heading)@relative
                prior=Pose2D(*position,heading)
                identities,_=self.associate([measured(self.ranger,relative)],prior)
                with self.subTest(zone=zone['id'],corner=index):
                    self.assertEqual(identities[0].corner_id,f"{zone['id']}.{index}")

    def test_compensated_point_can_be_identified(self):
        reading=measured(self.ranger,(540,-330),True)
        identities,report=self.associate([reading],Pose2D(1500,2100,90))
        self.assertEqual(identities[0].corner_id,'1.3')
        self.assertTrue(report['observations'][0]['inferred'])
        self.assertEqual(report['observations'][0]['measurement_tolerance_mm'],50.)

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

    def test_bounded_view_reuses_cached_polygon_for_pose_radius_and_association(self):
        cached=self.view.body_polygon.copy()
        reading=measured(self.ranger,(540.,-330.))
        with patch.object(self.view,'_body_polygon',side_effect=AssertionError(
                'bounded frame must reuse calibrated body polygon')) as rebuild:
            for prior in (Pose2D(1500.,2100.,90.),Pose2D(1500.,900.,-90.),
                          Pose2D(1500.,1490.,90.),Pose2D(1700.,1700.,45.)):
                for radius in (0.,200.,500.):
                    with self.subTest(prior=prior,radius=radius):
                        expected=cached@heading_rotation(prior.heading_deg).T+[prior.x_mm,prior.y_mm]
                        np.testing.assert_allclose(self.view.footprint(prior),expected)
                        # A larger display/map extent cannot expand a bounded
                        # camera's localization footprint or force new clipping.
                        np.testing.assert_allclose(self.view.footprint(prior,extent_mm=50000.),expected)
                        landmarks=self.view.landmarks(prior,radius,self.model)
                        contour=expected.astype(np.float32).reshape(-1,1,2)
                        for landmark in landmarks:
                            distance=-cv2.pointPolygonTest(contour,
                                tuple(map(float,landmark['map_xy_mm'])),True)
                            self.assertEqual(landmark['possibly_visible'],distance<=radius+1e-4)
                        identities,report=associate_ground_corners([reading],lower_layers([reading]),
                            prior,self.model,self.view,GroundAssociationConfig(prior_radius_mm=radius))
                        np.testing.assert_allclose(report['central_footprint_map_mm'],expected)
                        self.assertEqual(report['landmarks'],landmarks)
                        if radius in (0.,200.) and prior.y_mm in (2100.,900.):
                            self.assertEqual(identities[0].corner_id,
                                             '1.3' if prior.y_mm==2100. else '2.1')
            rebuild.assert_not_called()
        np.testing.assert_array_equal(self.view.body_polygon,cached)

    def test_distorted_border_footprint_contains_valid_ground_rays(self):
        self.ranger.distortion=np.array([.15,-.2,.002,-.002,.08])
        view=CalibratedGroundView(self.ranger)
        contour=view.body_polygon.astype(np.float32).reshape(-1,1,2)
        for u in np.linspace(0,639,29):
            for v in np.linspace(241,479,17):
                reading=self.ranger.project((u,v))
                if reading.valid:
                    self.assertGreaterEqual(cv2.pointPolygonTest(contour,(reading.x_mm,reading.y_mm),True),-.01)

    def test_upper_and_unknown_points_never_get_new_map_ids(self):
        reading=measured(self.ranger,(540,-330))
        layers=lower_layers([reading])
        for label in ('upper','unknown'):
            layers.corners[0].layer=label
            identities,report=associate_ground_corners([reading],layers,Pose2D(1500,2100,90),
                                                       self.model,self.view)
            self.assertIsNone(identities[0].corner_id)
            self.assertFalse(report['observations'])

    def test_geometry_only_projection_cannot_be_used_as_localization_measurement(self):
        reading=replace(measured(self.ranger,(540,-330)),geometry_only=True)
        layers=lower_layers([reading])
        self.assertTrue(reading.valid)
        self.assertTrue(layers.corners[0].localization_eligible)
        identities,report=associate_ground_corners([reading],layers,Pose2D(1500,2100,90),
                                                  self.model,self.view)
        self.assertIsNone(identities[0].corner_id)
        self.assertFalse(report['observations'])
        self.assertIsNone(report['vision_vehicle_position_mm'])
        self.assertIsNone(report['compatible_vehicle_position_mm'])

    def test_large_radius_single_point_remains_ambiguous(self):
        reading=measured(self.ranger,(700,0))
        identities,report=self.associate([reading],Pose2D(1500,2100,90),prior_radius_mm=500)
        self.assertIsNone(identities[0].corner_id)
        self.assertGreater(len(identities[0].candidates),1)
        self.assertEqual(report['status'],'ambiguous')

    def test_two_ground_points_yield_one_shared_pose(self):
        readings=[measured(self.ranger,(540,-330)),measured(self.ranger,(540,330),True)]
        identities,report=self.associate(readings,Pose2D(1540,2070,90))
        self.assertEqual([i.corner_id for i in identities],['1.3','1.4'])
        self.assertEqual(report['hypothesis_count'],1)
        np.testing.assert_allclose(report['vision_vehicle_position_mm'],[1500,2100],atol=1e-6)

    def test_individual_matches_with_inconsistent_vehicle_positions_are_not_confirmed(self):
        readings=[measured(self.ranger,(540,-230)),measured(self.ranger,(540,230))]
        identities,report=self.associate(readings,Pose2D(1500,2100,90))
        self.assertTrue(all(i.candidates for i in identities))
        self.assertTrue(all(i.corner_id is None for i in identities))
        self.assertEqual(report['matched_count'],1)
        self.assertEqual(report['hypothesis_count'],2)

    def test_duplicate_landmark_assignment_is_rejected(self):
        readings=[measured(self.ranger,(540,-330)),measured(self.ranger,(541,-329),True)]
        identities,report=self.associate(readings,Pose2D(1500,2100,90))
        self.assertTrue(all(i.corner_id is None for i in identities))
        self.assertEqual(report['matched_count'],1)

    def test_three_pairwise_overlapping_disks_can_have_empty_joint_intersection(self):
        disks=[([0,0],1),([1.9,0],1),([.95,1.9*math.sqrt(3)/2],1)]
        self.assertIsNone(disk_intersection_witness(disks))
        self.assertIsNotNone(disk_intersection_witness([([0,0],1),([2,0],1)]))
        self.assertIsNotNone(disk_intersection_witness([([0,0],0),([0,0],1)]))

    def test_search_limit_cannot_confirm_a_truncated_solution_set(self):
        reading=measured(self.ranger,(540,-330))
        identities,report=self.associate([reading],Pose2D(1500,2100,90),max_search_nodes=1)
        self.assertEqual(report['status'],'search_limit_no_confirmation')
        self.assertIsNone(identities[0].corner_id)

    def test_missing_prior_map_or_changed_calibration_clears_ids(self):
        reading=measured(self.ranger,(540,-330))
        for prior,model,status in ((None,self.model,'waiting_for_prior'),
                                   (Pose2D(1500,2100,90),None,'map_unavailable')):
            identities,report=associate_ground_corners([reading],lower_layers([reading]),prior,model,self.view)
            self.assertEqual(report['status'],status)
            self.assertIsNone(identities[0].corner_id)
        with patch.object(self.ranger,'check_sources',return_value=False):
            identities,report=self.associate([reading],Pose2D(1500,2100,90))
            self.assertEqual(report['status'],'calibration_unavailable')
            self.assertIsNone(identities[0].corner_id)

    def test_legacy_id_replaced_or_cleared_without_mutating_measurements(self):
        reading=replace(measured(self.ranger,(540,-330)),label='R1/2.2',corner_id='2.2')
        identities,_=self.associate([reading],Pose2D(1500,2100,90))
        updated=apply_ground_identities([reading],identities)[0]
        self.assertEqual(updated.corner_id,'1.3')
        self.assertEqual(updated.label,'R1/1.3')
        self.assertEqual(reading.corner_id,'2.2')
        identities[0].corner_id=None
        self.assertIsNone(apply_ground_identities([reading],identities)[0].corner_id)

    def test_out_of_range_point_participates_in_ranking_but_not_localization(self):
        inside=measured(self.ranger,(540,-330))
        geometric=replace(inside,pixel=(1.,2.),inferred=True,x_mm=1200.,y_mm=0.,
                          distance_mm=1200.)
        outside=replace(geometric,valid=False,reason='beyond_distance_limit',
                        x_mm=None,y_mm=None,z_mm=None,distance_mm=None,bearing_deg=None)
        layers=classify_bottom_edges([outside,inside],None,
                                     ranking_readings=[geometric,inside])
        self.assertEqual(layers.candidate_count,2)
        self.assertEqual(layers.eligible_count,1)
        self.assertEqual(layers.ranking_count,2)
        self.assertEqual(len(layers.corners),2)
        self.assertEqual(layers.status,'nearest_fallback')
        self.assertEqual(layers.lower_seed,tuple(map(int,np.rint(inside.pixel))))
        self.assertFalse(layers.by_pixel[(1,2)].localization_eligible)
        identities,report=associate_ground_corners([outside,inside],layers,
            Pose2D(1500,2100,90),self.model,self.view)
        self.assertIsNone(identities[0].corner_id)
        self.assertEqual(identities[1].corner_id,'1.3')
        self.assertEqual(len(report['observations']),1)

        # A sole out-of-range point is still the geometric LOW* fallback,
        # but neither a map identity nor a visual vehicle position is emitted.
        only_outside=classify_bottom_edges([outside],None,ranking_readings=[geometric])
        self.assertEqual(only_outside.status,'nearest_fallback')
        self.assertEqual(only_outside.lower_seed,(1,2))
        self.assertFalse(only_outside.corners[0].localization_eligible)
        identities,report=associate_ground_corners([outside],only_outside,
            Pose2D(1500,2100,90),self.model,self.view)
        self.assertIsNone(identities[0].corner_id)
        self.assertFalse(report['observations'])
        self.assertIsNone(report['vision_vehicle_position_mm'])

    def test_out_of_range_dimension_matched_lower_keeps_label_without_map_identity(self):
        extended=camera()
        extended.max_distance_mm=2500.
        geometric=[measured(extended,(1600.,0.),True),
                   measured(extended,(1300.,100.)),
                   measured(extended,(540.,-330.)),
                   measured(extended,(1200.,-330.),True)]
        readings=[replace(r,valid=False,reason='beyond_distance_limit',
                          x_mm=None,y_mm=None,z_mm=None,distance_mm=None,bearing_deg=None)
                  if r.distance_mm>1000. else r for r in geometric]
        from test_bottom_edge_classifier import graph
        keys=[tuple(map(int,np.rint(r.pixel)))for r in geometric]
        layers=classify_bottom_edges(readings,graph(keys[:2]),ranking_readings=geometric)
        self.assertEqual(layers.by_pixel[keys[3]].layer,'lower')
        self.assertEqual(layers.by_pixel[keys[3]].matched_edge,'length')
        self.assertFalse(layers.by_pixel[keys[3]].localization_eligible)
        labels_before={p:c.layer for p,c in layers.by_pixel.items()}
        identities,report=associate_ground_corners(readings,layers,Pose2D(1500,2100,90),
                                                 self.model,self.view)
        self.assertEqual(identities[2].corner_id,'1.3')
        self.assertIsNone(identities[3].corner_id)
        self.assertEqual(len(report['observations']),1)
        self.assertEqual({p:c.layer for p,c in layers.by_pixel.items()},labels_before)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({'prior_radius_mm':-1},{'visible_tolerance_mm':math.nan},
                       {'max_search_nodes':0}):
            with self.assertRaises(ValueError):
                GroundAssociationConfig(**kwargs)

    def test_prior_radius_boundary_and_no_match_are_explicit(self):
        reading=measured(self.ranger,(540,-330))
        for offset,expected in ((200,'1.3'),(201,None)):
            identities,report=self.associate([reading],Pose2D(1500+offset,2100,90),
                visible_tolerance_mm=0.)
            self.assertEqual(identities[0].corner_id,expected)
            if expected is None:
                self.assertEqual(report['status'],'no_match')


class GroundPipelineTests(unittest.TestCase):
    def test_launcher_forwards_radius_limit_and_disable_switch(self):
        from run_app import build_parser, vision_command
        args=build_parser().parse_args(['--pose-radius-mm','200'])
        command=vision_command(args,Path('stop'))
        self.assertEqual(command[command.index('--pose-radius-mm')+1],'200')
        self.assertEqual(float(command[command.index('--range-max-mm')+1]),1000.)
        self.assertIn('--ground-association',command)
        bounded=vision_command(build_parser().parse_args(['--range-max-mm','1000']),Path('stop'))
        self.assertEqual(float(bounded[bounded.index('--range-max-mm')+1]),1000.)
        command=vision_command(build_parser().parse_args(['--no-ground-association']),Path('stop'))
        self.assertIn('--no-ground-association',command)

    def test_inferred_lower_point_gets_id_in_actual_pipeline_report_and_preview(self):
        import purple_camera_segmentation as vision
        import preview_display
        from pose_io import write_pose_atomic
        ranger=camera()
        ranger.include_inferred=False
        pixel=np.rint(measured(camera(),(540,-330),True).pixel).astype(int)
        frame=np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,320),(550,430),(140,30,140),-1)
        class Camera:
            def read(self):return True,frame.copy()
            def get(self,_):return 30.
            def release(self):pass
        trackbars={}
        with tempfile.TemporaryDirectory() as directory, patch('pose_association.PROJECT',Path(directory)):
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,Pose2D(1500,2100,90))
            before=pose.read_bytes()
            output=Path(directory)/'output/association/lingyan/latest.json'
            argv=['vision','--scene','front-facing','--ranging','--target-z-mm','0',
                  '--range-max-mm','1000',
                  '--pose-file',str(pose),'--association-output',str(output),'--parameters']
            with patch('sys.argv',argv),patch.object(vision,'open_camera',return_value=Camera()), \
                 patch('corner_ranging.CornerRanger.from_camera',return_value=ranger), \
                 patch.object(vision,'find_effective_boundary_corners',return_value=([],[(pixel,90.)])), \
                 patch.object(preview_display,'_corner_labels',wraps=preview_display._corner_labels) as labels, \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda n,w,v,*_:trackbars.update({n:v})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda n,w:trackbars[n]), \
                 patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'imshow'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
                self.assertTrue(any(i.corner_id=='1.3' for call in labels.call_args_list
                                    for i in call.args[2]))
            data=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(pose.read_bytes(),before)
        self.assertEqual(data['max_distance_mm'],1000.)
        self.assertEqual(data['confirmed_corner_ids'],['1.3'])
        self.assertTrue(data['corners'][0]['inferred'])
        self.assertEqual(data['corners'][0]['corner_id'],'1.3')
        self.assertEqual(data['ground_association']['config']['prior_radius_mm'],200.)
        self.assertEqual(data['ground_association']['status'],'unique')
        self.assertFalse(data['pose_corrected'])
        self.assertFalse(ranger.include_inferred)

    def test_usb_profile_default_cap_retains_full_ranking_before_simulated_pose_match(self):
        import purple_camera_segmentation as vision
        from copy import copy
        from pose_io import write_pose_atomic
        from test_bottom_edge_classifier import graph
        ranger=camera()
        ranger.include_inferred=False
        ranger.matrix=np.array([[900.,0.,960.],[0.,675.,540.],[0.,0.,1.]])
        ranger.size=(1920,1080)
        extended=copy(ranger)
        extended.include_inferred=True
        extended.max_distance_mm=2500.
        geometric=[measured(extended,(1600.,0.),True),
                   measured(extended,(1300.,100.)),
                   measured(extended,(540.,-330.)),
                   measured(extended,(1200.,-330.),True)]
        keys=[tuple(map(int,np.rint(r.pixel)))for r in geometric]
        supports=graph(keys[:2],keys[2:])
        visible=[(keys[i],90.)for i in (1,2)]
        hidden=[(keys[i],90.)for i in (0,3)]
        frame=np.full((1080,1920,3),235,np.uint8)
        cv2.rectangle(frame,(250,580),(1700,1000),(140,30,140),-1)
        class Camera:
            def read(self):return True,frame.copy()
            def getBackendName(self):return 'MSMF'
            def get(self,prop):
                return {cv2.CAP_PROP_FRAME_WIDTH:1920.,
                        cv2.CAP_PROP_FRAME_HEIGHT:1080.,
                        cv2.CAP_PROP_FPS:30.,
                        cv2.CAP_PROP_FOURCC:float(cv2.VideoWriter_fourcc(*'MJPG'))}.get(prop,0.)
            def release(self):pass
        def detector(*args,**kwargs):
            kwargs['boundary_result']['result']=supports
            return visible,hidden
        trackbars={}
        with tempfile.TemporaryDirectory() as directory,patch('pose_association.PROJECT',Path(directory)):
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,Pose2D(1500.,2100.,90.))
            before=pose.read_bytes()
            output=Path(directory)/'output/association/usb0bda3035_1080p/latest.json'
            argv=['vision','--camera','usb0bda3035_1080p','--scene','front-facing',
                  '--ranging','--target-z-mm','0','--pose-file',str(pose),
                  '--association-output',str(output),'--parameters']
            with patch('sys.argv',argv),patch.object(vision,'open_camera',return_value=Camera()), \
                 patch('corner_ranging.CornerRanger.from_camera',return_value=ranger), \
                 patch.object(vision,'find_effective_boundary_corners',side_effect=detector), \
                 patch.object(vision,'draw_boundary_result'), \
                 patch.object(vision,'compose_preview',return_value=frame), \
                 patch.object(vision,'classify_bottom_edges',wraps=classify_bottom_edges) as classify, \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda n,w,v,*_:trackbars.update({n:v})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda n,w:trackbars[n]), \
                 patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'imshow'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
                ranked=classify.call_args.kwargs['ranking_readings']
                self.assertEqual(len(ranked),4)
                self.assertTrue(all(r.valid and r.geometry_only for r in ranked))
                self.assertEqual(sum(r.valid for r in classify.call_args.args[0]),1)
            data=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(pose.read_bytes(),before)
        layers=data['bottom_edge_test']
        self.assertEqual(data['max_distance_mm'],1000.)
        self.assertEqual(layers['candidate_count'],4)
        self.assertEqual(layers['ranking_count'],4)
        self.assertEqual(layers['eligible_count'],1)
        self.assertEqual(layers['upper_seed'],list(keys[0]))
        self.assertEqual(layers['lower_seed'],list(keys[2]))
        self.assertEqual(layers['upper_count'],2)
        self.assertEqual(layers['lower_count'],2)
        self.assertEqual(data['confirmed_corner_ids'],['1.3'])
        self.assertFalse(data['pose_corrected'])
        self.assertTrue(all(not r['geometry_only']for r in data['corners']))
        self.assertEqual(len(data['ground_association']['observations']),1)
        self.assertFalse(ranger.include_inferred)

    def test_map_shows_prior_and_view_and_clears_when_simulated_pose_changes(self):
        import tkinter as tk
        from pose_map_gui import PoseMapApp, build_parser
        from pose_association import association_payload
        ranger=camera()
        reading=measured(ranger,(540,-330),True)
        prior=Pose2D(1500,2100,90)
        identities,ground=associate_ground_corners([reading],lower_layers([reading]),prior,
            load_templates(),CalibratedGroundView(ranger))
        data=association_payload('test',1,ranger,apply_ground_identities([reading],identities),prior,identities)
        data['ground_association']=ground
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pose.json'
            root=tk.Tk();root.withdraw()
            app=PoseMapApp(root,build_parser().parse_args(['--output',str(path),
                             '--x','1500','--y','2100','--heading','90']))
            try:
                root.update()
                app.publish_pose()
                before=path.read_bytes()
                app.set_live_association(data)
                self.assertTrue(app.canvas.find_withtag('pose_uncertainty'))
                self.assertTrue(app.canvas.find_withtag('central_view'))
                self.assertTrue(app.canvas.find_withtag('possible_view'))
                self.assertIn('1.3',app.live_identity_var.get())
                self.assertIn('补偿',app.live_identity_var.get())
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(app.pose,prior)
                app.pose=Pose2D(1800,2100,90)
                app._draw_pose()
                self.assertFalse(app.canvas.find_withtag('ground_association'))
            finally:
                app.close()


if __name__=='__main__':
    unittest.main()
