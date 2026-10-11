from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from field_zones import FieldZone, read_zones, write_zones_atomic
from pose_io import Pose2D, write_pose_atomic
from safety_identity import (ActiveTemplate, CornerObservation, TemplateReader,
    active_templates, associate_corners, directed_rays, supported_corner_rays,
    load_templates, map_corners)
import purple_camera_segmentation as vision


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.model = load_templates()
        self.zones = read_zones(Path(__file__).with_name("field_zones.json"))

    def test_clockwise_ids(self):
        self.assertEqual(map_corners(self.model['safety_zones'][1]),
                         [(1170, 360), (1830, 360), (1830, 0), (1170, 0)])

    def test_ten_half_field_viewpoints(self):
        cases = [(Pose2D(2200, 2820, 180), 1, 'east'),
                 (Pose2D(800, 2820, 0), 1, 'west'),
                 (Pose2D(2200, 180, 180), 2, 'east'),
                 (Pose2D(800, 180, 0), 2, 'west'),
                 (Pose2D(1500, 900, -90), 2, 'north'),
                 (Pose2D(1500, 2100, 90), 1, 'south'),
                 (Pose2D(800, 2100, 45), 1, 'southwest'),
                 (Pose2D(2200, 2100, 135), 1, 'southeast'),
                 (Pose2D(800, 900, -45), 2, 'northwest'),
                 (Pose2D(2200, 900, -135), 2, 'northeast')]
        # Independent expected order: image near-left, near-right, far-left, far-right.
        expected = {'east': [3,2,4,1], 'west': [1,4,2,3],
                    'north': [2,1,3,4], 'south': [4,3,1,2]}
        observed = [(0, 50), (180, 130), (0, -130), (180, -50)]
        for pose, safety_id, side in cases:
            with self.subTest(safety_id=safety_id, side=side):
                templates, _ = active_templates(pose, self.zones, self.model)
                self.assertEqual([(t.safety_id, t.side) for t in templates], [(safety_id, side)])
                if templates[0].identity_mode == 'pose_projection':
                    points = {
                        'southwest': ((180, 100), f'{safety_id}.1'),
                        'southeast': ((180, 100), f'{safety_id}.4'),
                        'northwest': ((180, 100), f'{safety_id}.2'),
                        'northeast': ((180, 100), f'{safety_id}.3'),
                    }
                    point, corner_id = points[side]
                    result = associate_corners(
                        [CornerObservation(point, False, ())], templates, self.model,
                        pose=pose, image_width=640,
                    )[0]
                    self.assertEqual(result.corner_id, corner_id)
                    continue
                observations = [CornerObservation((i*40, 100), False, rays) for i,rays in enumerate(observed)]
                results = associate_corners(observations, templates, self.model)
                self.assertEqual([r.corner_id for r in results], [f'{safety_id}.{i}' for i in expected[side]])

    def test_half_boundary_diagonal_and_rejections(self):
        diagonal, _ = active_templates(Pose2D(2143.2, 2101.3, 113.7), self.zones, self.model)
        self.assertEqual([(t.safety_id, t.side, t.scene) for t in diagonal], [(1, 'southeast', 'general')])
        self.assertEqual(active_templates(Pose2D(1500, 1500, 90), self.zones, self.model)[0][0].safety_id, 1)
        self.assertFalse(active_templates(Pose2D(1500, 1500, -90), self.zones, self.model)[0])
        self.assertEqual(active_templates(Pose2D(1500, 1499.9, -90), self.zones, self.model)[0][0].safety_id, 2)
        for pose in (None, Pose2D(1500,1500,0), Pose2D(2200,180,90),
                     Pose2D(3100,180,180), Pose2D(2200,20,90)):
            self.assertFalse(active_templates(pose, self.zones, self.model)[0])
        unbound = [replace(z, safety_zone_id=0, view_side='') for z in self.zones]
        self.assertFalse(active_templates(Pose2D(2200,180,180), unbound, self.model)[0])

    def test_live_southwest_diagonal_four_point_extrema_mapping(self):
        # Current live frame: one component exposes four corners.  Their image
        # extrema are left=(201,140), top=(392,100), right=(519,129), bottom=(502,183).
        pose = Pose2D(893.643, 1917.967, 60.607)
        templates, _ = active_templates(pose, self.zones, self.model)
        observations = [
            CornerObservation((201, 140), False, (), 1),
            CornerObservation((392, 100), False, (), 1),
            CornerObservation((519, 129), False, (), 1),
            CornerObservation((502, 183), False, (), 1),
        ]
        results = associate_corners(observations, templates, self.model, pose=pose, image_width=640)
        self.assertEqual([result.corner_id for result in results], ['1.1', '1.2', '1.3', '1.4'])

    def test_single_edge_ambiguity_and_duplicate_conflicts(self):
        templates, _ = active_templates(Pose2D(2200,180,180), self.zones, self.model)
        observations = [CornerObservation((100,100), False, (180,)),
                        CornerObservation((200,100), True, (180,130))]
        results = associate_corners(observations, templates, self.model)
        self.assertIsNone(results[0].corner_id)
        self.assertEqual(results[1].corner_id, '2.2')
        ambiguous = templates + [replace(templates[0], safety_id=1)]
        result = associate_corners(observations[1:], ambiguous, self.model)[0]
        self.assertIsNone(result.corner_id)
        self.assertEqual(result.candidates, {'1.2','2.2'})
        duplicated = [observations[1], replace(observations[1], point=(250,100))]
        self.assertTrue(all(r.corner_id is None and r.reason == 'duplicate corner conflict'
                            for r in associate_corners(duplicated, templates, self.model)))

    def test_rays_are_endpoint_directed_and_not_infinite_lines(self):
        def segment(a,b):
            return SimpleNamespace(fitted_start=np.array(a), fitted_end=np.array(b))
        segments = [segment((430,365),(320,365)), segment((390,285),(430,365)),
                    segment((300,300),(400,300)),  # unrelated remote line
                    segment((430,265),(430,465))]  # passes through corner, not endpoint
        rays = directed_rays((430,365),segments,False,640,self.model)
        self.assertEqual(len(rays),2)
        templates,_ = active_templates(Pose2D(2200,200,180), self.zones,self.model)
        result = associate_corners([CornerObservation((430,365),False,rays)],templates,self.model)[0]
        self.assertEqual(result.corner_id,'2.2')

    def test_inferred_gap_is_bounded(self):
        s = SimpleNamespace(fitted_start=np.array((380,365)),fitted_end=np.array((320,365)))
        self.assertFalse(directed_rays((430,365),[s],False,640,self.model))
        self.assertTrue(directed_rays((430,365),[s],True,640,self.model))
        self.assertFalse(directed_rays((500,365),[s],True,640,self.model))

    def test_supported_corner_directions_do_not_flip_at_subpixel_endpoints(self):
        for epsilon in (-.01, 0., .01):
            sides = [SimpleNamespace(fitted_start=np.array((100+epsilon,100)),
                                     fitted_end=np.array((300,100))),
                     SimpleNamespace(fitted_start=np.array((100,100+epsilon)),
                                     fitted_end=np.array((100,0)))]
            rays = supported_corner_rays((100,100), sides)
            self.assertEqual(len(rays), 2)
            self.assertAlmostEqual(rays[0], 0.)
            self.assertAlmostEqual(rays[1], 90.)

    def test_front_identity_reuses_validated_support_instead_of_rejecting_inset(self):
        point = np.array((100,100))
        sides = [SimpleNamespace(fitted_start=np.array((120,100)), fitted_end=np.array((300,100))),
                 SimpleNamespace(fitted_start=np.array((120,80)), fitted_end=np.array((0,200)))]
        self.assertFalse(directed_rays(point, sides, False, 640, self.model))
        mask = np.zeros((480,640), np.uint8)
        contour = np.array([(100,100),(300,100),(0,200)], np.int32).reshape(-1,1,2)
        recognitions = [((contour,), [(point,135.)], [])]
        templates = [ActiveTemplate(1,'south','front-facing',(4,3,1,2))]
        with patch.object(vision, 'extract_valid_boundary_segments', return_value=[]), \
                patch.object(vision, 'directed_rays', side_effect=AssertionError('validated sides must be reused')):
            result = vision.identify_frame_corners(
                recognitions, {1}, mask, mask, vision.FRONT_RULES, templates,
                self.model, 'front-facing', corner_supports={(100,100):sides})
        self.assertEqual(result[0].corner_id, '1.1')

    def test_binding_survives_rename_and_old_files_remain_unbound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'zones.json'
            zone = replace(self.zones[0], name='重命名后的观察区')
            write_zones_atomic(path,[zone])
            self.assertEqual(read_zones(path),[zone])
            raw=json.loads(path.read_text(encoding='utf-8'))
            del raw['zones'][0]['safety_zone_id']
            del raw['zones'][0]['view_side']
            path.write_text(json.dumps(raw),encoding='utf-8')
            self.assertEqual(read_zones(path)[0].safety_zone_id,0)
            with self.assertRaises(ValueError):
                replace(zone,view_side='').validated()

    def test_invalid_or_missing_template_clears_cached_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'templates.json'
            path.write_text(json.dumps(self.model),encoding='utf-8')
            reader=TemplateReader(path)
            self.assertIsNotNone(reader.poll())
            path.write_text('{}',encoding='utf-8')
            self.assertIsNone(reader.poll())
            self.assertIsNotNone(reader.error)
            path.write_text('[1,2,3]',encoding='utf-8')
            self.assertIsNone(reader.poll())
            path.unlink()
            self.assertIsNone(reader.poll())


def synthetic_frame():
    frame = np.full((480,640,3), 235, dtype=np.uint8)
    # Clearly synthetic trapezoid; all four exterior boundaries touch white floor.
    cv2.fillPoly(frame,[np.array([(180,365),(460,365),(380,270),(260,270)])],(140,30,140))
    return frame


class FrontalGeometryTests(unittest.TestCase):
    def test_nearby_front_and_back_corners_keep_distinct_support_pairs(self):
        def line(a,b):
            return SimpleNamespace(fitted_start=np.array(a),fitted_end=np.array(b))
        near = np.array((1048,543))
        far = np.array((973,410))
        side = line(far,near)
        supports = {tuple(near):(line((0,543),near),side),
                    tuple(far):(line((278,410),far),side)}
        visible,inferred = vision.resolve_front_corner_overlap(
            [(near,61.),(far,119.)],[],(1080,1920,3),
            vision.USB_0BDA_3035_SCENE_RULES['front-facing'],corner_supports=supports)
        self.assertEqual(len(visible),2)
        self.assertFalse(inferred)

    def test_duplicate_support_pair_still_merges(self):
        pair = [SimpleNamespace(fitted_start=np.array((100,100)),fitted_end=np.array((300,100))),
                SimpleNamespace(fitted_start=np.array((100,100)),fitted_end=np.array((100,300)))]
        a,b = np.array((100,100)),np.array((105,105))
        visible,_ = vision.resolve_front_corner_overlap(
            [(a,90.),(b,90.)],[],(480,640,3),corner_supports={tuple(a):pair,tuple(b):pair})
        self.assertEqual([tuple(p) for p,_ in visible],[(105,105)])

    def test_compensation_still_wins_nearby_visible_conflict(self):
        a,b = np.array((100,100)),np.array((105,105))
        visible,inferred = vision.resolve_front_corner_overlap(
            [(a,90.)],[(b,90.)],(480,640,3),corner_supports={})
        self.assertFalse(visible)
        self.assertEqual([tuple(p) for p,_ in inferred],[(105,105)])

    def detect(self, frame):
        rules = vision.FRONT_RULES
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (rules.h_low,rules.s_min,rules.v_min), (rules.h_high,255,255))
        white = vision.make_white_field_mask(hsv,rules.white_s_max,rules.white_v_min)
        contours = [vision.find_largest_purple(mask, rules.min_area)[0][-1]]
        supports = {}
        visible = vision.find_front_connected_fitted_corners_across_components(
            contours,white,mask,rules,corner_supports=supports)
        inferred = vision.find_front_occluded_corners_across_components(
            contours,white,mask,visible,rules,corner_supports=supports)
        return visible,inferred,supports

    def test_exact_shared_endpoints_keep_all_corners_without_compensation(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,100),(500,400),(140,30,140),-1)
        visible,inferred,supports = self.detect(frame)
        self.assertEqual({tuple(p) for p,_ in visible}, {(100,100),(500,100),(500,400),(100,400)})
        self.assertTrue(all(abs(a-90.)<.01 for _,a in visible))
        self.assertFalse(inferred)
        self.assertEqual(len(supports),4)

    def test_occluded_corner_uses_valid_sides_and_retains_compensation(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.fillPoly(frame,[np.array([(80,380),(520,380),(430,180),(170,180)])],(140,30,140))
        cv2.rectangle(frame,(40,330),(145,410),(0,0,0),-1)
        # A second obstruction leaves two separate green chains, the existing
        # conservative gap condition required by the compensation detector.
        cv2.rectangle(frame,(280,160),(340,205),(0,0,0),-1)
        visible,inferred,supports = self.detect(frame)
        self.assertEqual(len(inferred),1)
        self.assertLess(np.linalg.norm(inferred[0][0]-np.array((80,380))),3.)
        self.assertTrue(all(np.linalg.norm(p-np.array((80,380)))>12 for p,_ in visible))
        self.assertIn(tuple(inferred[0][0]),supports)


class EffectiveBoundaryCornerTests(unittest.TestCase):
    def detect(self, frame, scene='front-facing'):
        rules = vision.SCENE_RULES[scene]
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,(rules.h_low,rules.s_min,rules.v_min),(rules.h_high,255,255))
        white = vision.make_white_field_mask(hsv,rules.white_s_max,rules.white_v_min)
        contours = [d[-1] for d in vision.find_largest_purple(mask,rules.min_area)]
        supports = {}
        visible,inferred = vision.find_effective_boundary_corners(
            contours,white,mask,rules,12,corner_supports=supports)
        return visible,inferred,supports

    def test_short_projected_side_and_obtuse_corner_are_retained(self):
        frame = np.full((1080,1920,3),235,np.uint8)
        points = np.array([(150,650),(1700,650),(1694,555),(600,555)])
        cv2.fillPoly(frame,[points],(140,30,140))
        for scene in ('front-facing','general'):
            visible,inferred,_ = self.detect(frame,scene)
            self.assertEqual(len(visible),4)
            self.assertTrue(any(a>165 for _,a in visible))
            self.assertFalse(inferred)
            for point in points:
                self.assertLessEqual(min(np.linalg.norm(p-point) for p,_ in visible),3.)

    def test_rotated_views_have_same_four_corners(self):
        center = np.array((320,320))
        base = np.array([(-100,-70),(100,-70),(130,70),(-130,70)])
        for degrees in (0,35,90,140):
            theta = np.radians(degrees)
            rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            points = np.rint(base@rotation.T+center).astype(np.int32)
            frame = np.full((640,640,3),235,np.uint8)
            cv2.fillPoly(frame,[points],(140,30,140))
            visible,inferred,_ = self.detect(frame,'general')
            self.assertEqual(len(visible),4,(degrees,visible))
            self.assertFalse(inferred)
            for point in points:
                self.assertLess(min(np.linalg.norm(p-point) for p,_ in visible),3.)

    def test_single_local_occlusion_is_inferred_even_if_other_edges_connect(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.fillPoly(frame,[np.array([(80,380),(520,380),(430,180),(170,180)])],(140,30,140))
        cv2.rectangle(frame,(40,330),(145,410),(0,0,0),-1)
        for source,expected in [(frame,np.array((80,380))),
                                (cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE),np.array((99,80)))]:
            visible,inferred,supports = self.detect(source)
            self.assertEqual(len(inferred),1,inferred)
            self.assertLess(np.linalg.norm(inferred[0][0]-expected),3.)
            self.assertTrue(all(np.linalg.norm(p-expected)>12 for p,_ in visible))
            self.assertIn(tuple(inferred[0][0]),supports)

    def test_crop_does_not_create_corner_or_compensation(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(-100,100),(200,300),(140,30,140),-1)
        visible,inferred,_ = self.detect(frame)
        self.assertEqual({tuple(p) for p,_ in visible},{(200,100),(200,300)})
        self.assertFalse(inferred)

    def test_straight_edge_gap_is_not_a_corner(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,100),(500,400),(140,30,140),-1)
        cv2.rectangle(frame,(280,360),(320,430),(0,0,0),-1)
        visible,inferred,_ = self.detect(frame)
        self.assertEqual({tuple(p) for p,_ in visible},{(100,100),(500,100),(500,400),(100,400)})
        self.assertFalse(inferred)

    def test_separate_closed_green_regions_do_not_infer_between_them(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,100),(200,200),(140,30,140),-1)
        cv2.rectangle(frame,(240,240),(340,340),(140,30,140),-1)
        visible,inferred,_ = self.detect(frame)
        self.assertEqual(len(visible),8)
        self.assertFalse(inferred)


class CameraPipelineTests(unittest.TestCase):
    def test_live_main_with_synthetic_camera_and_pose_file(self):
        self.run_pipeline(Pose2D(2200,200,180), {'2.2','2.3'}, 179, 'synthetic_pipeline.png')

    def test_front_detector_from_north_and_south(self):
        self.run_pipeline(Pose2D(1500,2100,90), {'1.3','1.4'}, 80, 'synthetic_front_1.png')
        self.run_pipeline(Pose2D(1500,900,-90), {'2.1','2.2'}, 80, 'synthetic_front_2.png')

    def test_general_main_uses_relaxed_effective_boundary_detector(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.fillPoly(frame,[np.array([(50,380),(590,380),(585,340),(200,340)])],(140,30,140))
        with patch(__name__+'.synthetic_frame',return_value=frame):
            results = self.run_pipeline(Pose2D(800,2100,45),set(),179,'synthetic_general.png')
        self.assertEqual(len(results),4)

    def run_pipeline(self, pose_value, expected_ids, expected_white_s, filename):
        frame=synthetic_frame()
        class Camera:
            def read(self): return True,frame.copy()
            def get(self,_): return 30
            def release(self): pass
        trackbars={}
        previews=[]
        captured=[]
        real_associate=vision.identify_frame_corners
        def identify(*args, **kwargs):
            result=real_associate(*args, **kwargs)
            captured.append(result)
            return result
        with tempfile.TemporaryDirectory() as directory:
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,pose_value)
            with patch('sys.argv',['vision','--parameters','--scene','auto','--pose-file',str(pose),
                                  '--zones-file',str(Path(__file__).with_name('field_zones.json'))]), \
                 patch.object(vision,'open_camera',return_value=Camera()), \
                 patch.object(vision,'identify_frame_corners',side_effect=identify), \
                 patch.object(vision,'preview_dimensions',return_value=(1440,792)), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda name,win,value,*_:trackbars.update({name:value})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda name,win:trackbars[name]), \
                 patch.object(cv2,'setTrackbarPos',side_effect=lambda name,win,value:trackbars.update({name:value})), \
                 patch.object(cv2,'getWindowProperty',return_value=1), \
                 patch.object(cv2,'imshow',side_effect=lambda win,image:previews.append(image.copy()) if win.startswith('Camera') else None), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
        ids={r.corner_id for r in captured[0] if r.corner_id}
        self.assertTrue(expected_ids <= ids, (expected_ids, ids))
        self.assertEqual(trackbars['white S max'],expected_white_s)
        self.assertEqual(previews[0].shape,(792,1440,3))
        output=Path(__file__).resolve().parent/'照片'
        output.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(output/filename),previews[0])
        return captured[0]


if __name__ == '__main__':
    unittest.main()
