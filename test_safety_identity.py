from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from field_zones import read_zones, write_zones_atomic
from pose_io import Pose2D, write_pose_atomic
from safety_identity import (TemplateReader, active_templates, load_templates, map_corners)
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
        for pose, safety_id, side in cases:
            with self.subTest(safety_id=safety_id, side=side):
                templates, _ = active_templates(pose, self.zones, self.model)
                self.assertEqual([(t.safety_id, t.side) for t in templates], [(safety_id, side)])
                expected_scene = ('start-facing' if side in ('east', 'west') else
                                  'front-facing' if side in ('north', 'south') else 'general')
                self.assertEqual(templates[0].scene, expected_scene)

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
        for scene in ('start-facing','front-facing','general'):
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
            for scene in ('start-facing','front-facing','general'):
                visible,inferred,_ = self.detect(frame,scene)
                self.assertEqual(len(visible),4,(scene,degrees,visible))
                self.assertFalse(inferred)
                for point in points:
                    self.assertLess(min(np.linalg.norm(p-point) for p,_ in visible),3.)

    def test_single_local_occlusion_is_inferred_even_if_other_edges_connect(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.fillPoly(frame,[np.array([(80,380),(520,380),(430,180),(170,180)])],(140,30,140))
        cv2.rectangle(frame,(40,330),(145,410),(0,0,0),-1)
        for source,expected in [(frame,np.array((80,380))),
                                (cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE),np.array((99,80)))]:
            for scene in ('start-facing','front-facing','general'):
                visible,inferred,supports = self.detect(source,scene)
                self.assertEqual(len(inferred),1,(scene,inferred))
                self.assertLess(np.linalg.norm(inferred[0][0]-expected),3.)
                self.assertTrue(all(np.linalg.norm(p-expected)>12 for p,_ in visible))
                self.assertIn(tuple(inferred[0][0]),supports)

    def test_crop_does_not_create_corner_or_compensation(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(-100,100),(200,300),(140,30,140),-1)
        for scene in ('start-facing','front-facing','general'):
            visible,inferred,_ = self.detect(frame,scene)
            self.assertEqual({tuple(p) for p,_ in visible},{(200,100),(200,300)})
            self.assertFalse(inferred)

    def test_straight_edge_gap_is_not_a_corner(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,100),(500,400),(140,30,140),-1)
        cv2.rectangle(frame,(280,360),(320,430),(0,0,0),-1)
        for scene in ('start-facing','front-facing','general'):
            visible,inferred,_ = self.detect(frame,scene)
            self.assertEqual({tuple(p) for p,_ in visible},{(100,100),(500,100),(500,400),(100,400)})
            self.assertFalse(inferred)

    def test_separate_closed_green_regions_do_not_infer_between_them(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,100),(200,200),(140,30,140),-1)
        cv2.rectangle(frame,(240,240),(340,340),(140,30,140),-1)
        for scene in ('start-facing','front-facing','general'):
            visible,inferred,_ = self.detect(frame,scene)
            self.assertEqual(len(visible),8)
            self.assertFalse(inferred)


class CameraPipelineTests(unittest.TestCase):
    def test_live_main_with_synthetic_camera_and_pose_file(self):
        self.run_pipeline(Pose2D(2200,200,180), 2, 179, 'synthetic_pipeline.png')

    def test_front_detector_from_north_and_south(self):
        self.run_pipeline(Pose2D(1500,2100,90), 2, 80, 'synthetic_front_1.png')
        self.run_pipeline(Pose2D(1500,900,-90), 2, 80, 'synthetic_front_2.png')

    def test_general_main_uses_relaxed_effective_boundary_detector(self):
        frame = np.full((480,640,3),235,np.uint8)
        cv2.fillPoly(frame,[np.array([(50,380),(590,380),(585,340),(200,340)])],(140,30,140))
        with patch(__name__+'.synthetic_frame',return_value=frame):
            results = self.run_pipeline(Pose2D(800,2100,45),4,179,'synthetic_general.png')
        self.assertEqual(len(results),4)

    def run_pipeline(self, pose_value, expected_min_corners, expected_white_s, filename):
        frame=synthetic_frame()
        class Camera:
            def read(self): return True,frame.copy()
            def get(self,_): return 30
            def release(self): pass
        trackbars={}
        previews=[]
        captured=[]
        real_compose=vision.compose_preview
        def compose(*args, **kwargs):
            captured.append([point for _,visible,hidden in args[2] for point in visible+hidden])
            self.assertEqual(args[3],[])  # Recognition-only mode cannot reuse legacy map IDs.
            return real_compose(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,pose_value)
            with patch('sys.argv',['vision','--parameters','--scene','auto','--pose-file',str(pose),
                                  '--zones-file',str(Path(__file__).with_name('field_zones.json'))]), \
                 patch.object(vision,'open_camera',return_value=Camera()), \
                 patch.object(vision,'compose_preview',side_effect=compose), \
                 patch.object(vision,'preview_dimensions',return_value=(1440,792)), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda name,win,value,*_:trackbars.update({name:value})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda name,win:trackbars[name]), \
                 patch.object(cv2,'setTrackbarPos',side_effect=lambda name,win,value:trackbars.update({name:value})), \
                 patch.object(cv2,'getWindowProperty',return_value=1), \
                 patch.object(cv2,'imshow',side_effect=lambda win,image:previews.append(image.copy()) if win.startswith('Camera') else None), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
        self.assertGreaterEqual(len(captured[0]),expected_min_corners)
        self.assertEqual(trackbars['white S max'],expected_white_s)
        self.assertEqual(previews[0].shape,(792,1440,3))
        output=Path(__file__).resolve().parent/'tmp'/'nearest_visual_pose_baseline'
        output.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(output/filename),previews[0])
        return captured[0]


if __name__ == '__main__':
    unittest.main()
