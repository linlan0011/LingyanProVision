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
    active_templates, associate_corners, directed_rays, load_templates, map_corners)
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


class CameraPipelineTests(unittest.TestCase):
    def test_live_main_with_synthetic_camera_and_pose_file(self):
        self.run_pipeline(Pose2D(2200,200,180), {'2.2','2.3'}, 179, 'synthetic_pipeline.png')

    def test_front_detector_from_north_and_south(self):
        self.run_pipeline(Pose2D(1500,2100,90), {'1.3','1.4'}, 80, 'synthetic_front_1.png')
        self.run_pipeline(Pose2D(1500,900,-90), {'2.1','2.2'}, 80, 'synthetic_front_2.png')

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
        def identify(*args):
            result=real_associate(*args)
            captured.append(result)
            return result
        with tempfile.TemporaryDirectory() as directory:
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,pose_value)
            with patch('sys.argv',['vision','--scene','auto','--pose-file',str(pose),
                                  '--zones-file',str(Path(__file__).with_name('field_zones.json'))]), \
                 patch.object(vision,'open_camera',return_value=Camera()), \
                 patch.object(vision,'identify_frame_corners',side_effect=identify), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'createTrackbar',side_effect=lambda name,win,value,*_:trackbars.update({name:value})), \
                 patch.object(cv2,'getTrackbarPos',side_effect=lambda name,win:trackbars[name]), \
                 patch.object(cv2,'setTrackbarPos',side_effect=lambda name,win,value:trackbars.update({name:value})), \
                 patch.object(cv2,'getWindowProperty',return_value=1), \
                 patch.object(cv2,'imshow',side_effect=lambda win,image:previews.append(image.copy())), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(vision.main(),0)
        ids={r.corner_id for r in captured[0] if r.corner_id}
        self.assertTrue(expected_ids <= ids, (expected_ids, ids))
        self.assertEqual(trackbars['white S max'],expected_white_s)
        self.assertEqual(previews[0].shape,(566,1280,3))
        output=Path(__file__).resolve().parent/'照片'
        output.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(output/filename),previews[0])


if __name__ == '__main__':
    unittest.main()
