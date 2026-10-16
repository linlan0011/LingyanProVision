from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import cv2
import numpy as np

import purple_camera_segmentation as vision
import preview_display
from pose_io import write_pose_atomic
from test_nearest_corner_pose import camera, scene, pixel_key, measured


class NearestPosePipelineTests(unittest.TestCase):
    def test_actual_loop_publishes_visual_heading_and_keeps_all_markers_without_layers(self):
        self.assert_visual_pipeline('front-facing')

    def test_side_loop_uses_shared_supports_and_publishes_visual_pose(self):
        self.assert_visual_pipeline('start-facing')

    def assert_visual_pipeline(self, scene_name):
        ranger = camera()
        actual,reading,supports = scene(ranger)
        from dataclasses import replace
        prior = replace(actual,heading_deg=86.)
        nearest_pixel = pixel_key(reading.pixel)
        far = measured(ranger,(1400.,0.),True,geometry=True)
        far_pixel = pixel_key(far.pixel)
        frame = np.full((480,640,3),235,np.uint8)
        cv2.rectangle(frame,(100,280),(550,450),(140,30,140),-1)

        class Camera:
            def read(self): return True,frame.copy()
            def get(self,prop):
                return {cv2.CAP_PROP_FRAME_WIDTH:640.,cv2.CAP_PROP_FRAME_HEIGHT:480.,
                        cv2.CAP_PROP_FPS:30.}.get(prop,0.)
            def release(self): pass

        def detector(*args,**kwargs):
            self.assertIsNotNone(kwargs['white_core_mask'])
            self.assertTrue(args[3].infer_hidden_corners)
            kwargs['corner_supports'].update(supports)
            return [],[(np.array(nearest_pixel),90.),(np.array(far_pixel),90.)]

        with tempfile.TemporaryDirectory() as directory,patch('pose_association.PROJECT',Path(directory)):
            pose=Path(directory)/'pose.json'
            write_pose_atomic(pose,prior)
            before=pose.read_bytes()
            output=Path(directory)/'output/association/lingyan/latest.json'
            argv=['vision','--scene',scene_name,'--ranging','--target-z-mm','0',
                  '--pose-file',str(pose),'--association-output',str(output)]
            with patch('sys.argv',argv),patch.object(vision,'open_camera',return_value=Camera()), \
                 patch('corner_ranging.CornerRanger.from_camera',return_value=ranger), \
                 patch.object(vision,'find_effective_boundary_corners',side_effect=detector), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'imshow'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(preview_display,'_corner_labels',wraps=preview_display._corner_labels) as labels:
                self.assertEqual(vision.main(),0)
                self.assertEqual(sum(len(hidden) for _,_,hidden in labels.call_args.args[1]),2)
            data=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(pose.read_bytes(),before)
        self.assertTrue(data['pose_corrected'])
        self.assertEqual(data['confirmed_corner_ids'],['1.3'])
        self.assertAlmostEqual(data['corrected_pose']['heading_deg'],90.,places=6)
        self.assertNotAlmostEqual(data['corrected_pose']['heading_deg'],prior.heading_deg)
        self.assertNotIn('bottom_edge_test',data)
        self.assertNotIn('ground_association',data)
        self.assertEqual(len(data['corners']),2)
        self.assertEqual(data['visual_pose']['candidate_count'],2)
        self.assertEqual(data['visual_pose']['selected_corner']['pixel'],list(nearest_pixel))
        self.assertEqual(data['visual_pose']['candidate_view']['prior_radius_mm'],200.)
        self.assertEqual(data['visual_pose']['candidate_view']['possible_safety_ids'],[1])

    def test_launcher_forwards_all_new_angle_controls_and_disable_alias(self):
        from run_app import build_parser,vision_command
        args=build_parser().parse_args(['--identity-angle-tol-deg','12','--orthogonal-angle-tol-deg','10',
                                      '--heading-consistency-deg','8','--no-visual-pose'])
        command=vision_command(args,Path('stop'))
        self.assertIn('--no-ground-association',command)
        for flag,value in (('--identity-angle-tol-deg','12'),('--orthogonal-angle-tol-deg','10'),
                           ('--heading-consistency-deg','8')):
            self.assertEqual(command[command.index(flag)+1],value)


if __name__ == '__main__':
    unittest.main()
