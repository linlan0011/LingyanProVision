"""Regression checks for frame-bound, read-only map feedback and camera exit."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ground_corner_map import draw_visual_feedback
from pose_io import Pose2D
import run_app


class RecordingCanvas:
    def __init__(self):
        self.draws = []

    def create_line(self, *args, **kwargs):
        self.draws.append(('line', args, kwargs))

    def create_oval(self, *args, **kwargs):
        self.draws.append(('oval', args, kwargs))

    def create_text(self, *args, **kwargs):
        self.draws.append(('text', args, kwargs))


class IdentityTransform:
    x_min, x_max, y_min, y_max = -10000., 10000., -10000., 10000.

    @staticmethod
    def world_to_canvas(x, y):
        return x, y



def report():
    pose=dict(x_mm=1500.,y_mm=2100.,heading_deg=90.)
    return dict(frame_number=417,created_at_unix=100.,confirmed_corner_ids=['1.3'],
                pose_corrected=True,corrected_pose=pose.copy(),
                simulated_prior=dict(x_mm=1500.,y_mm=2100.,heading_deg=86.),
                visual_pose=dict(schema='safety.nearest_visual_pose.v1',status='corrected',
                    pose_corrected=True,heading_source='visual_ground_edges',corner_id='1.3',
                    corrected_pose=pose.copy(),selected_corner=dict(inferred=False,range_mm=630.,
                        pixel=[320,344],body_xy_mm=[540.,-330.]),
                    candidate_view=dict(prior_radius_mm=200.,max_distance_mm=1000.,
                        possible_safety_ids=[1],central_footprint_map_mm=[],possible_footprint_map_mm=[],
                        landmarks=[dict(corner_id='1.3',map_xy_mm=[1830.,2640.],possibly_visible=True)])))


class GroundFeedbackTests(unittest.TestCase):
    def draw(self,data):
        canvas=RecordingCanvas()
        with patch('ground_corner_map.time.time',return_value=102.5):
            summary=draw_visual_feedback(canvas,IdentityTransform(),data,Pose2D(1500.,2100.,86.))
        arrows=[draw for draw in canvas.draws if draw[0]=='line' and draw[2].get('arrow')=='last']
        return '\n'.join(summary),arrows

    def test_visual_arrow_uses_visual_heading_and_displays_both_poses(self):
        data=report()
        before=deepcopy(data)
        summary,arrows=self.draw(data)
        self.assertEqual(len(arrows),1)
        x,y,ex,ey=arrows[0][1]
        self.assertAlmostEqual(ex,x,places=6)
        self.assertAlmostEqual(ey-y,64.,places=6)
        self.assertIn('惯导输入：X=1500.0 Y=2100.0 θ=86.00°',summary)
        self.assertIn('视觉校正：X=1500.0 Y=2100.0 θ=90.00°',summary)
        self.assertIn('Δθ=+4.00°',summary)
        self.assertIn('回传帧：417；反馈年龄：2.5 s（非拍摄时间）',summary)
        self.assertEqual(data,before)

    def test_unconfirmed_status_never_draws_stale_pose(self):
        for status in ('identity_no_match','ambiguous_safety_zone','two_support_edges_required'):
            data=report()
            data['visual_pose']['status']=status
            summary,arrows=self.draw(data)
            self.assertFalse(arrows)
            self.assertIn('本帧未更新视觉位姿',summary)

    def test_mismatched_ids_or_pose_fields_never_draw(self):
        for ids in ([],['1.4'],['1.3','1.4']):
            data=report();data['confirmed_corner_ids']=ids
            self.assertFalse(self.draw(data)[1])
        data=report();data['corrected_pose']['heading_deg']=86.
        self.assertFalse(self.draw(data)[1])
        data=report();data['pose_corrected']=False
        self.assertFalse(self.draw(data)[1])

    def test_invalid_visual_pose_is_not_drawn(self):
        for value in (None,[],3.,dict(x_mm=float('nan'),y_mm=2100.,heading_deg=90.),
                      dict(x_mm=1500.,y_mm=2100.,heading_deg='90')):
            data=report();data['visual_pose']['corrected_pose']=value;data['corrected_pose']=value
            self.assertFalse(self.draw(data)[1])

    def test_unknown_feedback_age_and_unlimited_range_are_explicit(self):
        data=report();del data['created_at_unix']
        data['visual_pose']['candidate_view']['max_distance_mm']=None
        summary,_=self.draw(data)
        self.assertIn('反馈年龄：未知',summary)
        self.assertIn('定位量程：不限距离',summary)

    def test_real_map_keeps_input_and_updates_separate_visual_state_then_clears(self):
        import tkinter as tk
        from pose_map_gui import PoseMapApp,build_parser
        root=tk.Tk();root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pose.json'
            app=PoseMapApp(root,build_parser().parse_args(['--output',str(path),
                '--x','1500','--y','2100','--heading','86']))
            try:
                root.update();app.publish_pose()
                before=path.read_bytes();prior=app.pose
                app.set_live_association(report())
                self.assertEqual(app.corrected_pose,Pose2D(1500.,2100.,90.))
                self.assertTrue(app.canvas.find_withtag('corrected_pose'))
                self.assertTrue(app.canvas.find_withtag('pose_uncertainty'))
                self.assertEqual(app.pose,prior)
                self.assertEqual(path.read_bytes(),before)
                app.set_live_association(None)
                self.assertIsNone(app.corrected_pose)
                self.assertFalse(app.canvas.find_withtag('corrected_pose'))
                app.set_live_association(report())
                app.live_output_var.set(False)
                app._set_pose(Pose2D(1500.,2100.,85.),'test changed input')
                self.assertIsNone(app.corrected_pose)
                self.assertFalse(app.canvas.find_withtag('corrected_pose'))
            finally:
                app.close()

class LauncherFeedbackTests(unittest.TestCase):
    def test_default_command_restores_one_metre_localization_cap(self):
        args=run_app.build_parser().parse_args([])
        command=run_app.vision_command(args,Path('stop'))
        self.assertEqual(args.range_max_mm,1000.)
        self.assertEqual(command[command.index('--range-max-mm')+1],'1000.0')

    def test_unlimited_command_omits_range_argument_and_numeric_override_survives(self):
        args = run_app.build_parser().parse_args([])
        args.range_max_mm = None
        self.assertNotIn('--range-max-mm', run_app.vision_command(args, Path('stop')))
        args.range_max_mm = 1500.
        command = run_app.vision_command(args, Path('stop'))
        self.assertEqual(command[command.index('--range-max-mm') + 1], '1500.0')

    def test_camera_failure_clears_previous_feedback_before_error_dialog(self):
        root = Mock()
        app = Mock()
        process = Mock()
        process.poll.return_value = 7
        events = []
        app.set_live_association.side_effect = lambda data: events.append(('feedback', data))
        root.winfo_exists.return_value = False
        root.after.side_effect = lambda delay, callback: setattr(root, 'pending', callback)
        root.mainloop.side_effect = lambda: root.pending()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_app, 'PROJECT', Path(directory)), \
                patch.object(run_app.tk, 'Tk', return_value=root), \
                patch.object(run_app, 'PoseMapApp', return_value=app), \
                patch.object(run_app.subprocess, 'Popen', return_value=process), \
                patch.object(run_app, 'stop_vision'), \
                patch.object(run_app.messagebox, 'showerror',
                             side_effect=lambda *a, **k: events.append(('error', None))):
            code = run_app.main(['--no-ranging', '--pose-file', str(Path(directory)/'pose.json')])
        self.assertEqual(code, 7)
        self.assertEqual(events, [('feedback', None), ('error', None)])
        self.assertIn('旧识别已清空', app.status_var.set.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
