"""Regression checks for frame-bound, read-only map feedback and camera exit."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ground_corner_map import draw_ground_feedback
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
    return dict(frame_number=417, created_at_unix=100., confirmed_corner_ids=['1.3'],
                pose_corrected=False, ground_association=dict(
        config={'prior_radius_mm': 200.}, possible_footprint_map_mm=[],
        central_footprint_map_mm=[], observations=[dict(
            corner_id='1.3', map_center_mm=[1830., 2640.], map_radius_mm=230.,
            inferred=False, range_mm=630.)], landmarks=[dict(
            corner_id='1.3', map_xy_mm=[1830., 2640.], possibly_visible=True)],
        vision_vehicle_position_mm=[1500., 2100.], status='unique',
        max_distance_mm=None, possible_safety_ids=[1]))


class GroundFeedbackTests(unittest.TestCase):
    def draw(self, data):
        canvas = RecordingCanvas()
        with patch('ground_corner_map.time.time', return_value=102.5):
            summary = draw_ground_feedback(canvas, IdentityTransform(), data,
                                           Pose2D(1500., 2100., 90.))
        candidates = [draw for draw in canvas.draws if draw[2].get('fill') == '#00dddd']
        return '\n'.join(summary), candidates

    def test_matching_unique_ids_show_candidate_and_feedback_frame_age(self):
        data = report()
        before = deepcopy(data)
        summary, candidates = self.draw(data)
        self.assertEqual(len(candidates), 3)
        self.assertIn('回传帧：417；反馈年龄：2.5 s（非拍摄时间）', summary)
        self.assertIn('视觉反算候选：(1500.0, 2100.0) mm（未反写位姿）', summary)
        self.assertIn('量程：不限距离', summary)
        self.assertEqual(data, before)

    def test_no_confirmed_ids_rejects_stale_position_field(self):
        data = report()
        data['confirmed_corner_ids'] = []
        data['ground_association']['observations'][0]['corner_id'] = None
        summary, candidates = self.draw(data)
        self.assertEqual(candidates, [])
        self.assertIn('下沿编号：?（未确认）', summary)
        self.assertIn('未输出位置：本帧无确认编号', summary)
        self.assertNotIn('1500.0, 2100.0', summary)

    def test_non_unique_status_cannot_show_stale_position_even_with_ids(self):
        for status in ('no_match', 'ambiguous', 'waiting_for_ground_corners'):
            with self.subTest(status=status):
                data = report()
                data['ground_association']['status'] = status
                summary, candidates = self.draw(data)
                self.assertEqual(candidates, [])
                self.assertIn('未输出位置：本帧身份未确认', summary)
                self.assertNotIn('视觉反算候选：(', summary)

    def test_partial_unique_with_corresponding_ids_can_show_candidate(self):
        data = report()
        data['ground_association']['status'] = 'partial_unique'
        self.assertEqual(len(self.draw(data)[1]), 3)

    def test_missing_or_mismatched_root_ids_fail_closed(self):
        for ids in ([], ['1.4'], ['1.3', '1.4']):
            with self.subTest(ids=ids):
                data = report()
                data['confirmed_corner_ids'] = ids
                summary, candidates = self.draw(data)
                self.assertEqual(candidates, [])
                self.assertNotIn('视觉反算候选：(', summary)

    def test_numeric_range_and_missing_feedback_timestamp_are_explicit(self):
        data = report()
        data['ground_association']['max_distance_mm'] = 1000.
        del data['created_at_unix']
        summary, _ = self.draw(data)
        self.assertIn('量程：1000 mm', summary)
        self.assertIn('反馈年龄：未知', summary)
        data['ground_association']['status'] = 'calibration_unavailable'
        self.assertIn('等待地面测距', self.draw(data)[0])

    def test_none_position_does_not_leave_candidate_mark(self):
        data = report()
        data['ground_association']['vision_vehicle_position_mm'] = None
        summary, candidates = self.draw(data)
        self.assertEqual(candidates, [])
        self.assertIn('未输出位置：本帧无视觉反算结果', summary)

    def test_invalid_position_coordinates_are_not_drawn(self):
        for estimate in (3., [1500.], [float('nan'), 2100.], ['1500', 2100.]):
            with self.subTest(estimate=estimate):
                data = report()
                data['ground_association']['vision_vehicle_position_mm'] = estimate
                summary, candidates = self.draw(data)
                self.assertEqual(candidates, [])
                self.assertIn('未输出位置：本帧视觉反算坐标无效', summary)


class LauncherFeedbackTests(unittest.TestCase):
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
