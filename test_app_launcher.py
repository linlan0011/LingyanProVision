from pathlib import Path
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
import io
from unittest.mock import patch

import run_app
from pose_io import Pose2D, write_pose_atomic


class LauncherTests(unittest.TestCase):
    def test_invalid_calibration_fails_before_opening_camera_or_ui(self):
        with patch("corner_ranging.CornerRanger.from_camera",side_effect=ValueError("mismatched intrinsics")), \
                patch.object(run_app.tk,"Tk") as root, redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exited:
                run_app.main(["--camera","lingyan"])
            self.assertEqual(exited.exception.code,2)
            self.assertIn("--no-ranging",error.getvalue())
            root.assert_not_called()

    def test_default_command_shares_absolute_files_and_auto_scene(self):
        args = run_app.build_parser().parse_args([])
        command = run_app.vision_command(args, Path('C:/session/stop'))
        self.assertEqual(command[command.index('--scene')+1], 'auto')
        self.assertTrue(Path(command[command.index('--pose-file')+1]).is_absolute())
        self.assertEqual(command[0],sys.executable)

    def test_real_child_receives_live_pose_and_stops_with_ui(self):
        # Replace only the physical camera with a subprocess that uses the real
        # pose reader and identity rules. The GUI, launcher and IPC are real.
        child_code = '''
from pathlib import Path
import json, sys, time
from pose_io import PoseFileReader
from field_zones import read_zones
from safety_identity import load_templates, active_templates
pose_path, seen_path, stop_path = map(Path,sys.argv[1:])
reader=PoseFileReader(pose_path)
model=load_templates()
zones=read_zones('field_zones.json')
previous=None
while not stop_path.exists():
    pose=reader.poll()
    if pose is not None and pose != previous:
        templates,_=active_templates(pose,zones,model)
        safety_ids=sorted({t.safety_id for t in templates})
        with seen_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'x':pose.x_mm,'safety_ids':safety_ids})+'\\n')
        previous=pose
    time.sleep(.02)
'''
        with tempfile.TemporaryDirectory() as directory:
            pose=Path(directory)/'pose.json'
            seen=Path(directory)/'seen.jsonl'
            write_pose_atomic(pose,Pose2D(1500,2100,90))
            real_app=run_app.PoseMapApp
            app_holder=[]
            proc_holder=[]
            real_popen=run_app.subprocess.Popen
            def capture_process(*args,**kwargs):
                process=real_popen(*args,**kwargs)
                proc_holder.append(process)
                return process
            def create_ui(root,args):
                root.withdraw()
                app=real_app(root,args)
                app_holder.append(app)
                def change_pose():
                    app._set_pose(Pose2D(2200,200,180),'test live update')
                def close():
                    root.tk.call(root.protocol('WM_DELETE_WINDOW'))
                root.after(600,change_pose)
                root.after(1400,close)
                return app
            with patch.object(run_app,'PoseMapApp',side_effect=create_ui), \
                 patch.object(run_app,'vision_command',side_effect=lambda args,stop:
                              [sys.executable,'-u','-c',child_code,str(pose),str(seen),str(stop)]), \
                 patch.object(run_app.subprocess,'Popen',side_effect=capture_process):
                self.assertEqual(run_app.main(['--pose-file',str(pose)]),0)
            rows=[json.loads(line) for line in seen.read_text(encoding='utf-8').splitlines()]
            self.assertIn({'x':1500.0,'safety_ids':[1]},rows)
            self.assertIn({'x':2200.0,'safety_ids':[2]},rows)
            self.assertEqual(proc_holder[0].poll(),0)

    def test_ranging_feedback_round_trip_does_not_correct_ui_pose(self):
        import pose_association
        child_code = '''
from pathlib import Path
import sys, time
from pose_io import PoseFileReader
from pose_association import AssociationWriter, association_payload
from test_corner_ranging import fixture
from safety_identity import CornerObservation, CornerIdentity
pose_path, feedback_path, stop_path = map(Path,sys.argv[1:])
reader=PoseFileReader(pose_path)
writer=AssociationWriter(feedback_path)
ranger=fixture()
while not stop_path.exists():
    pose=reader.poll()
    if pose is not None:
        identity=CornerIdentity(CornerObservation((320.,360.),False,()),{'1.3'},'1.3')
        readings=ranger.measure([(None,[((320.,360.),90.)],[])],[identity])
        writer.write(association_payload('usb0bda3035_1080p',1,ranger,readings,pose,[identity]))
    time.sleep(.04)
'''
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pose_association, 'PROJECT', Path(directory)):
            pose_path = Path(directory)/'pose.json'
            write_pose_atomic(pose_path, Pose2D(1500,2100,90))
            output = pose_association.association_output_path('usb0bda3035_1080p')
            received = []
            actual_ui = run_app.PoseMapApp
            def create_ui(root,args):
                root.withdraw()
                app = actual_ui(root,args)
                actual_feedback = app.set_live_association
                def display(data):
                    if data is not None:
                        before = pose_path.read_bytes()
                        prior = app.pose
                        actual_feedback(data)
                        self.assertEqual(app.pose, prior)
                        self.assertEqual(pose_path.read_bytes(), before)
                        self.assertIn('1.3', app.live_identity_var.get())
                        received.append(prior.x_mm)
                    else:
                        actual_feedback(data)
                app.set_live_association = display
                root.after(700, lambda: app._set_pose(Pose2D(2200,200,180),'test input update'))
                root.after(1800, lambda: root.tk.call(root.protocol('WM_DELETE_WINDOW')))
                return app
            with patch.object(run_app,'PoseMapApp',side_effect=create_ui), \
                    patch.object(run_app,'vision_command',side_effect=lambda args,stop:
                        [sys.executable,'-u','-c',child_code,str(pose_path),str(output),str(stop)]):
                self.assertEqual(run_app.main(['--camera','usb0bda3035_1080p',
                    '--ranging','--target-z-mm','0','--pose-file',str(pose_path)]),0)
            self.assertIn(1500., received)
            self.assertIn(2200., received)


if __name__=='__main__':
    unittest.main()
