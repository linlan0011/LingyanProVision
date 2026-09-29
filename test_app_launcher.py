from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_app
from pose_io import Pose2D, write_pose_atomic


class LauncherTests(unittest.TestCase):
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
from safety_identity import load_templates, active_templates, associate_corners, CornerObservation
pose_path, seen_path, stop_path = map(Path,sys.argv[1:])
reader=PoseFileReader(pose_path)
model=load_templates()
zones=read_zones('field_zones.json')
previous=None
while not stop_path.exists():
    pose=reader.poll()
    if pose is not None and pose != previous:
        templates,_=active_templates(pose,zones,model)
        result=associate_corners([CornerObservation((430,365),False,(180,135))],templates,model)[0]
        with seen_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'x':pose.x_mm,'corner':result.corner_id})+'\\n')
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
            self.assertIn({'x':1500.0,'corner':'1.3'},rows)
            self.assertIn({'x':2200.0,'corner':'2.2'},rows)
            self.assertEqual(proc_holder[0].poll(),0)


if __name__=='__main__':
    unittest.main()
