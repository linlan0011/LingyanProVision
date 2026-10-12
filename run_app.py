"""One entry point for the pose selector and live safety-zone recognition."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox

from pose_io import read_pose
from pose_map_gui import PoseMapApp, build_parser as pose_parser
from camera_profiles import load_profiles
from corner_ranging import DEFAULT_MAX_DISTANCE_MM

PROJECT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Start pose UI and live camera together")
    parser.add_argument('--camera', choices=load_profiles(), default='usb0bda3035_1080p')
    parser.add_argument('--index', type=int)
    parser.add_argument('--ranging', action=argparse.BooleanOptionalAction, default=True,
                        help='full app enables calibrated corner ranging; --no-ranging disables it')
    parser.add_argument('--target-z-mm', type=float, default=0.,
                        help='known corner height in body coordinates; current ground corners default to 0 mm')
    parser.add_argument('--range-max-mm', type=float, default=DEFAULT_MAX_DISTANCE_MM)
    parser.add_argument('--pose-radius-mm', type=int, default=200)
    parser.add_argument('--ground-association', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--range-inferred', action='store_true')
    parser.add_argument('--association-output', type=Path)
    parser.add_argument('--pose-file', type=Path, default=PROJECT/'selected_pose.json')
    parser.add_argument('--zones-file', type=Path, default=PROJECT/'field_zones.json')
    parser.add_argument('--templates-file', type=Path, default=PROJECT/'safety_templates.json')
    return parser


def vision_command(args, stop_file):
    command = [sys.executable, '-u', str(PROJECT/'purple_camera_segmentation.py'),
            '--camera', args.camera, '--scene', 'auto',
            '--pose-file', str(args.pose_file.resolve()),
            '--zones-file', str(args.zones_file.resolve()),
            '--templates-file', str(args.templates_file.resolve()),
            '--stop-file', str(stop_file)]
    command.extend(('--pose-radius-mm', str(args.pose_radius_mm),
                    '--ground-association' if args.ground_association else '--no-ground-association'))
    if args.index is not None:
        command.extend(('--index', str(args.index)))
    if args.ranging:
        if args.target_z_mm is None:
            raise ValueError('实时测距需指定角点高度 --target-z-mm；地面点填 0')
        command.extend(('--ranging', '--target-z-mm', str(args.target_z_mm),
                        '--range-max-mm', str(args.range_max_mm)))
        if args.range_inferred:
            command.append('--range-inferred')
        from pose_association import association_output_path
        output = association_output_path(args.camera, args.association_output, args.pose_file)
        command.extend(('--association-output', str(output)))
    return command


def stop_vision(process, stop_file):
    if process is None or process.poll() is not None:
        return
    Path(stop_file).touch()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        # A blocked camera driver may not return to the normal stop check.
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main(argv=None):
    args = build_parser().parse_args(argv)
    association_output = None
    if args.ranging:
        from pose_association import association_output_path
        association_output = association_output_path(args.camera, args.association_output, args.pose_file)
    started_at = datetime.now().timestamp()
    ui_args = pose_parser().parse_args([
        '--output', str(args.pose_file.resolve()), '--zones', str(args.zones_file.resolve()),
        '--templates', str(args.templates_file.resolve()),
    ])
    # The pose shown on startup must be the same one consumed by the camera.
    try:
        pose = read_pose(args.pose_file)
        ui_args.x, ui_args.y, ui_args.heading = pose.x_mm, pose.y_mm, pose.heading_deg
    except (OSError, ValueError, KeyError, TypeError):
        pass
    logs = PROJECT/'output'/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs/f"vision_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"
    root = tk.Tk()
    root.geometry('1180x880')
    app = PoseMapApp(root, ui_args)
    root.title('安全区识别 — 实时选点与相机联动')
    process = None
    monitor_job = None
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix='lingyan_session_') as directory, log_path.open('w', encoding='utf-8') as log:
        stop_file = Path(directory)/'stop'

        def close_all():
            nonlocal monitor_job
            stop_file.touch()
            if monitor_job is not None:
                root.after_cancel(monitor_job)
                monitor_job = None
            app.close()

        def monitor():
            nonlocal monitor_job, exit_code
            monitor_job = None
            code = process.poll()
            if code is None:
                if association_output is not None:
                    from pose_association import read_live_association
                    app.set_live_association(read_live_association(
                        association_output, args.camera, started_at, app.pose))
                monitor_job = root.after(200, monitor)
            elif code == 0:
                close_all()
            else:
                exit_code = code
                app.status_var.set(f'相机程序退出（{code}）；日志：{log_path.name}')
                messagebox.showerror('相机未正常运行',
                    f'请检查相机连接、是否被其他程序占用，以及相机序号。\n\n日志：{log_path}', parent=root)

        root.protocol('WM_DELETE_WINDOW', close_all)
        try:
            app.publish_pose(automatic=True)
            environment = os.environ.copy()
            environment['PYTHONIOENCODING'] = 'utf-8'
            environment['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(vision_command(args, stop_file), cwd=PROJECT,
                stdout=log, stderr=subprocess.STDOUT, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            print(f'Pose UI + camera started. Camera log: {log_path}', flush=True)
            monitor_job = root.after(200, monitor)
            root.mainloop()
        except OSError as exc:
            exit_code = 1
            messagebox.showerror('启动失败', str(exc), parent=root)
        finally:
            stop_vision(process, stop_file)
            try:
                if root.winfo_exists():
                    close_all()
            except tk.TclError:
                pass
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
