from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

import camera_profiles as profiles
from calibration.ranging_report import apply_analysis_range, analyse_records, percentile95, render_report, save_reports
from calibration.validate_ranging import main, read_records
from test_corner_ranging import fixture


def calibration():
    ranger = fixture()
    return ({"camera_matrix": ranger.matrix.tolist(), "distortion_coefficients": ranger.distortion.tolist(),
             "image_width":640,"image_height":480,"model":"opencv_pinhole"},
            {"rotation_camera_to_body":ranger.rotation.tolist(),"camera_position_body_mm":ranger.center.tolist()})


def record(x=1500,y=0,u=320,v=360,z=0):
    return dict(x_mm=x,y_mm=y,z_mm=z,u_px=u,v_px=v)


class GeometryReportTests(unittest.TestCase):
    def test_55_points_inclusive_adjustable_range_keeps_full_report(self):
        intrinsics,mount = calibration()
        xs = [300+10*i for i in range(19)]+[500]+[600+10*i for i in range(35)]
        data = analyse_records([record(x) for x in xs],intrinsics,mount)
        original_summary = dict(data["summary"])
        apply_analysis_range(data,x_max_mm=500)
        self.assertEqual(len(data["points"]),55)
        self.assertEqual(data["points"][-1]["point_id"],"P055")
        self.assertEqual(data["summary"],original_summary)
        self.assertEqual(data["range_summary"]["count"],20)
        self.assertIn("P020",data["analysis_range"]["point_ids"])
        self.assertNotIn("P021",data["analysis_range"]["point_ids"])
        self.assertAlmostEqual(data["range_summary"]["mean_horizontal_error_mm"],
                               sum(1500-x for x in xs[:20])/20)
        text = render_report(data)
        self.assertIn("X≤500 mm",text)
        self.assertIn("入选=20/55",text)
        apply_analysis_range(data,x_min_mm=500,x_max_mm=600)
        self.assertEqual(data["analysis_range"]["point_ids"],["P020","P021"])
        apply_analysis_range(data)
        self.assertEqual(data["range_summary"]["count"],55)
        self.assertFalse(data["analysis_range"]["active"])

    def test_range_uses_reference_not_prediction_and_handles_empty_invalid(self):
        intrinsics,mount = calibration()
        data = analyse_records([record(400,-800),record(500,0,u=-1),record(600,100)],intrinsics,mount)
        apply_analysis_range(data,x_max_mm=500,y_max_mm=0,r_min_mm=600)
        self.assertEqual(data["analysis_range"]["point_ids"],["P001"])
        self.assertGreater(data["points"][0]["predicted_body_mm"][0],500)
        apply_analysis_range(data,x_min_mm=500,x_max_mm=500)
        self.assertEqual(data["range_summary"]["count"],1)
        self.assertEqual(data["range_summary"]["invalid_count"],1)
        self.assertIsNone(data["range_summary"]["mean_horizontal_error_mm"])
        apply_analysis_range(data,x_max_mm=100)
        self.assertEqual(data["range_summary"]["count"],0)
        self.assertIsNone(data["range_summary"]["rmse_horizontal_mm"])
        self.assertIn("本范围没有采集点",render_report(data))
        before = json.dumps(data,sort_keys=True)
        for bounds in ({"x_min_mm":501,"x_max_mm":500},{"y_min_mm":2,"y_max_mm":1},
                       {"r_min_mm":-1},{"x_max_mm":float("nan")},{"r_max_mm":float("inf")}):
            with self.assertRaises(ValueError):
                apply_analysis_range(data,**bounds)
            self.assertEqual(json.dumps(data,sort_keys=True),before)

    def test_per_point_metrics_statistics_and_invalid_denominator(self):
        intrinsics,mount = calibration()
        data = analyse_records([record(),record(1497,-4),record(1490),record(u=-1)],
                               intrinsics,mount,error_limit_mm=6)
        s = data["summary"]
        self.assertEqual((s["count"],s["valid_count"],s["invalid_count"]),(4,3,1))
        self.assertEqual((s["pass_count"],s["fail_count"]),(2,1))
        self.assertAlmostEqual(s["mean_horizontal_error_mm"],5)
        self.assertAlmostEqual(s["median_horizontal_error_mm"],5)
        self.assertAlmostEqual(s["rmse_horizontal_mm"],math.sqrt(125/3))
        self.assertAlmostEqual(s["p95_horizontal_error_mm"],10)
        r = data["points"][1]
        self.assertAlmostEqual(r["x_error_mm"],3)
        self.assertAlmostEqual(r["y_error_mm"],4)
        self.assertAlmostEqual(r["distance_error_mm"],1500-math.hypot(1497,-4))
        self.assertAlmostEqual(r["bearing_error_deg"],-math.degrees(math.atan2(-4,1497)))
        self.assertIsNone(data["points"][3]["horizontal_error_mm"])
        self.assertIsNone(data["points"][3]["passed"])
        self.assertEqual(sum(g["count"] for g in data["groups_by_x"]),4)
        self.assertEqual(sum(g["count"] for g in data["distance_groups"]),4)
        text = render_report(data)
        for phrase in ("P001","P004","RMSE","P95","逐点误差","按前向 X 分组","pixel_outside_image"):
            self.assertIn(phrase,text)
        json.dumps(data,allow_nan=False)

    def test_geometry_and_runtime_limits_are_separate(self):
        intrinsics,mount = calibration()
        data = analyse_records([record(4500,v=280),record(18000,v=250),record(v=240)],intrinsics,mount)
        far,near,invalid = data["points"]
        self.assertTrue(far["valid"])
        self.assertFalse(far["runtime_usable"])
        self.assertEqual(far["runtime_reason"],"beyond_distance_limit")
        self.assertTrue(near["valid"])
        self.assertEqual(near["runtime_reason"],"near_horizon")
        self.assertFalse(invalid["valid"])
        self.assertEqual(data["runtime_summary"]["count"],0)
        self.assertIsNone(data["runtime_summary"]["mean_horizontal_error_mm"])

    def test_origin_undefined_angles_and_all_invalid(self):
        intrinsics,mount = calibration()
        data = analyse_records([record(0,0)],intrinsics,mount)
        self.assertIsNone(data["points"][0]["bearing_error_deg"])
        self.assertIsNone(data["points"][0]["relative_distance_error_percent"])
        invalid = analyse_records([record(v=100)],intrinsics,mount)
        self.assertEqual(invalid["summary"]["valid_count"],0)
        self.assertIsNone(invalid["mean_horizontal_error_mm"])
        self.assertIn("--",render_report(invalid))
        json.dumps(invalid,allow_nan=False)

    def test_angle_wrap_at_180_and_known_height(self):
        from calibration.lens_models import project_points
        import cv2
        intrinsics,mount = calibration()
        rotation = np.diag([-1.,-1.,1.])@np.array(mount["rotation_camera_to_body"])
        mount["rotation_camera_to_body"] = rotation.tolist()
        predicted = np.array([[-1500.,10.,100.]])
        rvec,_ = cv2.Rodrigues(rotation.T)
        pixel = project_points(predicted,rvec,-rotation.T@np.array(mount["camera_position_body_mm"]),
            np.array(intrinsics["camera_matrix"]),np.zeros(5),intrinsics["model"])[0]
        data = analyse_records([record(-1500,-10,*pixel,z=100)],intrinsics,mount)
        r = data["points"][0]
        self.assertTrue(r["valid"])
        self.assertLess(abs(r["bearing_error_deg"]),1.)
        self.assertAlmostEqual(r["y_error_mm"],20)
        self.assertAlmostEqual(r["predicted_body_mm"][2],100)

    def test_input_and_limit_validation(self):
        intrinsics,mount = calibration()
        for rows,options in (([],{}),([record(x=float("nan"))],{}),([{}],{}),
                             ([record()],{"bin_mm":0}),([record()],{"error_limit_mm":-1}),
                             ([record()],{"max_distance_mm":0})):
            with self.assertRaises(ValueError):
                analyse_records(rows,intrinsics,mount,**options)
        self.assertEqual(percentile95(list(range(1,21))),19)
        self.assertEqual(percentile95([1,2,3,4,5]),5)

    def test_fisheye_projection_and_duplicate_measurements_are_preserved(self):
        from calibration.lens_models import project_points
        import cv2
        intrinsics,mount = calibration()
        intrinsics["model"] = "opencv_fisheye"
        intrinsics["distortion_coefficients"] = [0.,0.,0.,0.]
        R = np.array(mount["rotation_camera_to_body"])
        rvec,_ = cv2.Rodrigues(R.T)
        pixel = project_points(np.array([[1500.,0.,0.]]),rvec,-R.T@np.array(mount["camera_position_body_mm"]),
                               np.array(intrinsics["camera_matrix"]),np.zeros(4),intrinsics["model"])[0]
        data = analyse_records([record(u=pixel[0],v=pixel[1])]*2,intrinsics,mount)
        self.assertEqual(data["summary"]["count"],2)
        self.assertEqual(data["summary"]["unique_position_count"],1)
        self.assertLess(data["max_horizontal_error_mm"],1e-6)

    def test_changed_calibration_and_bad_input_sources_are_rejected(self):
        from calibration import ranging_report
        intrinsics,mount = calibration()
        with tempfile.TemporaryDirectory() as directory, patch.object(profiles,"PROJECT",Path(directory)):
            intrinsic_path = Path(directory)/"intrinsics.json"
            mount_path = Path(directory)/"mount_pose.json"
            intrinsic_path.write_text(json.dumps(intrinsics),encoding="utf-8")
            mount["intrinsics_sha256"] = profiles.file_sha256(intrinsic_path)
            mount_path.write_text(json.dumps(mount),encoding="utf-8")
            analyse = ranging_report.analyse_records
            def changed(*args,**kwargs):
                result = analyse(*args,**kwargs)
                mount_path.write_text(json.dumps(mount)+"\n",encoding="utf-8")
                return result
            with patch.object(ranging_report,"load_intrinsics",return_value=(intrinsic_path,intrinsics)), \
                    patch.object(ranging_report,"load_mount",return_value=(mount_path,mount)), \
                    patch.object(ranging_report,"analyse_records",side_effect=changed):
                with self.assertRaisesRegex(ValueError,"changed during validation"):
                    ranging_report.build_camera_report("lingyan",[record()])
            csv_path = profiles.data_dir("lingyan")/"bad.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("x_mm,u_px\n1,2\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"must contain"):
                read_records([csv_path],"lingyan")
            csv_path.write_text("x_mm,y_mm,z_mm,u_px,v_px\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"no points"):
                read_records([csv_path],"lingyan")
            other = Path(directory)/"other.png"
            other.write_bytes(b"frame")
            csv_path.write_text("x_mm,y_mm,z_mm,u_px,v_px,frame_file\n1500,0,0,320,360,"+str(other)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"another camera"):
                read_records([csv_path],"lingyan")

    def test_cli_multiple_csvs_backup_isolation_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(profiles,"PROJECT",Path(directory)), \
                patch.object(profiles,"REGISTRY",Path(directory)/"camera_profiles.json"):
            profiles.register_profile("testcam",profiles.CameraProfile("test",640,480,"none"))
            intrinsics,mount = calibration()
            root = profiles.results_dir("testcam")
            root.mkdir(parents=True)
            intrinsic_path = root/"intrinsics.json"
            intrinsics["camera_id"] = "testcam"
            intrinsic_path.write_text(json.dumps(intrinsics),encoding="utf-8")
            mount["camera_id"] = "testcam"
            mount["model"] = "opencv_pinhole"
            mount["intrinsics_sha256"] = profiles.file_sha256(intrinsic_path)
            mount_path = root/"mount_pose.json"
            mount_path.write_text(json.dumps(mount),encoding="utf-8")
            a,b = profiles.data_dir("testcam")/"a.csv",profiles.data_dir("testcam")/"b.csv"
            a.parent.mkdir(parents=True)
            a.write_text("x_mm,y_mm,z_mm,u_px,v_px\n1500,0,0,320,360\n1497,-4,0,320,360\n",encoding="utf-8-sig")
            b.write_text("x_mm,y_mm,z_mm,u_px,v_px\n1490,0,0,320,360\n",encoding="utf-8")
            before = {p:p.read_bytes() for p in (a,b,intrinsic_path,mount_path)}
            out = root/"combined.json"
            for _ in range(2):
                with redirect_stdout(io.StringIO()) as terminal:
                    self.assertEqual(main(["--camera","testcam","--points",str(a),str(b),
                                           "--output",str(out),"--max-error-mm","6","--x-max-mm","1497"]),0)
                self.assertIn("P003",terminal.getvalue())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["count"],3)
            self.assertEqual(data["range_summary"]["count"],2)
            self.assertEqual(data["analysis_range"]["x_max_mm"],1497)
            self.assertEqual(data["points"][2]["source_file"],str(b.resolve()))
            self.assertEqual(len(data["input_files"]),2)
            self.assertEqual(data["intrinsics_sha256"],profiles.file_sha256(intrinsic_path))
            self.assertTrue(out.with_suffix(".txt").is_file())
            self.assertEqual(len(list(root.glob("combined.backup_*.json"))),1)
            self.assertEqual(len(list(root.glob("combined.backup_*.txt"))),1)
            self.assertEqual({p:p.read_bytes() for p in before},before)
            for forbidden in (root/"intrinsics.json",profiles.results_dir("lingyan")/"bad.json"):
                with self.assertRaises(ValueError):
                    save_reports(forbidden,data)
            with self.assertRaises(ValueError):
                read_records([a,a],"testcam")

    def test_collector_report_reads_all_current_points_without_mutation(self):
        from calibration.collect_mount_points import Collector,Point
        intrinsics,mount = calibration()
        expected = analyse_records([record(),record(1497,-4)],intrinsics,mount)
        expected["camera_id"] = "lingyan"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"intrinsics.json"
            path.write_text(json.dumps(intrinsics),encoding="utf-8")
            root = tk.Tk()
            root.withdraw()
            app = Collector(root,purpose="validation",intrinsics=path,sessions=Path(directory)/"sessions")
            app.points = [Point(1500,0,0,320,360,"frame.png"),Point(1497,-4,0,320,360,"frame.png")]
            before = [asdict(p) for p in app.points]
            try:
                with patch("calibration.collect_mount_points.load_intrinsics",return_value=(path,intrinsics)), \
                        patch("calibration.ranging_report.build_camera_report",return_value=expected) as build, \
                        patch("calibration.collect_mount_points.messagebox.showerror") as errors:
                    app.show_range_report()
                    self.assertFalse(errors.called)
                    self.assertEqual(len(build.call_args.args[1]),2)
                    window = [w for w in root.winfo_children() if isinstance(w,tk.Toplevel)][0]
                    def find_texts(widget):
                        return ([widget] if isinstance(widget,tk.Text) else [])+[
                            text for child in widget.winfo_children() for text in find_texts(child)]
                    texts = find_texts(window)
                    self.assertIn("P002",texts[0].get("1.0","end"))
                    window.analysis_range_vars["x_max_mm"].set("1497")
                    self.assertTrue(window.refresh_analysis_range())
                    self.assertEqual(expected["range_summary"]["count"],1)
                    self.assertIn("X≤1497 mm",texts[0].get("1.0","end"))
                    window.analysis_range_vars["x_max_mm"].set("")
                    self.assertTrue(window.refresh_analysis_range())
                    self.assertEqual(expected["range_summary"]["count"],2)
                    window.analysis_range_vars["x_max_mm"].set("invalid")
                    self.assertFalse(window.refresh_analysis_range())
                    self.assertTrue(errors.called)
                    self.assertEqual([asdict(p) for p in app.points],before)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
