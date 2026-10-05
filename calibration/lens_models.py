"""Keep OpenCV pinhole and fisheye distortion models separate."""
from __future__ import annotations

import cv2
import numpy as np


def calibrate(object_points, image_points, image_size, model):
    if model == "opencv_pinhole":
        return cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    if model != "opencv_fisheye":
        raise ValueError(f"unsupported calibration model: {model}")
    obj = [np.asarray(p, np.float64).reshape(1, -1, 3) for p in object_points]
    pixels = [np.asarray(p, np.float64).reshape(1, -1, 2) for p in image_points]
    return cv2.fisheye.calibrate(
        obj, pixels, image_size, np.eye(3, dtype=np.float64), np.zeros((4, 1), np.float64),
        flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )


def project_points(objects, rvec, tvec, matrix, distortion, model):
    if model == "opencv_pinhole":
        return cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    if model == "opencv_fisheye":
        return cv2.fisheye.projectPoints(
            np.asarray(objects, np.float64).reshape(1, -1, 3), rvec, tvec,
            matrix, np.asarray(distortion, np.float64).reshape(4, 1),
        )[0].reshape(-1, 2)
    raise ValueError(f"unsupported calibration model: {model}")


def undistort_points(pixels, matrix, distortion, model):
    points = np.asarray(pixels, np.float64).reshape(1, -1, 2)
    if model == "opencv_pinhole":
        result = cv2.undistortPoints(points, matrix, distortion)
    elif model == "opencv_fisheye":
        result = cv2.fisheye.undistortPoints(points, matrix, np.asarray(distortion).reshape(4, 1))
    else:
        raise ValueError(f"unsupported calibration model: {model}")
    return result.reshape(-1, 2)


def solve_pose(objects, pixels, matrix, distortion, model, *, rvec=None, tvec=None):
    if model == "opencv_pinhole":
        return cv2.solvePnP(objects, pixels, matrix, distortion, rvec, tvec,
                            rvec is not None, cv2.SOLVEPNP_ITERATIVE)
    normalized = undistort_points(pixels, matrix, distortion, model)
    return cv2.solvePnP(
        np.asarray(objects, np.float64), normalized,
        np.eye(3, dtype=np.float64), None, rvec, tvec,
        rvec is not None, cv2.SOLVEPNP_ITERATIVE,
    )


def solve_pose_ransac(objects, pixels, matrix, distortion, model):
    if model == "opencv_pinhole":
        points, effective_matrix, effective_distortion, threshold = pixels, matrix, distortion, 3.0
    elif model == "opencv_fisheye":
        points = undistort_points(pixels, matrix, distortion, model)
        effective_matrix, effective_distortion = np.eye(3, dtype=np.float64), None
        threshold = 3.0 / min(matrix[0, 0], matrix[1, 1])
    else:
        raise ValueError(f"unsupported calibration model: {model}")
    return cv2.solvePnPRansac(
        objects, points, effective_matrix, effective_distortion,
        flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=300,
        reprojectionError=threshold, confidence=0.999,
    )


def undistort_preview(source, matrix, distortion, model):
    size = (source.shape[1], source.shape[0])
    if model == "opencv_pinhole":
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 1.0, size)
        return cv2.undistort(source, matrix, distortion, None, new_matrix)
    if model == "opencv_fisheye":
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix, np.asarray(distortion).reshape(4, 1), size, np.eye(3), balance=1.0, new_size=size)
        return cv2.fisheye.undistortImage(
            source, matrix, np.asarray(distortion).reshape(4, 1), Knew=new_matrix, new_size=size)
    raise ValueError(f"unsupported calibration model: {model}")
