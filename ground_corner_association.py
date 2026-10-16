"""Retained calibrated view prefilter over a bounded input-position disk."""
from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.lens_models import project_points, undistort_points
from pose_io import normalize_heading_deg
from safety_identity import ActiveTemplate, map_corners


def heading_rotation(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def _clip_polygon(polygon, normal, offset):
    """Clip a convex XY polygon by normal @ XY + offset >= 0."""
    output = []
    if not len(polygon):
        return np.empty((0, 2))
    previous = polygon[-1]
    old = float(normal @ previous + offset)
    for point in polygon:
        value = float(normal @ point + offset)
        if (value >= 0) != (old >= 0):
            output.append(previous + (point-previous) * old / (old-value))
        if value >= 0:
            output.append(point)
        previous, old = point, value
    return np.asarray(output, float).reshape(-1, 2)


class CalibratedGroundView:
    """Conservative ground footprint from calibrated image-border rays.

    The undistorted border's convex hull is expanded by an angular margin.
    A circumscribed range polygon preserves the entire exact range disk.
    These outer approximations are only used to reject impossible candidates;
    actual observation ranges and image projections are checked separately.
    """
    def __init__(self, ranger, image_margin_px=2.):
        if ranger.target_z_mm != 0:
            raise ValueError('candidate safety view requires the ground plane Z=0')
        self.ranger = ranger
        self.image_margin_px = image_margin_px
        width, height = ranger.size
        u, v = np.linspace(0, width-1, 513), np.linspace(0, height-1, 513)
        border = np.vstack((np.c_[u, np.zeros_like(u)],
                            np.c_[np.full_like(v, width-1), v],
                            np.c_[u, np.full_like(u, height-1)],
                            np.c_[np.zeros_like(v), v]))
        normalized = undistort_points(border, ranger.matrix, ranger.distortion, ranger.model)
        if not np.isfinite(normalized).all():
            raise ValueError('nonfinite calibrated border rays')
        hull = cv2.convexHull(normalized.astype(np.float32)).reshape(-1, 2).astype(float)
        # Dense border samples + outward margin avoid an inward approximation
        # at curved image edges. Convexification may retain extra candidates.
        angular_margin = (image_margin_px+1.) / min(ranger.matrix[0, 0], ranger.matrix[1, 1])
        camera_rotation = ranger.rotation.T
        camera_offset = -camera_rotation @ ranger.center
        self.halfplanes = [(np.array([1., 0.]), 0.),
                           (camera_rotation[2, :2], camera_offset[2]-1e-6)]
        for first, second in zip(hull, np.roll(hull, -1, axis=0)):
            dx, dy = second-first
            normal = np.array([-dy, dx])
            offset = -normal @ first + angular_margin*np.linalg.norm(normal)
            weights = np.r_[normal, offset]
            self.halfplanes.append((weights @ camera_rotation[:, :2],
                                    float(weights @ camera_offset)))
        # An unlimited frustum may be unbounded at the horizon. The map-facing
        # footprint is cropped dynamically to cover EVERY map landmark plus
        # the prior disk. This finite computation window is not a range gate.
        self.body_polygon = self._body_polygon(ranger.max_distance_mm or 10000.)

    def _body_polygon(self, extent_mm):
        radius = self.ranger.max_distance_mm
        if radius is None:
            radius = extent_mm
        angles = np.arange(128) * 2*math.pi/128
        polygon = np.c_[np.cos(angles), np.sin(angles)] * radius / math.cos(math.pi/128)
        for normal, offset in self.halfplanes:
            polygon = _clip_polygon(polygon, normal, offset)
        return polygon

    def map_extent(self, prior, radius, model):
        if self.ranger.max_distance_mm is not None:
            return self.ranger.max_distance_mm
        center = np.array([prior.x_mm, prior.y_mm])
        distances = [np.linalg.norm(np.asarray(point)-center)
                     for zone in model['safety_zones'] for point in map_corners(zone)]
        return float(max(distances, default=0.) + radius + 1.)

    def contains_body(self, xy):
        """Exact calibrated image/range check for a particular body point."""
        xy = np.asarray(xy, float)
        if not np.isfinite(xy).all() or xy[0] <= 0:
            return False
        if (self.ranger.max_distance_mm is not None
                and np.linalg.norm(xy) > self.ranger.max_distance_mm + 1e-6):
            return False
        body = np.r_[xy, 0.]
        camera = self.ranger.rotation.T @ (body-self.ranger.center)
        if camera[2] <= 0:
            return False
        rvec, _ = cv2.Rodrigues(self.ranger.rotation.T)
        pixel = project_points(body.reshape(1, 3), rvec,
                               -self.ranger.rotation.T @ self.ranger.center,
                               self.ranger.matrix, self.ranger.distortion, self.ranger.model)[0]
        width, height = self.ranger.size
        margin = self.image_margin_px
        return bool(np.isfinite(pixel).all() and
                    -margin <= pixel[0] <= width-1+margin and
                    -margin <= pixel[1] <= height-1+margin)

    def footprint(self, prior, *, extent_mm=None):
        # With a fixed range cap the body-frame footprint is constant. Reuse
        # the startup result, including when callers supply a map extent.
        polygon = (self.body_polygon if self.ranger.max_distance_mm is not None or extent_mm is None
                   else self._body_polygon(extent_mm))
        return polygon @ heading_rotation(prior.heading_deg).T + [prior.x_mm, prior.y_mm]

    def landmarks(self, prior, radius, model):
        polygon = self.footprint(prior, extent_mm=self.map_extent(prior, radius, model))
        contour = polygon.astype(np.float32).reshape(-1, 1, 2)
        result = []
        for zone in model['safety_zones']:
            for index, point in enumerate(map_corners(zone), 1):
                distance = (-cv2.pointPolygonTest(contour, tuple(map(float, point)), True)
                            if len(contour) >= 3 else math.inf)
                possible = distance <= radius+1e-4
                result.append(dict(corner_id=f"{zone['id']}.{index}", safety_id=zone['id'],
                                   map_xy_mm=list(point), possibly_visible=possible,
                                   reason='within_disk_view_union' if possible else 'outside_disk_view_union'))
        return result


def circle_templates(prior, zones, model, safety_ids, radius):
    """Select detector templates over the whole disk, without a half-field cut."""
    if prior is None or model is None:
        return []
    templates = set()
    for zone in zones:
        if zone.safety_zone_id not in safety_ids or zone.view_side not in model['views']:
            continue
        dx = max(zone.x_min_mm-prior.x_mm, 0., prior.x_mm-zone.x_max_mm)
        dy = max(zone.y_min_mm-prior.y_mm, 0., prior.y_mm-zone.y_max_mm)
        if math.hypot(dx, dy) > radius:
            continue
        view = model['views'][zone.view_side]
        if abs(normalize_heading_deg(prior.heading_deg-view['heading_deg'])) > view['heading_tolerance_deg']:
            continue
        templates.add(ActiveTemplate(zone.safety_zone_id, zone.view_side, view['scene']))
    return sorted(templates, key=lambda t: (t.safety_id, t.side))


def candidate_safety_view(prior, model, view, radius=200.):
    """Keep a zone when any of its four corners can enter the disk-view union.

    This is the only retained use of the input-position uncertainty disk.
    It does not associate a measured corner with a map corner.
    """
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('position radius must be finite and nonnegative')
    report = dict(schema='safety.candidate_view.v1', status='ready',
                  prior_radius_mm=radius, max_distance_mm=view.ranger.max_distance_mm if view else None,
                  possible_safety_ids=[], landmarks=[], central_footprint_map_mm=[],
                  possible_footprint_map_mm=[], heading_model='fixed_input_heading')
    if view is None or not view.ranger.check_sources():
        report['status'] = 'calibration_unavailable'
    elif prior is None:
        report['status'] = 'waiting_for_prior'
    elif model is None:
        report['status'] = 'map_unavailable'
    else:
        report['landmarks'] = view.landmarks(prior, radius, model)
        report['possible_safety_ids'] = sorted({point['safety_id'] for point in report['landmarks']
                                               if point['possibly_visible']})
        extent = view.map_extent(prior, radius, model)
        footprint = view.footprint(prior, extent_mm=extent)
        report['central_footprint_map_mm'] = footprint.tolist()
        if len(footprint):
            angles = np.arange(64)*2*math.pi/64
            offsets = np.c_[np.cos(angles), np.sin(angles)] * radius/math.cos(math.pi/64)
            union = cv2.convexHull((footprint[:, None, :]+offsets).reshape(-1, 2).astype(np.float32))
            report['possible_footprint_map_mm'] = union.reshape(-1, 2).tolist()
    return report
