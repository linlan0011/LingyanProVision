"""Bounded-position association of measured lower corners to fixed map IDs.

Heading is held fixed throughout the prior disk. Layer labels are supplied by
the existing detector/classifier; this module does not prove physical height.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from calibration.lens_models import project_points, undistort_points
from pose_io import normalize_heading_deg
from safety_identity import ActiveTemplate, CornerIdentity, CornerObservation, map_corners


@dataclass(frozen=True)
class GroundAssociationConfig:
    prior_radius_mm: float = 200.
    visible_tolerance_mm: float = 30.
    inferred_tolerance_mm: float = 50.
    image_margin_px: float = 2.
    max_search_nodes: int = 20000

    def __post_init__(self):
        values = (self.prior_radius_mm, self.visible_tolerance_mm,
                  self.inferred_tolerance_mm, self.image_margin_px)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('association radii and margins must be finite and nonnegative')
        if self.max_search_nodes < 1:
            raise ValueError('association search budget must be positive')


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
            raise ValueError('lower-corner association requires the ground plane Z=0')
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
        angles = np.arange(128) * 2*math.pi/128
        radius = ranger.max_distance_mm / math.cos(math.pi/128)
        polygon = np.c_[np.cos(angles), np.sin(angles)] * radius
        camera_rotation = ranger.rotation.T
        camera_offset = -camera_rotation @ ranger.center
        polygon = _clip_polygon(polygon, np.array([1., 0.]), 0.)
        polygon = _clip_polygon(polygon, camera_rotation[2, :2], camera_offset[2]-1e-6)
        for first, second in zip(hull, np.roll(hull, -1, axis=0)):
            dx, dy = second-first
            normal = np.array([-dy, dx])
            offset = -normal @ first + angular_margin*np.linalg.norm(normal)
            weights = np.r_[normal, offset]
            polygon = _clip_polygon(polygon, weights @ camera_rotation[:, :2],
                                    float(weights @ camera_offset))
        self.body_polygon = polygon

    def contains_body(self, xy):
        """Exact calibrated image/range check for a particular body point."""
        xy = np.asarray(xy, float)
        if not np.isfinite(xy).all() or xy[0] <= 0:
            return False
        if np.linalg.norm(xy) > self.ranger.max_distance_mm + 1e-6:
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

    def footprint(self, prior):
        return self.body_polygon @ heading_rotation(prior.heading_deg).T + [prior.x_mm, prior.y_mm]

    def landmarks(self, prior, radius, model):
        polygon = self.footprint(prior)
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
        templates.add(ActiveTemplate(zone.safety_zone_id, zone.view_side, view['scene'],
                                    tuple(view.get('slots', ())), view.get('identity_enabled', True),
                                    view.get('identity_mode', 'directed_edges'),
                                    tuple(view.get('image_slots', ()))))
    return sorted(templates, key=lambda t: (t.safety_id, t.side))


def disk_intersection_witness(disks):
    """Find a point in ALL closed disks, including tangent/zero-radius cases.

    Pairwise overlap alone is insufficient for three or more disks. A nonempty
    intersection contains a disk center or a boundary intersection of two disks.
    """
    disks = [(np.asarray(center, float), float(radius)) for center, radius in disks]
    candidates = [center for center, _ in disks]
    for index, (first, r1) in enumerate(disks):
        for second, r2 in disks[index+1:]:
            vector = second-first
            distance = float(np.linalg.norm(vector))
            if distance < 1e-9 or distance > r1+r2+1e-7 or distance < abs(r1-r2)-1e-7:
                continue
            along = (r1*r1-r2*r2+distance*distance)/(2*distance)
            height = math.sqrt(max(0., r1*r1-along*along))
            unit = vector/distance
            base = first+along*unit
            normal = np.array([-unit[1], unit[0]])
            candidates.extend((base+height*normal, base-height*normal))
    for point in candidates:
        if all(np.linalg.norm(point-center) <= radius+1e-6 for center, radius in disks):
            return point
    return None


def associate_ground_corners(readings, layers, prior, model, view, config=None):
    config = config or GroundAssociationConfig()
    identities = [CornerIdentity(CornerObservation(tuple(r.pixel), r.inferred, ()),
                                 reason='not_lower_layer') for r in readings]
    report = dict(schema='safety.ground_circle_association.v1', config=asdict(config),
                  status='waiting_for_ground_corners', heading_model='fixed_prior_heading',
                  max_distance_mm=view.ranger.max_distance_mm if view else None,
                  identity_condition='conditional_on_lower_layer_and_prior_bound',
                  pose_corrected=False, possible_safety_ids=[], landmarks=[], observations=[],
                  hypothesis_count=0, matched_count=0, search_nodes=0,
                  vision_vehicle_position_mm=None, compatible_vehicle_position_mm=None,
                  central_footprint_map_mm=[], possible_footprint_map_mm=[])
    if view is None or not view.ranger.check_sources():
        report['status'] = 'calibration_unavailable'
        return identities, report
    if prior is None:
        report['status'] = 'waiting_for_prior'
        return identities, report
    if model is None:
        report['status'] = 'map_unavailable'
        return identities, report
    center = np.array([prior.x_mm, prior.y_mm])
    rotation = heading_rotation(prior.heading_deg)
    landmarks = view.landmarks(prior, config.prior_radius_mm, model)
    report['landmarks'] = landmarks
    report['possible_safety_ids'] = sorted({p['safety_id'] for p in landmarks if p['possibly_visible']})
    footprint = view.footprint(prior)
    report['central_footprint_map_mm'] = footprint.tolist()
    if len(footprint):
        angles = np.arange(64)*2*math.pi/64
        offsets = np.c_[np.cos(angles), np.sin(angles)] * config.prior_radius_mm/math.cos(math.pi/64)
        union = cv2.convexHull((footprint[:, None, :]+offsets).reshape(-1, 2).astype(np.float32))
        report['possible_footprint_map_mm'] = union.reshape(-1, 2).tolist()
    if layers is None:
        return identities, report
    layer_map = layers.by_pixel
    observations = []
    for index, reading in enumerate(readings):
        key = tuple(map(int, np.rint(reading.pixel)))
        layer = layer_map.get(key)
        if layer is None or layer.layer not in ('lower_seed', 'lower'):
            continue
        identity = identities[index]
        identity.reason = 'invalid_ground_range'
        if (not reading.valid or reading.distance_mm is None or reading.x_mm is None
                or reading.y_mm is None or reading.z_mm is None
                or not math.isclose(reading.z_mm, 0., abs_tol=1e-5)
                or not view.contains_body([reading.x_mm, reading.y_mm])):
            continue
        relative = np.array([reading.x_mm, reading.y_mm])
        expected_map = center+rotation@relative
        tolerance = config.inferred_tolerance_mm if reading.inferred else config.visible_tolerance_mm
        record = dict(pixel=list(key), inferred=reading.inferred, layer=layer.layer,
                      body_xy_mm=relative.tolist(), range_mm=reading.distance_mm,
                      map_center_mm=expected_map.tolist(), map_radius_mm=config.prior_radius_mm+tolerance,
                      measurement_tolerance_mm=tolerance, candidates=[], corner_id=None,
                      reason='no_compatible_map_corner', vehicle_hypotheses=[])
        options = []
        for landmark in landmarks:
            if not landmark['possibly_visible']:
                continue
            vehicle = np.asarray(landmark['map_xy_mm'])-rotation@relative
            if np.linalg.norm(vehicle-center) <= config.prior_radius_mm+tolerance+1e-6:
                options.append((landmark['corner_id'], vehicle, tolerance))
                record['vehicle_hypotheses'].append(dict(corner_id=landmark['corner_id'],
                                                        xy_mm=vehicle.tolist()))
        record['candidates'] = [option[0] for option in options]
        identity.candidates = set(record['candidates'])
        observations.append((index, record, options))
    report['observations'] = [record for _, record, _ in observations]
    if not observations:
        return identities, report
    best, solutions, witnesses, nodes, exceeded = -1, [], [], 0, False

    def search(position, assigned, used, disks, safety_id):
        nonlocal best, solutions, witnesses, nodes, exceeded
        nodes += 1
        if nodes > config.max_search_nodes:
            exceeded = True
            return
        if exceeded or len(used)+len(observations)-position < best:
            return
        if position == len(observations):
            count = len(used)
            if count > best:
                best, solutions, witnesses = count, [], []
            if count == best:
                solutions.append(tuple(assigned))
                witnesses.append(disk_intersection_witness(disks))
            return
        for corner_id, vehicle, tolerance in observations[position][2]:
            zone = corner_id.split('.')[0]
            if corner_id in used or (safety_id is not None and safety_id != zone):
                continue
            proposed = disks+[(vehicle, tolerance)]
            if disk_intersection_witness(proposed) is not None:
                search(position+1, assigned+[corner_id], used|{corner_id}, proposed, zone)
        search(position+1, assigned+[None], used, disks, safety_id)

    search(0, [], set(), [(center, config.prior_radius_mm)], None)
    report.update(hypothesis_count=len(solutions), matched_count=max(0, best), search_nodes=nodes)
    if exceeded:
        report['status'] = 'search_limit_no_confirmation'
        for index, record, _ in observations:
            record['reason'] = identities[index].reason = 'search_limit_no_confirmation'
        return identities, report
    confirmed_positions = []
    for position, (index, record, options) in enumerate(observations):
        alternatives = {solution[position] for solution in solutions}
        identity = identities[index]
        if len(alternatives) == 1 and None not in alternatives:
            identity.corner_id = record['corner_id'] = next(iter(alternatives))
            record['reason'] = identity.reason = 'unique_joint_circle_match'
            confirmed_positions.append(next(vehicle for name, vehicle, _ in options
                                            if name == identity.corner_id))
        elif options:
            record['reason'] = identity.reason = 'ambiguous_or_joint_conflict'
        else:
            identity.reason = record['reason']
    if confirmed_positions:
        report['vision_vehicle_position_mm'] = np.mean(confirmed_positions, axis=0).tolist()
        report['compatible_vehicle_position_mm'] = witnesses[0].tolist()
        report['status'] = 'unique' if len(solutions) == 1 else 'partial_unique'
    else:
        report['status'] = 'ambiguous' if best > 0 else 'no_match'
    return identities, report


def apply_ground_identities(readings, identities):
    """Replace legacy IDs, including clearing labels on upper/unmatched points."""
    from dataclasses import replace
    names = {tuple(r.observation.point): r.corner_id for r in identities}
    return [replace(r, corner_id=names.get(tuple(r.pixel)),
                    label=r.label.split('/')[0]+('/'+names[tuple(r.pixel)] if names.get(tuple(r.pixel)) else ''))
            for r in readings]
