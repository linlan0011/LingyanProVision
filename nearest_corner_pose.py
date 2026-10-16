"""Nearest-corner identity and visual SE(2) pose; gyro selects correspondence only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import permutations
import math

import numpy as np

from ground_corner_association import heading_rotation
from pose_io import normalize_heading_deg
from safety_identity import CornerIdentity, CornerObservation, map_corners


@dataclass(frozen=True)
class VisualPoseConfig:
    identity_tolerance_deg: float = 15.
    orthogonality_tolerance_deg: float = 15.
    heading_consistency_deg: float = 15.
    minimum_edge_span_mm: float = 20.
    maximum_line_residual_mm: float = 20.

    def __post_init__(self):
        for value in (self.identity_tolerance_deg, self.orthogonality_tolerance_deg,
                      self.heading_consistency_deg):
            if not math.isfinite(value) or not 0 <= value < 45:
                raise ValueError('visual angle tolerances must be at least 0 and below 45 degrees')
        for value in (self.minimum_edge_span_mm, self.maximum_line_residual_mm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('visual edge span/residual limits must be positive and finite')


def pixel_key(pixel):
    return tuple(map(int, np.rint(pixel)))


def _support_pixels(support, boundary_result):
    """Sample actual supported paths, including separated pieces, never gaps."""
    source = support
    geometry = None
    line_id = getattr(support, 'line_id', 0)
    if boundary_result is not None and line_id:
        source = next((line for line in boundary_result.lines if line.line_id == line_id), None)
        if source is None:
            return np.empty((0, 2))
        geometry = boundary_result.config.point_geometry
    paths = [getattr(member, 'boundary_path', ())
             for member in getattr(source, 'members', (source,))]
    sampled = []
    for path in paths:
        path = np.asarray(path, float).reshape(-1, 2)
        if len(path):
            path = path[np.unique(np.linspace(0, len(path)-1, min(17, len(path))).astype(int))]
            sampled.extend(geometry.to_pixels(path) if geometry is not None else path)
    return np.asarray(sampled, float).reshape(-1, 2)


def _edge_direction(corner, support, ranger, boundary_result, config):
    points = []
    for pixel in _support_pixels(support, boundary_result):
        reading = ranger.project(pixel, enforce_distance_limit=False)
        if reading.valid:
            points.append((reading.x_mm, reading.y_mm))
    if len(points) < 3:
        return None, 'insufficient_edge_samples'
    points = np.asarray(points, float)
    origin = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points-origin, full_matrices=False)
    direction = vectors[0]
    if float((origin-corner) @ direction) < 0:
        direction = -direction
    along = (points-corner) @ direction
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.sqrt(np.mean(((points-origin) @ normal)**2)))
    span = float(np.ptp(along))
    corner_residual = float(abs((corner-origin) @ normal))
    record = dict(line_id=getattr(support, 'line_id', 0), sample_count=len(points),
                  body_unit=direction.tolist(), body_angle_deg=math.degrees(math.atan2(direction[1], direction[0])),
                  span_mm=span, residual_mm=residual, corner_residual_mm=corner_residual)
    if span < config.minimum_edge_span_mm:
        return record, 'edge_too_short'
    if max(residual, corner_residual) > config.maximum_line_residual_mm:
        return record, 'edge_fit_unreliable'
    # A signed corner ray must have its observed span on ONE side of the corner.
    if float(np.min(along)) < -config.maximum_line_residual_mm:
        return record, 'support_crosses_corner'
    if float(np.max(along)) < config.minimum_edge_span_mm:
        return record, 'edge_direction_unreliable'
    return record, None


def _finite_ground(reading):
    return (reading.valid and reading.z_mm is not None
            and math.isclose(reading.z_mm, 0., abs_tol=1e-5)
            and all(value is not None and math.isfinite(value)
                    for value in (reading.x_mm, reading.y_mm, reading.distance_mm)))


def solve_nearest_corner_pose(readings, geometry_readings, supports, prior, model,
                              ranger, candidate_view, *, boundary_result=None, config=None):
    """Screen zones by the retained view; match directions, then solve without gyro values.

    No position/measurement error-circle association, upper/lower classification,
    nearest-to-prior ranking, or fallback to a second-nearest point is performed.
    """
    config = config or VisualPoseConfig()
    identities = [CornerIdentity(CornerObservation(tuple(r.pixel), r.inferred, ()),
                                 reason='not_selected') for r in readings]
    report = dict(schema='safety.nearest_visual_pose.v1', config=asdict(config),
                  status='no_valid_corner', candidate_view=candidate_view,
                  candidate_count=len(geometry_readings), selected_corner=None,
                  edge_directions=[], hypotheses=[], corner_id=None,
                  corrected_pose=None, pose_corrected=False, input_pose_overwritten=False,
                  heading_source='visual_ground_edges', position_source='visual_corner_and_heading',
                  identity_condition='nearest_point_and_supports_assumed_ground_plane',
                  delta_pose=None)
    if ranger is None or ranger.target_z_mm != 0 or not ranger.check_sources():
        report['status'] = 'calibration_unavailable'
        return identities, report
    # Geometry ranking precedes the localization range gate and includes inferred points.
    valid = [r for r in geometry_readings if _finite_ground(r)]
    if not valid:
        return identities, report
    nearest = min(valid, key=lambda r: (r.distance_mm, pixel_key(r.pixel), r.inferred))
    key = pixel_key(nearest.pixel)
    report['selected_corner'] = dict(pixel=list(key), inferred=nearest.inferred,
                                   body_xy_mm=[nearest.x_mm, nearest.y_mm],
                                   range_mm=nearest.distance_mm, bearing_deg=nearest.bearing_deg)
    matched = next(((index, r) for index, r in enumerate(readings) if pixel_key(r.pixel) == key), None)
    if matched is None or not _finite_ground(matched[1]) or matched[1].geometry_only:
        report['status'] = 'nearest_out_of_range' if matched and matched[1].reason == 'beyond_distance_limit' else 'nearest_unusable'
        return identities, report
    index, measured = matched
    identity = identities[index]

    def reject(reason):
        identity.reason = report['status'] = reason
        return identities, report

    if prior is None:
        return reject('waiting_for_prior')
    if model is None:
        return reject('map_unavailable')
    if candidate_view is None or candidate_view['status'] != 'ready':
        return reject(candidate_view['status'] if candidate_view else 'calibration_unavailable')
    safety_ids = set(candidate_view['possible_safety_ids'])
    if not safety_ids:
        return reject('no_candidate_safety_zone')
    pair = supports.get(key, ())
    if len(pair) != 2:
        return reject('two_support_edges_required')
    corner = np.array([measured.x_mm, measured.y_mm])
    for support in pair:
        edge, error = _edge_direction(corner, support, ranger, boundary_result, config)
        if edge is not None:
            report['edge_directions'].append(edge)
        if error:
            return reject(error)
    betas = [edge['body_angle_deg'] for edge in report['edge_directions']]
    separation = abs(normalize_heading_deg(betas[0]-betas[1]))
    report['orthogonality_error_deg'] = abs(separation-90.)
    if abs(separation-90.) > config.orthogonality_tolerance_deg+1e-7:
        return reject('edges_not_orthogonal')
    predicted = [normalize_heading_deg(prior.heading_deg+beta) for beta in betas]
    for edge, angle in zip(report['edge_directions'], predicted):
        edge['gyro_predicted_map_angle_deg'] = angle
    for zone in model['safety_zones']:
        if zone['id'] not in safety_ids:
            continue
        corners = np.asarray(map_corners(zone), float)
        for number, point in enumerate(corners):
            target_vectors = (corners[(number-1) % 4]-point, corners[(number+1) % 4]-point)
            phis = [math.degrees(math.atan2(v[1], v[0])) for v in target_vectors]
            for assignment in permutations(phis):
                errors = [abs(normalize_heading_deg(alpha-phi)) for alpha, phi in zip(predicted, assignment)]
                if max(errors) > config.identity_tolerance_deg+1e-7:
                    continue
                headings = [normalize_heading_deg(phi-beta) for phi, beta in zip(assignment, betas)]
                disagreement = abs(normalize_heading_deg(headings[0]-headings[1]))
                hypothesis = dict(corner_id=f"{zone['id']}.{number+1}", safety_id=zone['id'],
                                  map_xy_mm=point.tolist(), map_edge_angles_deg=list(assignment),
                                  identity_errors_deg=errors, edge_headings_deg=headings,
                                  heading_disagreement_deg=disagreement, corrected_pose=None)
                if disagreement <= config.heading_consistency_deg+1e-7:
                    angles = np.radians(headings)
                    heading = normalize_heading_deg(math.degrees(math.atan2(np.sin(angles).sum(), np.cos(angles).sum())))
                    position = point-heading_rotation(heading) @ corner
                    hypothesis['corrected_pose'] = dict(x_mm=float(position[0]), y_mm=float(position[1]), heading_deg=heading)
                report['hypotheses'].append(hypothesis)
    identity.candidates = {h['corner_id'] for h in report['hypotheses']}
    if not report['hypotheses']:
        return reject('identity_no_match')
    if len(report['hypotheses']) != 1:
        zones = {h['safety_id'] for h in report['hypotheses']}
        return reject('ambiguous_safety_zone' if len(zones) > 1 else 'ambiguous_corner')
    hypothesis = report['hypotheses'][0]
    if hypothesis['corrected_pose'] is None:
        return reject('visual_heading_inconsistent')
    pose = hypothesis['corrected_pose']
    identity.corner_id = report['corner_id'] = hypothesis['corner_id']
    identity.reason = 'unique_directed_edge_identity'
    report.update(status='corrected', corrected_pose=pose, pose_corrected=True,
                  delta_pose=dict(x_mm=pose['x_mm']-prior.x_mm, y_mm=pose['y_mm']-prior.y_mm,
                                  heading_deg=normalize_heading_deg(pose['heading_deg']-prior.heading_deg)))
    return identities, report


def apply_visual_identities(readings, identities):
    names = {pixel_key(identity.observation.point): identity.corner_id for identity in identities}
    return [replace(reading, corner_id=names.get(pixel_key(reading.pixel)),
                    label=reading.label.split('/')[0]+('/'+names[pixel_key(reading.pixel)] if names.get(pixel_key(reading.pixel)) else ''))
            for reading in readings]
