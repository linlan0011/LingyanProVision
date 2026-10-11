"""Ordered purple/white boundary pieces, multiple robust lines and their corners.

Colour thresholds belong to the caller. Green observations never include an
invented gap; only fitted lines and inferred corners may cross an occlusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from calibration.lens_models import undistort_points, project_points


@dataclass
class PointGeometry:
    """Rectified pixel coordinates for fitting; raw pixels for observations/I/O."""
    matrix: np.ndarray
    distortion: np.ndarray
    model: str

    @classmethod
    def from_intrinsics(cls, data):
        return cls(np.asarray(data['camera_matrix'], float),
                   np.asarray(data['distortion_coefficients'], float), data['model'])

    def to_geometry(self, points):
        normalized = undistort_points(points, self.matrix, self.distortion, self.model)
        homogeneous = np.column_stack((normalized, np.ones(len(normalized)))) @ self.matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def to_pixels(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(self.matrix).T
        return project_points(rays, np.zeros(3), np.zeros(3), self.matrix, self.distortion, self.model)


@dataclass
class PixelLineSupport:
    fitted_start: np.ndarray
    fitted_end: np.ndarray


@dataclass(frozen=True)
class BoundaryConfig:
    minimum_edge: float = 12.0
    minimum_white_support: float = 0.4
    approximation_px: float = 1.5
    fit_tolerance_px: float = 2.0
    merge_angle_deg: float = 8.0
    minimum_turn_deg: float = 5.0
    infer_corners: bool = True
    point_geometry: PointGeometry | None = None


@dataclass
class BoundaryPiece:
    component_index: int
    start_index: int
    end_index: int
    start: np.ndarray
    end: np.ndarray
    boundary_path: np.ndarray
    white_normal: np.ndarray
    support: float
    fitted_start: np.ndarray
    fitted_end: np.ndarray
    fitted_origin: np.ndarray
    fitted_direction: np.ndarray
    length: float


@dataclass
class BoundaryLine:
    members: list[BoundaryPiece]
    fitted_start: np.ndarray
    fitted_end: np.ndarray
    fitted_origin: np.ndarray
    fitted_direction: np.ndarray
    white_normal: np.ndarray
    length: float
    observed_length: float
    support: float
    residual_px: float
    line_id: int = 0


@dataclass
class OrderedBoundary:
    points: np.ndarray
    valid: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    orientation: float
    raw_points: np.ndarray


@dataclass
class BoundaryResult:
    image_shape: tuple[int, int] = (0, 0)
    boundaries: list[OrderedBoundary] = field(default_factory=list)
    pieces: list[BoundaryPiece] = field(default_factory=list)
    lines: list[BoundaryLine] = field(default_factory=list)
    visible: list = field(default_factory=list)
    inferred: list = field(default_factory=list)
    corner_supports: dict = field(default_factory=dict)
    config: BoundaryConfig = field(default_factory=BoundaryConfig)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(1e-9, float(np.linalg.norm(vector)))


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _sample(mask, points):
    xy = np.rint(points).astype(np.int32)
    height, width = mask.shape
    inside = ((xy[..., 0] >= 0) & (xy[..., 0] < width)
              & (xy[..., 1] >= 0) & (xy[..., 1] < height))
    values = np.zeros(inside.shape, dtype=bool)
    values[inside] = mask[xy[..., 1][inside], xy[..., 0][inside]] != 0
    return values


def _dense_contour(contour):
    """Let OpenCV reconstruct adjacent, ordered pixels from a compressed contour."""
    x, y, width, height = cv2.boundingRect(contour)
    local = np.zeros((height + 2, width + 2), np.uint8)
    shift = np.array([x - 1, y - 1], np.int32)
    cv2.drawContours(local, [contour.astype(np.int32) - shift], -1, 255, cv2.FILLED)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), np.float64)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float) + shift


def observe_boundary(contour, purple, white, config):
    points = _dense_contour(contour)
    count = len(points)
    if count < 3:
        return OrderedBoundary(points, np.zeros(count, bool), np.zeros_like(points),
                               np.zeros(count), 1., points)
    orientation = 1. if cv2.contourArea(points.astype(np.float32), oriented=True) > 0 else -1.
    # Symmetric local tangents suppress one-pixel staircase directions.
    radius = min(4, max(1, count // 8))
    tangent = np.roll(points, -radius, axis=0) - np.roll(points, radius, axis=0)
    tangent /= np.maximum(1e-9, np.linalg.norm(tangent, axis=1))[:, None]
    normals = orientation * np.column_stack((tangent[:, 1], -tangent[:, 0]))
    # Test tight neighbourhoods, without searching across a dark/coloured object.
    offsets = np.arange(1., 5.)[None, :, None]
    outside = points[:, None, :] + offsets * normals[:, None, :]
    inside = points[:, None, :] - offsets * normals[:, None, :]
    white_votes = _sample(white, outside) & ~_sample(purple, outside)
    purple_votes = _sample(purple, inside)
    white_rate = white_votes.mean(axis=1)
    purple_rate = purple_votes.mean(axis=1)
    # A distant acute corner can be only one purple pixel wide. Requiring a
    # fixed inward stripe would erase that real corner. The ordered contour
    # supplies purple-side ownership; inward support is a confidence weight.
    valid = ((white_rate >= max(.5, config.minimum_white_support))
             & _sample(purple, points) & white_votes[:, :2].any(axis=1))
    height, width = purple.shape
    valid &= ((points[:, 0] > 0) & (points[:, 0] < width - 1)
              & (points[:, 1] > 0) & (points[:, 1] < height - 1))
    raw_points = points
    if config.point_geometry is not None:
        points = config.point_geometry.to_geometry(points)
        tangent = np.roll(points, -radius, axis=0) - np.roll(points, radius, axis=0)
        tangent /= np.maximum(1e-9, np.linalg.norm(tangent, axis=1))[:, None]
        normals = orientation * np.column_stack((tangent[:, 1], -tangent[:, 0]))
    return OrderedBoundary(points, valid, normals, white_rate * (.5 + .5 * purple_rate),
                           orientation, raw_points)


def _runs(valid):
    """Return ordered valid index runs; only a completely valid contour is closed."""
    count = len(valid)
    if not count or not valid.any():
        return []
    if valid.all():
        return [(np.arange(count), True)]
    start = (int(np.flatnonzero(~valid)[0]) + 1) % count
    order = (start + np.arange(count)) % count
    selected = valid[order]
    transitions = np.diff(np.r_[False, selected, False].astype(np.int8))
    return [(order[a:b], False) for a, b in
            zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))]


def _fit(points, reference):
    vx, vy, x, y = cv2.fitLine(np.asarray(points, np.float32), cv2.DIST_HUBER,
                              0, .01, .001).ravel()
    direction = _unit((vx, vy))
    if np.dot(direction, reference) < 0:
        direction = -direction
    origin = np.array((x, y), dtype=float)
    parameters = (points - origin) @ direction
    start = origin + parameters.min() * direction
    end = origin + parameters.max() * direction
    residuals = np.abs((points - origin) @ np.array((-direction[1], direction[0])))
    return start, end, origin, direction, residuals


def split_boundary(boundary, component_index, config):
    pieces = []
    for indices, closed in _runs(boundary.valid):
        if len(indices) < 3:
            continue
        path = boundary.points[indices]
        vertices = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2),
                                   config.approximation_px, closed).reshape(-1, 2)
        positions = [int(np.argmin(np.sum((path - p) ** 2, axis=1))) for p in vertices]
        # OpenCV's closed approximation may rotate its first vertex.
        positions = sorted(set(positions))
        pairs = list(zip(positions[:-1], positions[1:]))
        if closed and len(positions) > 1:
            pairs.append((positions[-1], positions[0] + len(path)))
        for a, b in pairs:
            path_positions = np.arange(a, b + 1) % len(path)
            segment_indices = indices[path_positions]
            segment = boundary.points[segment_indices]
            if np.linalg.norm(segment[-1] - segment[0]) < 2.:
                continue
            start, end, origin, direction, _ = _fit(segment, segment[-1] - segment[0])
            pieces.append(BoundaryPiece(
                component_index, int(segment_indices[0]), int(segment_indices[-1]),
                segment[0], segment[-1], segment,
                _unit(boundary.normals[segment_indices].mean(axis=0)),
                float(boundary.confidence[segment_indices].mean()),
                start, end, origin, direction, float(np.linalg.norm(end - start)),
            ))
    return pieces


def _build_line(members):
    points = np.vstack([p.boundary_path for p in members])
    longest = max(members, key=lambda p: p.length)
    start, end, origin, direction, residuals = _fit(points, longest.fitted_direction)
    observed = sum(p.length for p in members)
    return BoundaryLine(
        members, start, end, origin, direction,
        _unit(sum(p.white_normal * p.length for p in members)),
        float(np.linalg.norm(end - start)), observed,
        sum(p.support * p.length for p in members) / max(1e-9, observed),
        float(np.quantile(residuals, .90)),
    )


def merge_boundary_pieces(pieces, boundaries, config):
    """Complete-group checks prevent chains of individually plausible bad merges."""
    groups = []
    tolerance = config.fit_tolerance_px

    def compatible(first, second):
        angular_noise = np.rad2deg(np.arctan2(2. * tolerance, max(2., min(first.length, second.length))))
        touching = (first.component_index == second.component_index
                    and (first.end_index == second.start_index or second.end_index == first.start_index))
        angle_limit = (config.merge_angle_deg if touching
                       else min(25., config.merge_angle_deg + angular_noise))
        cosine = float(np.cos(np.deg2rad(angle_limit)))
        if abs(np.dot(first.fitted_direction, second.fitted_direction)) < cosine:
            return False
        if np.dot(first.white_normal, second.white_normal) < .8:
            return False
        if first.component_index != second.component_index:
            if (boundaries[first.component_index].valid.all()
                    or boundaries[second.component_index].valid.all()):
                return False  # Separate closed objects are not occlusion fragments.
        for a, b in ((first, second), (second, first)):
            normal = np.array((-a.fitted_direction[1], a.fitted_direction[0]))
            distance = np.abs((np.array((b.fitted_start, b.fitted_end))
                               - a.fitted_origin) @ normal)
            if distance.max() > 2. * tolerance:
                return False
        return True

    def gap_compatible(first, second):
        direction = first.fitted_direction
        a = sorted(np.array((first.fitted_start, first.fitted_end)) @ direction)
        b = sorted(np.array((second.fitted_start, second.fitted_end)) @ direction)
        gap = max(0., max(a[0], b[0]) - min(a[1], b[1]))
        return gap <= max(4 * config.minimum_edge, .75 * (first.length + second.length))

    def proposal(first_group, second_group):
        if not all(compatible(a, b) for a in first_group for b in second_group):
            return None
        # Distant fragments can be connected through intermediate observations;
        # only geometry uses an all-pairs check, not the accumulated along-gap.
        if not any(gap_compatible(a, b) for a in first_group for b in second_group):
            return None
        proposed = _build_line(first_group + second_group)
        normal = np.array((-proposed.fitted_direction[1], proposed.fitted_direction[0]))
        if any(np.quantile(np.abs((old.boundary_path - proposed.fitted_origin) @ normal), .90)
               > tolerance for old in proposed.members):
            return None
        return proposed

    for piece in sorted(pieces, key=lambda p: p.length, reverse=True):
        matches = []
        for index, group in enumerate(groups):
            proposed = proposal(group, [piece])
            if proposed is not None:
                matches.append((proposed.residual_px, index))
        if matches:
            groups[min(matches)[1]].append(piece)
        else:
            groups.append([piece])
    # A later intermediate fragment may bridge two previously separated groups.
    changed = True
    while changed:
        changed = False
        for a, b in combinations(range(len(groups)), 2):
            if proposal(groups[a], groups[b]) is not None:
                groups[a].extend(groups.pop(b))
                changed = True
                break
    lines = [_build_line(group) for group in groups]
    for index, line in enumerate(lines, 1):
        line.line_id = index
    return lines


def line_intersection(first, second):
    denominator = _cross(first.fitted_direction, second.fitted_direction)
    if abs(denominator) < 1e-8:
        return None
    offset = second.fitted_origin - first.fitted_origin
    return (first.fitted_origin + _cross(offset, second.fitted_direction)
            / denominator * first.fitted_direction)


def _arc(boundary, start, end):
    count = len(boundary.points)
    return (start + np.arange((end - start) % count + 1)) % count


def _local_end(line, piece, end):
    """Project the actual local observation endpoint on the final group fit."""
    point = piece.end if end else piece.start
    return line.fitted_origin + np.dot(point - line.fitted_origin, line.fitted_direction) * line.fitted_direction


def extract_corners(result, config):
    group_for = {id(piece): line for line in result.lines for piece in line.members
                 if line.observed_length >= config.minimum_edge
                 and line.residual_px <= config.fit_tolerance_px}
    minimum = config.minimum_edge
    tolerance = config.fit_tolerance_px
    height, width = result.image_shape
    candidates = []

    def inside(point):
        if config.point_geometry is not None:
            point = config.point_geometry.to_pixels([point])[0]
        if not np.isfinite(point).all():
            return False
        return (1 < point[0] < width - 2 and 1 < point[1] < height - 2)

    def consider(previous, following, connected, arc_length=0.):
        first, second = group_for[id(previous)], group_for[id(following)]
        if first is second:
            return
        if previous.component_index != following.component_index and connected:
            return
        # Preserve contour orientation and reject concave occluder turns.
        before = _unit(previous.end - previous.start)
        after = _unit(following.end - following.start)
        orientation = result.boundaries[previous.component_index].orientation
        if _cross(before, after) * orientation <= 0:
            return
        point = line_intersection(first, second)
        if point is None or not inside(point):
            return
        endpoints = (_local_end(first, previous, True), _local_end(second, following, False))
        far = (_local_end(first, previous, False), _local_end(second, following, True))
        rays = [_unit(p - point) for p in far]
        if any(np.linalg.norm(p - point) < 2. for p in far):
            return
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(*rays), -1., 1.))))
        # A 12 px noisy fragment does not establish a shallow physical turn.
        # Long supports still permit highly acute/obtuse perspective corners.
        # The acceptance tolerance is a ceiling, not measured noise. Estimate
        # direction uncertainty from each fit's observed residual, with a
        # half-pixel quantisation floor, rather than penalising every clean
        # short edge as if it had the maximum permissible error.
        angular_noise = sum(np.rad2deg(np.arctan2(2. * max(.5, line.residual_px),
                                                  max(2., line.observed_length)))
                            for line in (first, second))
        minimum_turn = config.minimum_turn_deg + angular_noise
        if not minimum_turn <= angle <= 180. - minimum_turn:
            return
        distances = [float(np.linalg.norm(p - point)) for p in endpoints]
        direct = (connected and arc_length <= max(2. * minimum, 4. * tolerance)
                  and max(distances) <= 4. * tolerance + arc_length)
        if not direct:
            if connected or not config.infer_corners or not all(inside(p) for p in endpoints):
                return
            # The intersection must extend the open ends; never cross the middle.
            extensions = []
            for local_end, local_far in zip(endpoints, far):
                outward = _unit(local_end - local_far)
                extension = float(np.dot(point - local_end, outward))
                if extension < -2. * tolerance:
                    return
                extensions.append(max(0., extension))
            limit = max(minimum, min(first.observed_length, second.observed_length))
            if max(extensions) > limit or sum(distances) > 2. * limit:
                return
            if np.linalg.norm(endpoints[0] - endpoints[1]) < 2.:
                return
        score = min(first.support, second.support) / (1. + sum(distances) + first.residual_px + second.residual_px)
        candidates.append((not direct, -score, point, angle, first, second))

    active_by_component = []
    for component_index, boundary in enumerate(result.boundaries):
        pieces = sorted((p for p in result.pieces if p.component_index == component_index
                         and id(p) in group_for), key=lambda p: p.start_index)
        active_by_component.append(pieces)
        if len(pieces) < 2:
            continue
        for index, previous in enumerate(pieces):
            following = pieces[(index + 1) % len(pieces)]
            indices = _arc(boundary, previous.end_index, following.start_index)
            connected = bool(boundary.valid[indices].all())
            arc_length = float(np.linalg.norm(np.diff(boundary.points[indices], axis=0), axis=1).sum())
            # One failed sample on a <=2*tolerance pixel turn is sampling
            # uncertainty, not evidence of a sizeable occluder. It remains
            # absent from the green observations and is never filled in.
            if (not connected and np.count_nonzero(~boundary.valid[indices]) <= 1
                    and arc_length <= 2. * tolerance):
                connected = True
            consider(previous, following, connected, arc_length)

        # A short bevel/noisy transition may have no reliable corner on either
        # end. Try its two strong neighbours, but only if neither local junction
        # was accepted. This does not require a complete rectangle or a view.
        for index, middle in enumerate(pieces):
            previous, following = pieces[index - 1], pieces[(index + 1) % len(pieces)]
            mid_line = group_for[id(middle)]
            first, second = group_for[id(previous)], group_for[id(following)]
            if (len({id(first), id(mid_line), id(second)}) < 3
                    or mid_line.observed_length > max(2. * minimum, 4. * tolerance)
                    or min(first.observed_length, second.observed_length) < 2. * mid_line.observed_length):
                continue
            if any((a is mid_line or b is mid_line) for _, _, _, _, a, b in candidates):
                continue
            indices = _arc(boundary, previous.end_index, following.start_index)
            connected = bool(boundary.valid[indices].all())
            arc_length = float(np.linalg.norm(np.diff(boundary.points[indices], axis=0), axis=1).sum())
            consider(previous, following, connected, arc_length)

    # Across components, only open ends may participate, in contour order.
    for a, b in combinations(range(len(active_by_component)), 2):
        for first in active_by_component[a]:
            for second in active_by_component[b]:
                for previous, following in ((first, second), (second, first)):
                    before = result.boundaries[previous.component_index]
                    after = result.boundaries[following.component_index]
                    if (before.valid[(previous.end_index + 1) % len(before.valid)]
                            or after.valid[(following.start_index - 1) % len(after.valid)]):
                        continue
                    consider(previous, following, False)

    accepted = []
    radius = max(3., min(10., .5 * minimum))
    for inferred, _, point, angle, first, second in sorted(candidates, key=lambda c: c[:2]):
        if any(np.linalg.norm(point - old) <= radius for old in accepted):
            continue
        output_point = (config.point_geometry.to_pixels([point])[0]
                        if config.point_geometry is not None else point)
        rounded = np.rint(output_point).astype(np.int32)
        accepted.append(point)
        (result.inferred if inferred else result.visible).append((rounded, angle))
        if config.point_geometry is not None:
            supports = []
            for line in (first, second):
                endpoints = config.point_geometry.to_pixels([line.fitted_start, line.fitted_end])
                supports.append(PixelLineSupport(*endpoints))
            result.corner_supports[tuple(map(int, rounded))] = tuple(supports)
        else:
            result.corner_supports[tuple(map(int, rounded))] = (first, second)


def detect_boundaries(contours, white_mask, purple_mask, config=None):
    config = config or BoundaryConfig()
    result = BoundaryResult(config=config)
    result.image_shape = white_mask.shape[:2]
    if purple_mask is None:
        purple_mask = np.zeros_like(white_mask)
        cv2.drawContours(purple_mask, contours, -1, 255, cv2.FILLED)
    for index, contour in enumerate(contours):
        boundary = observe_boundary(contour, purple_mask, white_mask, config)
        result.boundaries.append(boundary)
        result.pieces.extend(split_boundary(boundary, index, config))
    result.lines = merge_boundary_pieces(result.pieces, result.boundaries, config)
    extract_corners(result, config)
    return result


def draw_boundary_result(image, result, *, fitted=True, labels=True):
    for boundary in result.boundaries:
        for indices, closed in _runs(boundary.valid):
            if len(indices) >= 2:
                cv2.polylines(image, [boundary.raw_points[indices].astype(np.int32).reshape(-1, 1, 2)],
                              closed, (0, 255, 0), 2, cv2.LINE_AA)
    if fitted:
        for line in result.lines:
            if line.observed_length < result.config.minimum_edge:
                continue
            points = np.linspace(line.fitted_start, line.fitted_end, 32)
            anchor = max(line.members, key=lambda p: p.length).fitted_origin
            if result.config.point_geometry is not None:
                points = result.config.point_geometry.to_pixels(points)
                anchor = result.config.point_geometry.to_pixels([anchor])[0]
            cv2.polylines(image, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                          False, (0, 0, 255), 1, cv2.LINE_AA)
            anchor = np.rint(anchor).astype(int)
            if labels:
                cv2.putText(image, f"L{line.line_id}", tuple(anchor + (3, -4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
