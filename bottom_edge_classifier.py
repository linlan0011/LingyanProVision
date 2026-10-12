"""Experimental layer labels applied AFTER all visible/inferred corners exist.

Implements the user's farthest/connected/nearest/edge-length heuristic. These
are layer hypotheses, not a proof of physical height or map identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class BottomEdgeConfig:
    length_mm: float = 660.
    width_mm: float = 360.
    tolerance_mm: float = 30.
    ground_z_mm: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.length_mm, self.width_mm)):
            raise ValueError('bottom-edge dimensions must be positive and finite')
        if not math.isfinite(self.tolerance_mm) or self.tolerance_mm < 0:
            raise ValueError('bottom-edge tolerance must be nonnegative and finite')
        if not math.isfinite(self.ground_z_mm):
            raise ValueError('ground-plane height must be finite')


@dataclass
class LayerCorner:
    pixel: tuple[int, int]
    inferred: bool
    range_mm: float | None
    layer: str = 'unknown'
    reason: str = 'not_classified'
    distance_to_seed_mm: float | None = None
    matched_edge: str | None = None
    edge_error_mm: float | None = None


@dataclass
class BottomEdgeResult:
    config: BottomEdgeConfig
    status: str
    candidate_count: int = 0
    corners: list[LayerCorner] = field(default_factory=list)
    upper_seed: tuple[int, int] | None = None
    lower_seed: tuple[int, int] | None = None
    # Only adjacent accepted corners on the SAME validated fitted line.
    connections: list[tuple[tuple[int, int], tuple[int, int], int]] = field(default_factory=list)
    eligible_count: int = 0

    @property
    def by_pixel(self):
        return {corner.pixel: corner for corner in self.corners}

    @property
    def upper_count(self):
        return sum(c.layer == 'upper' for c in self.corners)

    @property
    def lower_count(self):
        return sum(c.layer in ('lower_seed', 'lower') for c in self.corners)

    def payload(self):
        data = asdict(self)
        data.update(schema='safety.bottom_edge_test.v1',
                    method='farthest_one_hop_nearest_edge_length',
                    range_interpretation='all points projected to the stated ground plane',
                    layer_labels_are_hypotheses=True,
                    map_identity_verified=False,
                    lower_seed_has_edge_length_match=any(c.layer == 'lower' for c in self.corners),
                    upper_count=self.upper_count, lower_count=self.lower_count)
        return data


def _pixel(pixel):
    return tuple(map(int, np.rint(pixel)))


def direct_corner_connections(pixels, boundary_result):
    """One-hop graph using accepted support IDs, never image proximity alone.

    A fitted line may bridge gaps or extend toward an inferred corner. When
    three corners use that line, connect consecutive corners only. Do not
    recursively propagate the upper label around the whole outline.
    """
    if boundary_result is None:
        return []
    points = sorted(set(map(_pixel, pixels)))
    geometry = boundary_result.config.point_geometry
    coordinates = np.asarray(points, float).reshape(-1, 2)
    if geometry is not None and points:
        coordinates = geometry.to_geometry(coordinates)
    position = dict(zip(points, coordinates))
    members = {}
    for point in points:
        for support in boundary_result.corner_supports.get(point, ()):
            line_id = getattr(support, 'line_id', 0)
            if line_id:
                members.setdefault(line_id, set()).add(point)
    connections = []
    for line in boundary_result.lines:
        group = members.get(line.line_id, ())
        ordered = sorted(group, key=lambda p: float(
            (position[p] - line.fitted_origin) @ line.fitted_direction))
        connections.extend((first, second, line.line_id)
                           for first, second in zip(ordered[:-1], ordered[1:]))
    return connections


def classify_bottom_edges(readings, boundary_result, config=None):
    """Keep every input candidate; only the layer-test working set is filtered."""
    config = config or BottomEdgeConfig()
    # A duplicate visible/inferred pixel is one candidate; prefer the visible.
    unique = {}
    for reading in sorted(readings, key=lambda r: r.inferred):
        unique.setdefault(_pixel(reading.pixel), reading)
    result = BottomEdgeResult(config, 'no_candidates', len(unique))
    result.corners = [LayerCorner(point, r.inferred, r.distance_mm,
                                 reason='not_classified')
                      for point, r in unique.items()]
    if not unique:
        return result
    # A known out-of-range corner is deliberately excluded from this working
    # set, but remains in the detector output. Unknown ranging failures still
    # prevent ranking a potentially incomplete set.
    for corner in result.corners:
        if not unique[corner.pixel].valid:
            corner.reason = unique[corner.pixel].reason or 'unmeasurable'
    unique = {point: r for point, r in unique.items()
              if r.valid or r.reason != 'beyond_distance_limit'}
    result.eligible_count = sum(r.valid for r in unique.values())
    if not unique:
        result.status = 'no_in_range_candidates'
        return result
    if any(not r.valid or r.distance_mm is None
           or not all(v is not None and math.isfinite(v)
                      for v in (r.x_mm, r.y_mm, r.distance_mm))
           or r.z_mm is None or not math.isclose(r.z_mm, config.ground_z_mm, abs_tol=1e-5)
           for r in unique.values()):
        result.status = 'unmeasurable_or_wrong_plane'
        for corner in result.corners:
            reading = unique.get(corner.pixel)
            if reading is not None:
                corner.reason = reading.reason or result.status
        return result
    by_pixel = result.by_pixel
    if len(unique) < 3:
        nearest = min(unique, key=lambda p: (unique[p].distance_mm, p))
        result.lower_seed = nearest
        by_pixel[nearest].layer = 'lower_seed'
        by_pixel[nearest].reason = 'nearest_without_three_candidates'
        result.status = 'nearest_fallback'
        return result
    # Deterministic pixel order only resolves exact range ties; it is not
    # evidence that one tied point belongs to a different physical layer.
    farthest = max(unique, key=lambda p: (unique[p].distance_mm, p))
    result.upper_seed = farthest
    result.connections = direct_corner_connections(unique, boundary_result)
    upper = {farthest}
    for first, second, _ in result.connections:
        if first == farthest:
            upper.add(second)
        elif second == farthest:
            upper.add(first)
    for point in upper:
        by_pixel[point].layer = 'upper'
        by_pixel[point].reason = ('farthest_hypothesis' if point == farthest
                                  else 'directly_connected_to_farthest')
    remaining = [point for point in unique if point not in upper]
    if not remaining:
        result.status = 'no_remaining_candidates'
        return result
    nearest = min(remaining, key=lambda p: (unique[p].distance_mm, p))
    result.lower_seed = nearest
    by_pixel[nearest].layer = 'lower_seed'
    by_pixel[nearest].reason = 'nearest_remaining_hypothesis'
    seed = unique[nearest]
    for point in remaining:
        if point == nearest:
            continue
        reading = unique[point]
        distance = math.hypot(reading.x_mm - seed.x_mm, reading.y_mm - seed.y_mm)
        error, edge = min((abs(distance - config.length_mm), 'length'),
                          (abs(distance - config.width_mm), 'width'))
        corner = by_pixel[point]
        corner.distance_to_seed_mm = distance
        corner.edge_error_mm = error
        if error <= config.tolerance_mm:
            corner.layer = 'lower'
            corner.reason = 'bottom_edge_length_match'
            corner.matched_edge = edge
        else:
            corner.reason = 'bottom_edge_length_mismatch'
    result.status = 'classified'
    return result
