"""Geometry primitives for SmartRoute.

No third-party geospatial library is used (NFR4). Absolute distances use the
haversine great-circle formula. Point-to-segment projection is done in a local
equirectangular planar frame centred on the start of the route, which over a
single city trip (a few tens of kilometres) introduces an error well under
half a percent.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

LatLon = Tuple[float, float]

EARTH_RADIUS_M = 6371008.8


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def route_length_m(route: Sequence[LatLon]) -> float:
    """Total length of a polyline in metres (sum of its segment lengths)."""
    return sum(haversine_m(route[i], route[i + 1]) for i in range(len(route) - 1))


def to_local_xy(origin: LatLon, point: LatLon) -> Tuple[float, float]:
    """Project a point into a flat (x east, y north) frame in metres centred on origin."""
    lat0 = math.radians(origin[0])
    x = math.radians(point[1] - origin[1]) * EARTH_RADIUS_M * math.cos(lat0)
    y = math.radians(point[0] - origin[0]) * EARTH_RADIUS_M
    return x, y


def from_local_xy(origin: LatLon, xy: Tuple[float, float]) -> LatLon:
    """Inverse of ``to_local_xy``."""
    lat0 = math.radians(origin[0])
    lat = origin[0] + math.degrees(xy[1] / EARTH_RADIUS_M)
    lon = origin[1] + math.degrees(xy[0] / (EARTH_RADIUS_M * math.cos(lat0)))
    return lat, lon


def point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float],
                     clamp_start: bool = True) -> Tuple[float, float, Tuple[float, float]]:
    """Project planar point p onto segment a-b.

    Returns (t, distance, foot) where t is the position along the segment as a
    fraction of its length (0 at a, 1 at b), distance is the perpendicular
    distance in metres and foot is the projected point. With
    ``clamp_start=False`` t may go below 0, extending the segment backwards
    beyond a. A zero-length segment is handled explicitly by treating a as the
    projection point.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return 0.0, math.hypot(p[0] - a[0], p[1] - a[1]), a
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg2
    t = min(1.0, t) if not clamp_start else max(0.0, min(1.0, t))
    foot = (a[0] + t * dx, a[1] + t * dy)
    return t, math.hypot(p[0] - foot[0], p[1] - foot[1]), foot


@dataclass
class SegmentProjection:
    """Where a point sits relative to a route."""

    segment_index: int       # index of the closest route segment
    cross_track_m: float     # perpendicular distance from the route
    along_track_m: float     # distance along the route from its start; negative = before the start
    point: LatLon            # the projected point on the route


def nearest_projection(route: Sequence[LatLon], point: LatLon) -> SegmentProjection:
    """
    Find the closest point on the whole polyline `route` to `point`.

    This is the core primitive behind the Smart Corridor filter: instead
    of measuring distance-from-vehicle (a circle), we measure
    distance-from-route (a corridor that follows the direction of travel).

    The first segment is extended backwards past the start of the route, so a
    point behind the start gets a negative along-track position rather than
    collapsing onto the start itself.
    """
    if len(route) < 2:
        raise ValueError("a route needs at least two coordinates")
    origin = route[0]
    p = to_local_xy(origin, point)
    best = None
    travelled = 0.0
    for i in range(len(route) - 1):
        a = to_local_xy(origin, route[i])
        b = to_local_xy(origin, route[i + 1])
        seg_len = haversine_m(route[i], route[i + 1])
        t, dist, foot = point_to_segment(p, a, b, clamp_start=(i > 0))
        if best is None or dist < best.cross_track_m:
            best = SegmentProjection(i, dist, travelled + t * seg_len, from_local_xy(origin, foot))
        travelled += seg_len
    return best


def point_along(route: Sequence[LatLon], distance_m: float) -> LatLon:
    """The point ``distance_m`` metres along a route (clamped to its ends)."""
    if distance_m <= 0:
        return route[0]
    travelled = 0.0
    for a, b in zip(route, route[1:]):
        seg = haversine_m(a, b)
        if seg > 0 and travelled + seg >= distance_m:
            t = (distance_m - travelled) / seg
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        travelled += seg
    return route[-1]
