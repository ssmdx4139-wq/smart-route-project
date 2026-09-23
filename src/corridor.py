"""Stage 1: the Smart Corridor geometric filter (FR3, FR4).

A radius search draws a circle around the vehicle. The Smart Corridor instead
keeps only points of interest that sit close to the route itself and are not
meaningfully behind the driver's current progress along it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from geometry import LatLon, nearest_projection, route_length_m

DEFAULT_CORRIDOR_WIDTH_M = 150.0
DEFAULT_BEHIND_TOLERANCE_M = 250.0


@dataclass
class POI:
    """A candidate point of interest."""

    id: str
    name: str
    category: str
    lat: float
    lon: float
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def coord(self) -> LatLon:
        return (self.lat, self.lon)


@dataclass
class CorridorCandidate:
    """A POI together with where it sits relative to the route."""

    poi: POI
    cross_track_m: float     # perpendicular distance from the route
    along_track_m: float     # distance along the route from its start
    remaining_m: float       # route length left from that point to the destination


class SmartCorridor:
    """Filter POIs by distance from the route and position along it."""

    def __init__(self, route: Sequence[LatLon], corridor_width_m: float = DEFAULT_CORRIDOR_WIDTH_M,
                 behind_tolerance_m: float = DEFAULT_BEHIND_TOLERANCE_M):
        if len(route) < 2:
            raise ValueError("SmartCorridor needs a route with at least two coordinates")
        if corridor_width_m < 0:
            raise ValueError("corridor_width_m must not be negative")
        self.route = list(route)
        self.corridor_width_m = float(corridor_width_m)
        self.behind_tolerance_m = float(behind_tolerance_m)
        self.length_m = route_length_m(self.route)

    def project(self, poi: POI) -> CorridorCandidate:
        """Locate a POI relative to the route without filtering it."""
        proj = nearest_projection(self.route, poi.coord)
        return CorridorCandidate(poi, proj.cross_track_m, proj.along_track_m,
                                 max(0.0, self.length_m - proj.along_track_m))

    def is_inside(self, candidate: CorridorCandidate, progress_m: float = 0.0) -> bool:
        """True if the candidate is inside the corridor and not meaningfully behind."""
        return (candidate.cross_track_m <= self.corridor_width_m
                and candidate.along_track_m >= progress_m - self.behind_tolerance_m)

    def filter(self, pois: Sequence[POI], progress_m: float = 0.0) -> List[CorridorCandidate]:
        """Keep POIs inside the corridor, sorted by progress along the route.

        ``progress_m`` is how far along the route the vehicle already is. A POI
        is kept when its cross-track distance is within the corridor width and
        its along-track position is no more than ``behind_tolerance_m`` behind
        the vehicle.
        """
        survivors = [c for c in (self.project(p) for p in pois) if self.is_inside(c, progress_m)]
        survivors.sort(key=lambda c: c.along_track_m)
        return survivors

    def split(self, pois: Sequence[POI], progress_m: float = 0.0):
        """(survivors, rejected) - every POI projected, for display on the map."""
        kept: List[CorridorCandidate] = []
        rejected: List[CorridorCandidate] = []
        for c in (self.project(p) for p in pois):
            (kept if self.is_inside(c, progress_m) else rejected).append(c)
        kept.sort(key=lambda c: c.along_track_m)
        return kept, rejected


def corridor_reason(candidate: CorridorCandidate, corridor: SmartCorridor,
                    progress_m: float = 0.0) -> Optional[str]:
    """Human-readable reason a candidate was excluded, or None if it survived."""
    if candidate.along_track_m < progress_m - corridor.behind_tolerance_m:
        return "behind the driver"
    if candidate.cross_track_m > corridor.corridor_width_m:
        return "outside the corridor"
    return None
