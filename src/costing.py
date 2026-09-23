"""Stage 2: detour-time costing and ranking (FR5, FR6).

    detour = (t_to + t_from) - direct_time_s + penalty

t_to is the time from the current position to the candidate, t_from the time
from the candidate to the destination, and direct_time_s the time of the
direct trip. penalty is a fixed U-turn cost applied when the candidate sits
behind the driver's current position along the route.

Travel times come from a swappable backend:
  * OSRMBackend      - a live OSRM routing engine (/route/v1/driving/).
  * HeuristicBackend - a deterministic offline two-speed estimate, used by the
                       unit tests and the benchmark so they are reproducible
                       without network access (NFR3).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import net
from corridor import CorridorCandidate
from geometry import LatLon, haversine_m

try:
    from typing import Protocol
except ImportError:  # Python < 3.8
    Protocol = object

UTURN_PENALTY_S = 90.0
DEFAULT_OSRM_URL = "https://router.project-osrm.org"


class CostBackend(Protocol):
    """Anything that can say how long it takes to drive from a to b."""

    name: str

    def route_time_s(self, a: LatLon, b: LatLon) -> float:
        ...


class HeuristicBackend:
    """Deterministic offline estimate of driving time.

    Hops shorter than ``short_hop_m`` are assumed to run on local streets at
    ``side_street_kmh``; longer hops are assumed to include arterial running
    at ``arterial_kmh``. Every leg also pays a fixed junction penalty for the
    time lost at intersections.
    """

    name = "offline heuristic"

    def __init__(self, short_hop_m: float = 800.0, side_street_kmh: float = 30.0,
                 arterial_kmh: float = 80.0, junction_penalty_s: float = 20.0):
        self.short_hop_m = short_hop_m
        self.side_street_kmh = side_street_kmh
        self.arterial_kmh = arterial_kmh
        self.junction_penalty_s = junction_penalty_s

    def route_time_s(self, a: LatLon, b: LatLon) -> float:
        d = haversine_m(a, b)
        speed_kmh = self.side_street_kmh if d < self.short_hop_m else self.arterial_kmh
        return d / (speed_kmh / 3.6) + self.junction_penalty_s


class OSRMBackend:
    """Driving time from a live OSRM instance."""

    name = "live OSRM"

    def __init__(self, base_url: str = DEFAULT_OSRM_URL, profile: str = "driving", timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout_s = timeout_s
        self._cache: Dict[Tuple[LatLon, LatLon], float] = {}

    def route_time_s(self, a: LatLon, b: LatLon) -> float:
        key = (a, b)
        if key not in self._cache:
            url = "%s/route/v1/%s/%.6f,%.6f;%.6f,%.6f" % (self.base_url, self.profile, a[1], a[0], b[1], b[0])
            data = net.get_json(url, params={"overview": "false"}, timeout_s=self.timeout_s)
            if data.get("code") != "Ok" or not data.get("routes"):
                raise RuntimeError("OSRM could not route between %s and %s: %s" % (a, b, data.get("code")))
            self._cache[key] = float(data["routes"][0]["duration"])
        return self._cache[key]

    def prefetch(self, points: Sequence[LatLon], chunk: int = 90) -> None:
        """Fill the cache with every pair among ``points`` using OSRM's /table service.

        One table request replaces many single routes. The first two points
        (normally the vehicle and the destination) go into every chunk, so every
        candidate gets its time from the vehicle and to the destination.
        """
        unique = list(dict.fromkeys(points))
        head, rest = unique[:2], unique[2:]
        for i in range(0, max(len(rest), 1), chunk):
            pts = head + rest[i:i + chunk]
            coords = ";".join("%.6f,%.6f" % (p[1], p[0]) for p in pts)
            data = net.get_json("%s/table/v1/%s/%s" % (self.base_url, self.profile, coords),
                                params={"annotations": "duration"}, timeout_s=self.timeout_s * 2)
            if data.get("code") != "Ok":
                raise RuntimeError("OSRM table request failed: %s" % data.get("code"))
            for a, row in zip(pts, data["durations"]):
                for b, t in zip(pts, row):
                    if t is not None:
                        self._cache[(a, b)] = float(t)


@dataclass
class DetourResult:
    """Detour cost of one candidate, and where the numbers came from."""

    candidate: CorridorCandidate
    t_to_s: float        # current position -> candidate
    t_from_s: float      # candidate -> destination
    direct_s: float      # current position -> destination
    penalty_s: float     # U-turn penalty (0 unless the candidate is behind)
    detour_s: float      # extra time the stop adds to the trip
    source: str          # which backend produced the times

    @property
    def detour_min(self) -> float:
        return self.detour_s / 60.0


class DetourCostEngine:
    """Rank candidates by the extra travel time they add to the trip."""

    def __init__(self, backend: CostBackend, uturn_penalty_s: float = UTURN_PENALTY_S):
        self.backend = backend
        self.uturn_penalty_s = uturn_penalty_s

    def evaluate(self, origin: LatLon, destination: LatLon, candidate: CorridorCandidate,
                 progress_m: float = 0.0, direct_s: Optional[float] = None) -> DetourResult:
        """Detour cost of one candidate for a trip from origin to destination."""
        if direct_s is None:
            direct_s = self.backend.route_time_s(origin, destination)
        t_to = self.backend.route_time_s(origin, candidate.poi.coord)
        t_from = self.backend.route_time_s(candidate.poi.coord, destination)
        penalty = self.uturn_penalty_s if candidate.along_track_m - progress_m < 0 else 0.0
        detour = (t_to + t_from) - direct_s + penalty
        return DetourResult(candidate, t_to, t_from, direct_s, penalty, detour, self.backend.name)

    def rank(self, origin: LatLon, destination: LatLon, candidates: Sequence[CorridorCandidate],
             progress_m: float = 0.0) -> List[DetourResult]:
        """All candidates costed and sorted by ascending detour time."""
        if not candidates:
            return []
        direct_s = self.backend.route_time_s(origin, destination)
        results = [self.evaluate(origin, destination, c, progress_m, direct_s) for c in candidates]
        results.sort(key=lambda r: r.detour_s)
        return results
