"""Control group: a conventional radius-based search (FR10).

This reproduces the "search along route" behaviour of mainstream navigation
apps: draw a circle around the vehicle's current position and rank every
point of interest inside it by straight-line distance, with no notion of the
route's direction at all.
"""

from typing import List, Sequence, Tuple

from corridor import POI
from geometry import LatLon, haversine_m

DEFAULT_RADIUS_M = 5000.0


def radius_search(pois: Sequence[POI], position: LatLon, radius_m: float = DEFAULT_RADIUS_M) -> List[Tuple[POI, float]]:
    """POIs within ``radius_m`` of ``position``, nearest first, as (poi, distance_m) pairs."""
    hits = [(p, haversine_m(position, p.coord)) for p in pois]
    hits = [h for h in hits if h[1] <= radius_m]
    hits.sort(key=lambda h: h[1])
    return hits
