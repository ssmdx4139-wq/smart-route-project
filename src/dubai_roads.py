"""An offline, approximate network of Dubai's main roads (FR1 fallback).

When live OSRM routing is off or unreachable, the dashboard still needs a
route that follows roads between any two places in Dubai. This module holds a
small hand-built graph of the main corridors (Sheikh Zayed Road, Al Khail
Road, Sheikh Mohammed bin Zayed Road, Al Ain Road, Hessa Street, Umm Suqeim
Road, Ras Al Khor Road, Airport Road and a few connectors) and routes over it
with Dijkstra's algorithm, minimising travel time.

It also generates the offline sample stops for those roads, so a trip
anywhere on the network has candidates to filter. Sheikh Zayed Road is left
out of the generator because it already carries the curated sample in
data_loader.SAMPLE_DUBAI_POIS.

The coordinates are approximate (usually within a few hundred metres of the
real carriageway) and the stops are illustrative sample positions, not
surveyed businesses. Live mode replaces both with OpenStreetMap data.
"""

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from corridor import POI
from geometry import LatLon, from_local_xy, haversine_m, point_along, point_to_segment, to_local_xy

NODES: Dict[str, LatLon] = {
    # Sheikh Zayed Road - the same points as data_loader.SAMPLE_DUBAI_ROUTE
    "ibn": (25.0443, 55.1170), "marina": (25.0700, 55.1400), "dic": (25.0960, 55.1640),
    "moe": (25.1180, 55.1990), "szr_quoz": (25.1450, 55.2250), "safa": (25.1760, 55.2490),
    "downtown": (25.2040, 55.2700), "trade": (25.2270, 55.2860), "zaabeel": (25.2400, 55.3100),
    "garhoud": (25.2540, 55.3230), "rigga": (25.2650, 55.3230),
    # Al Khail Road
    "ak_jvc": (25.0560, 55.2030), "ak_hessa": (25.0760, 55.2150), "ak_ums": (25.1030, 55.2350),
    "ak_quoz": (25.1380, 55.2560), "ak_meydan": (25.1640, 55.2780), "ak_bb": (25.1830, 55.2920),
    "ak_rak": (25.1960, 55.3230),
    # Sheikh Mohammed bin Zayed Road
    "mbz_dip": (24.9950, 55.1750), "mbz_garn": (25.0150, 55.2150), "mbz_motor": (25.0350, 55.2580),
    "mbz_ranches": (25.0650, 55.3000), "mbz_warsan": (25.0950, 55.3400), "mbz_dso": (25.1150, 55.3700),
    "mbz_mizhar": (25.1650, 55.4100), "mbz_mirdif": (25.2050, 55.4300),
    # other junctions and places
    "garn_1": (25.0450, 55.1700), "hessa_1": (25.0860, 55.1880), "motor": (25.0460, 55.2400),
    "ums_1": (25.1120, 55.2150), "dhills": (25.1050, 55.2500), "ums_2": (25.0900, 55.2750),
    "baysq": (25.1877, 55.2812), "difc": (25.2110, 55.2800), "karama": (25.2450, 55.3030),
    "alain_1": (25.1600, 55.3500), "dso": (25.1185, 55.3825), "rak_1": (25.2050, 55.3450),
    "festival": (25.2220, 55.3520), "festival_n": (25.2380, 55.3470), "deira_cc": (25.2521, 55.3337),
    "dxb": (25.2480, 55.3650), "airport_1": (25.2350, 55.3850), "mirdif": (25.2166, 55.4078),
    "nad_hamar": (25.2000, 55.3800), "jlt": (25.0690, 55.1420), "jvc": (25.0600, 55.2100),
    "dip": (25.0010, 55.1600),
}

# (road name, speed km/h, node chain). Every road is two-way.
ROADS: List[Tuple[str, float, List[str]]] = [
    ("Sheikh Zayed Rd", 90, ["ibn", "marina", "dic", "moe", "szr_quoz", "safa", "downtown", "trade", "zaabeel", "garhoud", "rigga"]),
    ("Al Khail Rd", 80, ["ak_jvc", "ak_hessa", "ak_ums", "ak_quoz", "ak_meydan", "ak_bb", "ak_rak"]),
    ("Sheikh Mohammed bin Zayed Rd", 90, ["mbz_dip", "mbz_garn", "mbz_motor", "mbz_ranches", "mbz_warsan", "mbz_dso", "mbz_mizhar", "mbz_mirdif"]),
    ("Garn Al Sabkha St", 70, ["marina", "garn_1", "mbz_garn"]),
    ("Hessa St", 70, ["dic", "hessa_1", "ak_hessa", "motor", "mbz_motor"]),
    ("Umm Suqeim Rd", 70, ["moe", "ums_1", "ak_ums", "dhills", "ums_2", "mbz_ranches"]),
    ("Al Manara St", 60, ["szr_quoz", "ak_quoz"]),
    ("Al Meydan Rd", 60, ["safa", "ak_meydan"]),
    ("Marasi Dr / Financial Centre Rd", 50, ["downtown", "baysq", "ak_bb"]),
    ("Financial Centre Rd", 50, ["downtown", "difc", "trade"]),
    ("Sheikh Rashid Rd", 60, ["zaabeel", "karama"]),
    ("Al Ain Rd", 90, ["ak_rak", "alain_1", "dso", "mbz_dso"]),
    ("Ras Al Khor Rd", 70, ["ak_rak", "rak_1", "nad_hamar", "mirdif"]),
    ("Festival City link", 50, ["rak_1", "festival", "festival_n", "deira_cc"]),
    ("Airport Rd", 60, ["garhoud", "deira_cc", "dxb", "airport_1", "mirdif", "mbz_mirdif"]),
    ("Al Asayel St", 50, ["ak_jvc", "jvc", "ak_hessa"]),
    ("JLT access", 40, ["marina", "jlt"]),
    ("Dubai Investments Park access", 50, ["dip", "mbz_dip"]),
]

ACCESS_KMH = 30.0       # the short hop between a place and the nearest road
MIN_ACCESS_M = 30.0     # shorter hops are not drawn


@dataclass
class Edge:
    to: str
    length_m: float
    time_s: float
    road: str


def _build_graph() -> Dict[str, List[Edge]]:
    graph: Dict[str, List[Edge]] = {k: [] for k in NODES}
    for road, kmh, chain in ROADS:
        for a, b in zip(chain, chain[1:]):
            d = haversine_m(NODES[a], NODES[b])
            t = d / (kmh / 3.6)
            graph[a].append(Edge(b, d, t, road))
            graph[b].append(Edge(a, d, t, road))
    return graph


GRAPH = _build_graph()


@dataclass
class NetworkRoute:
    coords: List[LatLon]
    distance_m: float
    duration_s: float
    steps: List[Tuple[str, float]]     # (road name, metres) in driving order


def _snap(point: LatLon) -> Tuple[str, str, float, LatLon, str]:
    """Nearest point on any road segment: (node a, node b, fraction along a-b, foot, road)."""
    best = None
    for road, _, chain in ROADS:
        for a, b in zip(chain, chain[1:]):
            pa, pb = NODES[a], NODES[b]
            t, dist, foot = point_to_segment(to_local_xy(pa, point), (0.0, 0.0), to_local_xy(pa, pb))
            if best is None or dist < best[0]:
                best = (dist, a, b, t, from_local_xy(pa, foot), road)
    _, a, b, t, foot, road = best
    return a, b, t, foot, road


def route(origin: LatLon, destination: LatLon) -> NetworkRoute:
    """Fastest route over the network between two points, with access hops at each end."""
    sa, sb, st, sfoot, sroad = _snap(origin)
    da, db, dt, dfoot, droad = _snap(destination)
    speeds = {road: kmh for road, kmh, _ in ROADS}

    # temporary nodes at the snapped points, joined to both ends of their segment
    graph = {k: list(v) for k, v in GRAPH.items()}
    extra = {"@s": sfoot, "@d": dfoot}
    graph["@s"], graph["@d"] = [], []
    for tmp, (a, b, road) in (("@s", (sa, sb, sroad)), ("@d", (da, db, droad))):
        for n in (a, b):
            d = haversine_m(extra[tmp], NODES[n])
            t = d / (speeds[road] / 3.6)
            graph[tmp].append(Edge(n, d, t, road))
            graph[n].append(Edge(tmp, d, t, road))
    if {sa, sb} == {da, db}:  # both on the same segment
        d = haversine_m(sfoot, dfoot)
        graph["@s"].append(Edge("@d", d, d / (speeds[sroad] / 3.6), sroad))

    coord = lambda n: extra.get(n) or NODES[n]
    best: Dict[str, float] = {"@s": 0.0}
    prev: Dict[str, Tuple[str, Edge]] = {}
    queue = [(0.0, "@s")]
    while queue:
        t, n = heapq.heappop(queue)
        if n == "@d":
            break
        if t > best.get(n, float("inf")):
            continue
        for e in graph[n]:
            nt = t + e.time_s
            if nt < best.get(e.to, float("inf")):
                best[e.to], prev[e.to] = nt, (n, e)
                heapq.heappush(queue, (nt, e.to))

    path_edges: List[Edge] = []
    n = "@d"
    while n != "@s":
        p, e = prev[n]
        path_edges.append(e)
        n = p
    path_edges.reverse()

    coords = [coord("@s")]
    steps: List[Tuple[str, float]] = []
    for e in path_edges:
        coords.append(coord(e.to))
        if steps and steps[-1][0] == e.road:
            steps[-1] = (e.road, steps[-1][1] + e.length_m)
        elif e.length_m > 1:
            steps.append((e.road, e.length_m))
    distance = sum(e.length_m for e in path_edges)
    duration = sum(e.time_s for e in path_edges)

    # access hops from the places themselves to the road
    start_hop, end_hop = haversine_m(origin, sfoot), haversine_m(dfoot, destination)
    if start_hop > MIN_ACCESS_M:
        coords.insert(0, origin)
        steps.insert(0, ("local streets", start_hop))
    if end_hop > MIN_ACCESS_M:
        coords.append(destination)
        steps.append(("local streets", end_hop))
    distance += start_hop + end_hop
    duration += (start_hop + end_hop) / (ACCESS_KMH / 3.6)
    return NetworkRoute(coords, distance, duration, steps)


# ---------------------------------------------------------------------------
# Offline sample stops along the network
# ---------------------------------------------------------------------------
_CYCLE = ["petrol station", "supermarket", "mosque", "pharmacy", "park"]
FIRST_STOP_M, STOP_SPACING_M = 900.0, 1700.0


def _offset_point(a: LatLon, b: LatLon, at: LatLon, offset_m: float) -> LatLon:
    """A point ``offset_m`` to the left (positive) or right (negative) of segment a-b at ``at``."""
    bx, by = to_local_xy(a, b)
    length = (bx * bx + by * by) ** 0.5 or 1.0
    ax, ay = to_local_xy(a, at)
    return from_local_xy(a, (ax - by / length * offset_m, ay + bx / length * offset_m))


def _nearest_place(point: LatLon, places: Dict[str, LatLon]) -> str:
    return min(places, key=lambda name: haversine_m(places[name], point))


def network_pois(places: Dict[str, LatLon]) -> List[POI]:
    """Deterministic sample stops along every road except Sheikh Zayed Road.

    Stops sit 30-120 m from the carriageway (parks 170-260 m, as they are set
    back from the road), alternating sides, with a side-street decoy 600-900 m
    away at every third position.
    """
    pois: List[POI] = []
    k = 0
    for road, _, chain in ROADS:
        if road == "Sheikh Zayed Rd":
            continue
        line = [NODES[n] for n in chain]
        seg_lengths = [haversine_m(a, b) for a, b in zip(line, line[1:])]
        total = sum(seg_lengths)
        pos = FIRST_STOP_M
        while pos < total - 300:
            at = point_along(line, pos)
            travelled, i = 0.0, 0
            while i < len(seg_lengths) - 1 and travelled + seg_lengths[i] < pos:
                travelled += seg_lengths[i]
                i += 1
            cat = _CYCLE[k % len(_CYCLE)]
            side = 1 if k % 2 == 0 else -1
            off = (170 + (k * 37) % 90) if cat == "park" else (30 + (k * 29) % 90)
            p = _offset_point(line[i], line[i + 1], at, side * off)
            area = _nearest_place(p, places)
            pois.append(POI("n%03d" % k, "%s - %s, near %s" % (cat.capitalize(), road, area), cat, p[0], p[1],
                            {"pattern": "convenient", "source": "offline network sample"}))
            if k % 3 == 2:
                q = _offset_point(line[i], line[i + 1], at, -side * (600 + (k * 53) % 300))
                pois.append(POI("n%03dx" % k, "%s - side street off %s, near %s" % (cat.capitalize(), road, _nearest_place(q, places)),
                                cat, q[0], q[1], {"pattern": "side_street", "source": "offline network sample"}))
            k += 1
            pos += STOP_SPACING_M
    return pois


def along_steps(steps: Sequence[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    """(road, start_m, end_m) for each step, measured along the route."""
    out, at = [], 0.0
    for name, length in steps:
        out.append((name, at, at + length))
        at += length
    return out


def road_at(steps: Sequence[Tuple[str, float]], distance_m: float) -> Optional[str]:
    for name, start, end in along_steps(steps):
        if start <= distance_m <= end:
            return name
    return None
