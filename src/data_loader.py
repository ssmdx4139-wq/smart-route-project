"""Data acquisition (FR1, FR2): live OpenStreetMap data plus an offline sample.

Live path
  * load_live_graph / route_from_graph - drivable road graph via OSMnx and a
    shortest path over it (FR1).
  * load_live_pois - points of interest for one category from the Overpass
    API (FR2).

Offline path
  * SAMPLE_DUBAI_ROUTE / SAMPLE_DUBAI_POIS - a small, hand-placed sample
    along the Sheikh Zayed Road corridor, from near Ibn Battuta in the
    south-west to Al Rigga in the north-east. It deliberately includes all
    three failure patterns from Section 1.2 of the report: stops conveniently
    ahead near the road, decoys behind the start of the route, and side-street
    decoys far off the main road. The names describe the area only; the
    coordinates are illustrative sample positions, not surveyed locations of
    real businesses.

osmnx, geopy and requests are imported only when a live function is called,
so the offline sample, the tests and the benchmark need nothing beyond the
Python standard library (NFR3, NFR4).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import dubai_roads
import net
from corridor import POI
from geometry import LatLon, haversine_m

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [OVERPASS_URL, "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter"]
OSRM_SERVERS = ["https://router.project-osrm.org", "https://routing.openstreetmap.de/routed-car"]
HTTP_HEADERS = {"User-Agent": "SmartRoute-MSc-project/1.0 (educational demo)"}

# Each supported category and the OpenStreetMap tag(s) that identify it.
CATEGORY_TAGS: Dict[str, List[Dict[str, str]]] = {
    "supermarket": [{"shop": "supermarket"}],
    "pharmacy": [{"amenity": "pharmacy"}],
    "petrol station": [{"amenity": "fuel"}],
    "mosque": [{"amenity": "place_of_worship", "religion": "muslim"}],
    "park": [{"leisure": "park"}],
}
CATEGORIES = list(CATEGORY_TAGS)

# Demonstration route along Sheikh Zayed Road (about 32.8 km).
SAMPLE_DUBAI_ROUTE: List[LatLon] = [
    (25.0443, 55.1170),   # near Ibn Battuta
    (25.0700, 55.1400),   # Jebel Ali / Dubai Marina interchange
    (25.0960, 55.1640),   # Dubai Internet City
    (25.1180, 55.1990),   # Mall of the Emirates interchange
    (25.1450, 55.2250),   # Al Quoz
    (25.1760, 55.2490),   # Al Safa
    (25.2040, 55.2700),   # Downtown / Business Bay
    (25.2270, 55.2860),   # Trade Centre
    (25.2400, 55.3100),   # Za'abeel
    (25.2540, 55.3230),   # Garhoud approach
    (25.2650, 55.3230),   # Al Rigga
]
SAMPLE_ORIGIN: LatLon = SAMPLE_DUBAI_ROUTE[0]
SAMPLE_DESTINATION: LatLon = SAMPLE_DUBAI_ROUTE[-1]

# pattern: convenient (near the road, ahead), behind (behind the start of the route),
# side_street (close in a straight line but far off the road), off_route (a few hundred
# metres back from the road), noise (far off the route).
SAMPLE_DUBAI_POIS: List[POI] = [
    POI("s01", "Supermarket - Al Barsha, on Sheikh Zayed Rd", "supermarket", 25.10447, 55.17869, {"pattern": "convenient"}),  # 70 m off route, 9.2 km along
    POI("s02", "Supermarket - Al Quoz service road", "supermarket", 25.13133, 55.21328, {"pattern": "convenient"}),  # 110 m off route, 13.8 km along
    POI("s03", "Supermarket - Jebel Ali, behind the start", "supermarket", 25.03918, 55.11293, {"pattern": "behind"}),  # 40 m off route, 0.7 km behind
    POI("s04", "Supermarket - The Greens residential pocket", "supermarket", 25.07798, 55.13753, {"pattern": "side_street"}),  # 760 m off route
    POI("s05", "Supermarket - Al Wasl back streets", "supermarket", 25.17586, 55.23313, {"pattern": "noise"}),  # 1.3 km off route
    POI("s06", "Pharmacy - Dubai Internet City", "pharmacy", 25.08636, 55.15587, {"pattern": "convenient"}),  # 59 m off route, 6.1 km along
    POI("s07", "Pharmacy - Business Bay, on Sheikh Zayed Rd", "pharmacy", 25.18202, 55.25460, {"pattern": "convenient"}),  # 90 m off route, 20.8 km along
    POI("s08", "Pharmacy - Ibn Battuta, behind the start", "pharmacy", 25.04006, 55.11410, {"pattern": "behind"}),  # 70 m off route, 0.6 km behind
    POI("s09", "Pharmacy - Dubai Marina side street", "pharmacy", 25.07200, 55.13131, {"pattern": "side_street"}),  # 820 m off route
    POI("s10", "Pharmacy - Al Safa villas", "pharmacy", 25.15991, 55.23230, {"pattern": "off_route"}),  # 350 m off route
    POI("s11", "Petrol station - Sheikh Zayed Rd, Al Barsha", "petrol station", 25.10158, 55.17366, {"pattern": "convenient"}),  # 45 m off route, 8.6 km along
    POI("s12", "Petrol station - Trade Centre", "petrol station", 25.20824, 55.27389, {"pattern": "convenient"}),  # 80 m off route, 24.3 km along
    POI("s13", "Petrol station - Jebel Ali, behind the start", "petrol station", 25.03767, 55.11184, {"pattern": "behind"}),  # 61 m off route, 0.9 km behind
    POI("s14", "Petrol station - JLT cluster roads", "petrol station", 25.06503, 55.12661, {"pattern": "side_street"}),  # 700 m off route
    POI("s15", "Petrol station - Al Khail Rd, Al Quoz", "petrol station", 25.12726, 55.22900, {"pattern": "noise"}),  # 1.6 km off route
    POI("s16", "Mosque - Al Sufouh, by Sheikh Zayed Rd", "mosque", 25.08133, 55.15168, {"pattern": "convenient"}),  # 94 m off route, 5.4 km along
    POI("s17", "Mosque - Al Quoz, by Sheikh Zayed Rd", "mosque", 25.12376, 55.20626, {"pattern": "convenient"}),  # 130 m off route, 12.7 km along
    POI("s18", "Mosque - Ibn Battuta area, behind the start", "mosque", 25.03925, 55.11363, {"pattern": "behind"}),  # 90 m off route, 0.6 km behind
    POI("s19", "Mosque - Al Barsha 1 side street", "mosque", 25.11419, 55.18092, {"pattern": "side_street"}),  # 690 m off route
    POI("s20", "Mosque - Downtown back streets", "mosque", 25.19123, 55.25730, {"pattern": "off_route"}),  # 260 m off route
    POI("s21", "Park - Al Barsha (set back from the road)", "park", 25.11117, 55.18081, {"pattern": "off_route"}),  # 421 m off route
    POI("s22", "Park - Al Quoz community park", "park", 25.13952, 55.22407, {"pattern": "off_route"}),  # 330 m off route
    POI("s23", "Park - Jebel Ali, behind the start", "park", 25.03803, 55.11292, {"pattern": "behind"}),  # 120 m off route, 0.8 km behind
    POI("s24", "Park - Al Sufouh beach side", "park", 25.08546, 55.14004, {"pattern": "side_street"}),  # 1.1 km off route
    POI("s25", "Park - Downtown edge", "park", 25.20138, 55.27260, {"pattern": "off_route"}),  # 380 m off route
]


# Places the dashboard offers as start and destination. The two saved places
# come first; the SZR_POINTS names are the points of the demonstration route.
SZR_POINTS = ["Ibn Battuta", "Dubai Marina / Jebel Ali interchange", "Dubai Internet City", "Mall of the Emirates",
              "Al Quoz", "Al Safa", "Downtown / Business Bay", "Trade Centre", "Za'abeel", "Garhoud", "Al Rigga"]
N = dubai_roads.NODES
PLACES: Dict[str, LatLon] = {
    "\U0001F3E2 Office - Bay Square, Business Bay": N["baysq"],
    "\U0001F3E0 Home - Dubai Hills Estate": N["dhills"],
}
PLACES.update(zip(SZR_POINTS, SAMPLE_DUBAI_ROUTE))
PLACES.update({
    "Bay Square, Business Bay": N["baysq"],
    "Dubai Hills Estate": N["dhills"],
    "Jumeirah Lake Towers (JLT)": N["jlt"],
    "Jumeirah Village Circle (JVC)": N["jvc"],
    "DIFC": N["difc"],
    "Dubai Festival City": N["festival"],
    "Mirdif": N["mirdif"],
    "Dubai Silicon Oasis": N["dso"],
    "DXB Airport Terminal 3": N["dxb"],
    "Deira City Centre": N["deira_cc"],
    "Al Karama": N["karama"],
    "Motor City": N["motor"],
    "Dubai Investments Park": N["dip"],
    "Arabian Ranches": N["mbz_ranches"],
    "Meydan": N["ak_meydan"],
    "Ras Al Khor": N["ak_rak"],
})
_ROUTE_INDEX = {name: i for i, name in enumerate(SZR_POINTS)}
NETWORK_POIS: List[POI] = dubai_roads.network_pois({k: v for k, v in PLACES.items() if not k[0] in "\U0001F3E2\U0001F3E0"})


def resolve_place(name: str, allow_geocode: bool = True) -> Tuple[Optional[LatLon], str]:
    """Coordinates for a place: a known place, "lat, lon", or a geocoded search."""
    text = (name or "").strip()
    if text in PLACES:
        return PLACES[text], "known place"
    parts = [x.strip() for x in text.split(",")]
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1])), "coordinates"
        except ValueError:
            pass
    if allow_geocode and text:
        try:
            found = geocode(text)
            if found:
                return found, "geocoded with OpenStreetMap Nominatim"
        except Exception:
            pass
    return None, "not found"


def offline_route(origin_name: str, destination_name: str) -> Optional[List[LatLon]]:
    """The part of the demonstration route between two of its points, in either direction."""
    a, b = _ROUTE_INDEX.get(origin_name), _ROUTE_INDEX.get(destination_name)
    if a is None or b is None or a == b:
        return None
    return SAMPLE_DUBAI_ROUTE[a:b + 1] if a < b else SAMPLE_DUBAI_ROUTE[b:a + 1][::-1]


def place_name(name: str) -> str:
    """A place name without the saved-place emoji, for labels."""
    return name.lstrip("\U0001F3E2\U0001F3E0 ").strip()


def network_route(points: Sequence[LatLon]) -> dubai_roads.NetworkRoute:
    """Offline route through ``points`` over the approximate Dubai road network."""
    coords: List[LatLon] = []
    steps: List[Tuple[str, float]] = []
    dist = dur = 0.0
    for a, b in zip(points, points[1:]):
        leg = dubai_roads.route(a, b)
        coords += leg.coords if not coords else leg.coords[1:]
        for name, length in leg.steps:
            if steps and steps[-1][0] == name:
                steps[-1] = (name, steps[-1][1] + length)
            else:
                steps.append((name, length))
        dist += leg.distance_m
        dur += leg.duration_s
    return dubai_roads.NetworkRoute(coords, dist, dur, steps)


def osrm_route(points: Sequence[LatLon], timeout_s: float = 15.0) -> dubai_roads.NetworkRoute:
    """Road route through ``points`` from a live OSRM engine, with road names (FR1).

    Tries each server in OSRM_SERVERS in turn and raises if none answers.
    """
    coords = ";".join("%.6f,%.6f" % (p[1], p[0]) for p in points)
    error: Exception = RuntimeError("no OSRM server configured")
    for server in OSRM_SERVERS:
        try:
            data = net.get_json("%s/route/v1/driving/%s" % (server, coords), timeout_s=timeout_s,
                                params={"overview": "full", "geometries": "geojson", "steps": "true"})
            if data.get("code") != "Ok" or not data.get("routes"):
                raise RuntimeError("OSRM could not find a route: %s" % data.get("code"))
            r = data["routes"][0]
            steps: List[Tuple[str, float]] = []
            for leg in r.get("legs", []):
                for st in leg.get("steps", []):
                    name = st.get("ref") and st.get("name") and "%s (%s)" % (st["name"], st["ref"]) or st.get("name") or st.get("ref") or "local streets"
                    if steps and steps[-1][0] == name:
                        steps[-1] = (name, steps[-1][1] + float(st.get("distance", 0)))
                    elif st.get("distance", 0) > 0:
                        steps.append((name, float(st["distance"])))
            path = [(c[1], c[0]) for c in r["geometry"]["coordinates"]]
            return dubai_roads.NetworkRoute(path, float(r["distance"]), float(r["duration"]), steps)
        except Exception as exc:  # try the next server
            error = exc
    raise error


def load_osrm_route(points: Sequence[LatLon], timeout_s: float = 15.0) -> Tuple[List[LatLon], float, float]:
    """(coordinates, distance_m, duration_s) of the live OSRM route through ``points``. Raises on failure."""
    r = osrm_route(points, timeout_s)
    return r.coords, r.distance_m, r.duration_s


def sample_pois(category: Optional[str] = None) -> List[POI]:
    """The offline sample, optionally limited to one category."""
    return [p for p in SAMPLE_DUBAI_POIS if category is None or p.category == category]


def offline_pois(category: Optional[str] = None) -> List[POI]:
    """The curated Sheikh Zayed Road sample plus the generated stops on the rest of the network."""
    return sample_pois(category) + [p for p in NETWORK_POIS if category is None or p.category == category]


def bbox_around(route: Sequence[LatLon], margin_deg: float = 0.02) -> Tuple[float, float, float, float]:
    """(south, west, north, east) box around a route, with a margin."""
    lats = [p[0] for p in route]
    lons = [p[1] for p in route]
    return (min(lats) - margin_deg, min(lons) - margin_deg, max(lats) + margin_deg, max(lons) + margin_deg)


def overpass_query(category: str, bbox: Tuple[float, float, float, float]) -> str:
    """Overpass QL for one category inside a (south, west, north, east) box."""
    if category not in CATEGORY_TAGS:
        raise ValueError("unknown category %r; choose from %s" % (category, ", ".join(CATEGORIES)))
    box = "(%.5f,%.5f,%.5f,%.5f)" % bbox
    parts = []
    for tags in CATEGORY_TAGS[category]:
        filt = "".join('["%s"="%s"]' % kv for kv in tags.items())
        parts.append("node%s%s;way%s%s;" % (filt, box, filt, box))
    return "[out:json][timeout:60];(%s);out center tags;" % "".join(parts)


def load_live_pois(category: str, route: Sequence[LatLon], timeout_s: float = 60.0) -> List[POI]:
    """Points of interest for ``category`` around ``route`` from the Overpass API.

    Raises an exception if the query fails; callers fall back to the offline sample.
    """
    data = net.post_form_json(OVERPASS_URL, {"data": overpass_query(category, bbox_around(route))}, timeout_s)
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        name = tags.get("name:en") or tags.get("name") or tags.get("brand") or category.title()
        pois.append(POI("%s/%s" % (el.get("type", "node"), el.get("id")), name, category, float(lat), float(lon), tags))
    return pois


def _classify(tags: Dict[str, str]) -> Optional[str]:
    for category, options in CATEGORY_TAGS.items():
        if any(all(tags.get(k) == v for k, v in opt.items()) for opt in options):
            return category
    return None


def _thin(route: Sequence[LatLon], spacing_m: float) -> List[LatLon]:
    """Route points at least ``spacing_m`` apart (keeps the ends), to keep Overpass queries short."""
    out = [route[0]]
    for p in route[1:-1]:
        if haversine_m(out[-1], p) >= spacing_m:
            out.append(p)
    out.append(route[-1])
    return out


def load_live_pois_along(categories: Sequence[str], route: Sequence[LatLon], radius_m: float,
                         timeout_s: float = 30.0) -> Dict[str, List[POI]]:
    """POIs of every category within ``radius_m`` of the route, in one Overpass query (FR2).

    Uses Overpass's ``around`` filter along the route line, so only the strip
    round the route is downloaded, and tries each mirror in OVERPASS_MIRRORS.
    Raises if every mirror fails.
    """
    line = ",".join("%.5f,%.5f" % p for p in _thin(route, 250.0))
    parts = []
    for category in categories:
        for tags in CATEGORY_TAGS[category]:
            filt = "".join('["%s"="%s"]' % kv for kv in tags.items())
            parts.append("nwr%s(around:%d,%s);" % (filt, int(radius_m), line))
    query = "[out:json][timeout:25];(%s);out center tags;" % "".join(parts)
    error: Exception = RuntimeError("no Overpass server configured")
    for url in OVERPASS_MIRRORS:
        try:
            elements = net.post_form_json(url, {"data": query}, timeout_s).get("elements", [])
            break
        except Exception as exc:  # try the next mirror
            error = exc
    else:
        raise error
    found: Dict[str, List[POI]] = {c: [] for c in categories}
    for el in elements:
        tags = el.get("tags", {})
        category = _classify(tags)
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if category not in found or lat is None or lon is None:
            continue
        name = tags.get("name:en") or tags.get("name") or tags.get("brand") or category.capitalize()
        found[category].append(POI("%s/%s" % (el.get("type", "node"), el.get("id")), name, category, float(lat), float(lon), tags))
    return found


def load_live_graph(route: Sequence[LatLon] = SAMPLE_DUBAI_ROUTE, network_type: str = "drive"):
    """Drivable road graph around a route, downloaded with OSMnx (FR1)."""
    import osmnx as ox

    south, west, north, east = bbox_around(route)
    try:  # OSMnx 2.x takes (left, bottom, right, top)
        return ox.graph_from_bbox((west, south, east, north), network_type=network_type)
    except TypeError:  # OSMnx 1.x takes north, south, east, west
        return ox.graph_from_bbox(north, south, east, west, network_type=network_type)


def route_from_graph(graph, origin: LatLon, destination: LatLon) -> List[LatLon]:
    """Shortest drivable path between two points on an OSMnx graph, as (lat, lon) points (FR1)."""
    import networkx as nx
    import osmnx as ox

    orig = ox.distance.nearest_nodes(graph, origin[1], origin[0])
    dest = ox.distance.nearest_nodes(graph, destination[1], destination[0])
    nodes = nx.shortest_path(graph, orig, dest, weight="length")
    return [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nodes]


def geocode(place: str) -> Optional[LatLon]:
    """Look up a place name in Dubai with OpenStreetMap Nominatim (via geopy if installed)."""
    query = place + ", Dubai, United Arab Emirates"
    hits = net.get_json("https://nominatim.openstreetmap.org/search", timeout_s=10,
                        params={"q": query, "format": "json", "limit": "1"})
    return (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None
