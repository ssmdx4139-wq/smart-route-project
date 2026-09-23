"""The SmartRoute dashboard (FR8, FR9): Gradio for the interface, Folium for the map.

Run from the src folder:
    python app.py
then open the local address it prints (normally http://127.0.0.1:7860).

The screen is laid out as SmartRoute running on an Apple CarPlay head unit:
the CarPlay sidebar on the left (clock, signal, the SmartRoute app icon, Home /
Work / Day-Night buttons), the navigation map filling the screen with a
Waze-style turn banner, ETA bar, icon pins and the car, and a CarPlay tabbed
list panel (Drive / Where to / Settings) on the right.

Controls
  * From / To          - pick a Dubai place, type a place name, or type "lat, lon" (FR1)
  * Stops on the way   - one or more categories; several are ordered greedily (FR7)
  * Corridor width     - 50-500 m half-width of the Smart Corridor (FR3, FR9)
  * Distance into trip - where the vehicle is now; stops behind it are excluded (FR4)
  * Live data toggles  - Overpass POIs and OSRM routing, with labelled fallbacks (FR9, NFR5)

Live calls that fail fall back to the offline data and the heuristic cost
model, and the trip panel always says which source produced the numbers (NFR5).
"""

import datetime
import html
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import folium
import gradio as gr
from branca.element import MacroElement
from jinja2 import Template

import data_loader
import dubai_roads
from baseline_radius import radius_search
from corridor import CorridorCandidate, SmartCorridor, corridor_reason
from costing import DetourCostEngine, HeuristicBackend, OSRMBackend
from geometry import LatLon, haversine_m, point_along, route_length_m
from sequencer import greedy_sequence

CATEGORY_COLORS = {
    "supermarket": "#0a84ff",
    "pharmacy": "#ff375f",
    "petrol station": "#ff9f0a",
    "mosque": "#30b158",
    "park": "#32ade6",
}
CATEGORY_ICONS = {
    "supermarket": "&#128722;",     # shopping cart
    "pharmacy": "&#128138;",        # pill
    "petrol station": "&#9981;",    # fuel pump
    "mosque": "&#128332;",          # mosque
    "park": "&#127795;",            # tree
}
CATEGORY_LABELS = {"supermarket": "Supermarket", "pharmacy": "Pharmacy", "petrol station": "Petrol",
                   "mosque": "Mosque", "park": "Park"}
FILTERED_COLOR = "#8e8e93"
TRIP_COLOR = "#30d158"
CORRIDOR_CIRCLE_SPACING_M = 400.0
MAP_HEIGHT_PX = 700      # the height of the CarPlay screen
DEFAULT_FROM, DEFAULT_TO = list(data_loader.PLACES)[:2]   # Office -> Home
UI_FONT = "-apple-system,BlinkMacSystemFont,'SF Pro Display','Helvetica Neue',Arial,sans-serif"

# Map looks: Waze-like bright day map, CarPlay-like dark night map. OpenStreetMap tiles,
# restyled with CSS filters so no API key is needed.
THEMES = {
    "Night": {"tiles": "invert(1) hue-rotate(180deg) brightness(.9) contrast(.9) saturate(.5)", "bg": "#0b0b0d",
              "route": "#3ea6ff", "casing": "#0a3a7a", "done": "#5a5a5e", "band": "#3ea6ff",
              "card": "rgba(28,28,30,.94)", "text": "#ffffff", "sub": "#aeaeb2", "line": "rgba(255,255,255,.08)"},
    "Day": {"tiles": "saturate(.75) brightness(1.04) contrast(.95)", "bg": "#e9eef2",
            "route": "#35b0ff", "casing": "#0b5fb8", "done": "#9aa3ad", "band": "#35b0ff",
            "card": "rgba(255,255,255,.97)", "text": "#1c1c1e", "sub": "#6b7280", "line": "rgba(0,0,0,.08)"},
}


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------
def corridor_circle_points(route, spacing_m: float = CORRIDOR_CIRCLE_SPACING_M):
    """Points every ``spacing_m`` along the route, used to draw the corridor as a faint band."""
    points = [route[0]]
    carry = 0.0
    for a, b in zip(route, route[1:]):
        seg = haversine_m(a, b)
        if seg == 0:
            continue
        d = spacing_m - carry
        while d <= seg:
            t = d / seg
            points.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            d += spacing_m
        carry = (carry + seg) % spacing_m
    points.append(route[-1])
    return points


def split_route(route: List[LatLon], progress_m: float) -> Tuple[List[LatLon], List[LatLon]]:
    """(driven part, part still ahead) of the route at ``progress_m``."""
    here = point_along(route, progress_m)
    travelled, done = 0.0, [route[0]]
    for i, (a, b) in enumerate(zip(route, route[1:])):
        travelled += haversine_m(a, b)
        if travelled >= progress_m:
            return done + [here], [here] + list(route[i + 1:])
        done.append(b)
    return list(route), [route[-1]]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def why(exc: Exception) -> str:
    """A short, readable reason a live call failed, e.g. "SSLError: certificate verify failed"."""
    text = " ".join(str(exc).split())
    return "%s: %s" % (type(exc).__name__, text[:110]) if text else type(exc).__name__


def acquire_pois(categories: List[str], route, width_m: float, use_live: bool) -> Tuple[Dict[str, list], str]:
    """POIs for each category: live from Overpass along the route, or the offline sample."""
    note = ""
    if use_live:
        try:
            found = data_loader.load_live_pois_along(categories, route, max(width_m, 150.0) + 400.0)
            if any(found.values()):
                return found, "live OpenStreetMap (Overpass API)"
            note = " - live query returned no results"
        except Exception as exc:  # network, timeout, bad response
            note = " - live query failed (%s)" % why(exc)
    return {c: data_loader.offline_pois(c) for c in categories}, "offline sample dataset" + note


def acquire_route(from_name: str, to_name: str, origin: LatLon, destination: LatLon, use_osrm: bool):
    """Road route between the two places: live OSRM, else the offline road network (FR1)."""
    note = ""
    if use_osrm:
        try:
            return data_loader.osrm_route([origin, destination]), "live OSRM road route"
        except Exception as exc:
            note = " - live OSRM unavailable (%s)" % why(exc)
    demo = data_loader.offline_route(from_name, to_name)
    if demo:
        return (dubai_roads.NetworkRoute(demo, route_length_m(demo), 0.0, [("Sheikh Zayed Rd", route_length_m(demo))]),
                "demonstration route along Sheikh Zayed Road (offline sample)" + note)
    return data_loader.network_route([origin, destination]), "offline approximate Dubai road network" + note


def make_engine(use_osrm: bool, points: List[LatLon]) -> Tuple[DetourCostEngine, str]:
    """A cost engine on live OSRM if it answers, otherwise the offline heuristic."""
    if use_osrm:
        error = None
        for server in data_loader.OSRM_SERVERS:
            try:
                backend = OSRMBackend(server)
                backend.prefetch(points)
                return DetourCostEngine(backend), "live OSRM routing"
            except Exception as exc:
                error = exc
        return DetourCostEngine(HeuristicBackend()), "offline heuristic cost model - live OSRM unavailable (%s)" % why(error)
    return DetourCostEngine(HeuristicBackend()), "offline heuristic cost model"


def where_text(c: CorridorCandidate, progress_m: float) -> str:
    ahead = c.along_track_m - progress_m
    return ("%.1f km ahead" % (ahead / 1000)) if ahead >= 0 else ("%.1f km behind you" % (-ahead / 1000))


def route_slice(route: List[LatLon], start_m: float, end_m: float) -> List[LatLon]:
    """The part of the route between two along-track positions, in the direction start -> end."""
    if start_m > end_m:
        return route_slice(route, end_m, start_m)[::-1]
    out, travelled = [point_along(route, start_m)], 0.0
    for a, b in zip(route, route[1:]):
        travelled += haversine_m(a, b)
        if start_m < travelled < end_m:
            out.append(b)
    out.append(point_along(route, end_m))
    return out


def drive_via_stops(route, progress_m, stops: List[CorridorCandidate], live: bool, vehicle, destination):
    """The drive through the chosen stops: live OSRM when on, else the route with a spur to each stop."""
    if live:
        try:
            return data_loader.osrm_route([vehicle] + [c.poi.coord for c in stops] + [destination]).coords
        except Exception:
            pass
    path, pos = [vehicle], progress_m
    for c in stops:
        at = max(c.along_track_m, 0.0)
        path += route_slice(route, pos, at)[1:] + [c.poi.coord, point_along(route, at)]
        pos = at
    return path + route_slice(route, pos, route_length_m(route))[1:]




def bearing_deg(a: LatLon, b: LatLon) -> float:
    """Compass bearing from a to b, in degrees."""
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def turn_angle(route: List[LatLon], at_m: float, look_m: float = 120.0) -> float:
    """How sharply the route turns at ``at_m``: negative left, positive right, in degrees."""
    before = bearing_deg(point_along(route, at_m - look_m), point_along(route, at_m))
    after = bearing_deg(point_along(route, at_m), point_along(route, at_m + look_m))
    return (after - before + 540) % 360 - 180


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
@dataclass
class Step:
    kind: str                     # "go", "stop" or "arrive"
    text: str                     # e.g. "Turn right onto"
    road: str
    dist_m: float
    angle: float = 0.0            # turn angle for the arrow
    stop: Optional[CorridorCandidate] = None
    number: int = 0


def directions(route, steps, progress_m: float, length_m: float, stops: List[CorridorCandidate], to_label: str) -> List[Step]:
    """Road-by-road directions from the current position, with the stops in place."""
    events = sorted(((max(c.along_track_m, progress_m), i + 1, c) for i, c in enumerate(stops)), key=lambda e: e[0])
    out: List[Step] = []
    at = progress_m
    sections = [(n, s, e) for n, s, e in dubai_roads.along_steps(steps) if e > progress_m]
    scale = length_m / sections[-1][2] if sections and sections[-1][2] > 0 else 1.0  # step lengths vs. drawn route

    def go(name: str, start: float, end: float) -> None:
        if end - start <= 20:
            return
        if not out:
            out.append(Step("go", "Head along", name, end - start))
        elif out[-1].kind == "go" and out[-1].road == name or out[-1].kind == "stop":
            out.append(Step("go", "Continue on", name, end - start))
        else:
            angle = turn_angle(route, start)
            text = "Turn right onto" if angle > 35 else "Turn left onto" if angle < -35 else \
                   "Keep right onto" if angle > 10 else "Keep left onto" if angle < -10 else "Continue onto"
            out.append(Step("go", text, name, end - start, angle))

    for name, start, end in sections:
        start, end = max(start * scale, progress_m), end * scale
        while events and events[0][0] <= end:
            pos, n, c = events.pop(0)
            go(name, at, pos)
            out.append(Step("stop", "Stop %d" % n, c.poi.name, 0.0, 0.0, c, n))
            at = pos
        go(name, max(at, start), end)
        at = end
    for pos, n, c in events:  # anything left over (should not happen)
        out.append(Step("stop", "Stop %d" % n, c.poi.name, 0.0, 0.0, c, n))
    out.append(Step("arrive", "Arrive at", to_label, 0.0))
    return out


def arrow_svg(step: Step, size: int = 28, color: str = "#fff") -> str:
    """A maneuver icon: a bent arrow for turns, a star for stops, a flag for arrival."""
    if step.kind == "stop":
        return '<span style="font-size:%dpx;line-height:1">%s</span>' % (int(size * .85), CATEGORY_ICONS.get(step.stop.poi.category, "&#9733;"))
    if step.kind == "arrive":
        return '<span style="font-size:%dpx;line-height:1">&#127937;</span>' % int(size * .85)
    a = max(-90.0, min(90.0, step.angle))
    ex, ey = 12 + 7 * math.sin(math.radians(a)), 12 - 7 * math.cos(math.radians(a))
    path = "M12 22 V12 L%.1f %.1f" % (ex, ey)
    head = 'transform="translate(%.1f %.1f) rotate(%.0f)"' % (ex, ey, a)
    return ('<svg width="%d" height="%d" viewBox="0 0 24 24"><path d="%s" stroke="%s" stroke-width="3.2" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/><path %s d="M-5 3 L0 -3 L5 3 Z" fill="%s"/></svg>'
            % (size, size, path, color, head, color))


def km(m: float) -> str:
    return "%.0f m" % m if m < 950 else "%.1f km" % (m / 1000)


# ---------------------------------------------------------------------------
# The whole request
# ---------------------------------------------------------------------------
def card(title: str, body: str) -> Tuple[str, str]:
    """(map placeholder, panel) for a message instead of a trip."""
    msg = '<div class="sr-card sr-msg"><div class="sr-h">%s</div><div class="sr-sub">%s</div></div>' % (title, body)
    empty = '<div class="sr-empty"><div>&#128506;</div><p>%s</p></div>' % title
    return empty, msg


def plan_trip(from_name: str, to_name: str, categories: List[str], width_m: float, progress_km: float,
              use_live_pois: bool, use_live_osrm: bool, theme: str = "Night"):
    """Run the full pipeline and return (map HTML, trip panel HTML)."""
    started = time.perf_counter()
    origin, how_o = data_loader.resolve_place(from_name)
    destination, how_d = data_loader.resolve_place(to_name)
    if origin is None or destination is None:
        missing = from_name if origin is None else to_name
        return card("Place not found", "Could not find <b>%s</b>. Pick a place from the list, type a place name, "
                    "or type coordinates as <code>25.2, 55.27</code>." % html.escape(missing or ""))
    if haversine_m(origin, destination) < 200:
        return card("Start and destination are the same", "Pick two different places.")
    categories = [c for c in (categories or []) if c in data_loader.CATEGORY_TAGS]
    if not categories:
        return card("Pick at least one stop type", "Tap one or more of the stop buttons, then Go.")

    road, route_source = acquire_route(from_name, to_name, origin, destination, use_live_osrm)
    route = data_loader._thin(road.coords, 25.0) if len(road.coords) > 400 else road.coords
    length_m = route_length_m(route)
    progress_m = max(0.0, min(progress_km * 1000.0, length_m - 1.0))
    vehicle = point_along(route, progress_m)
    corridor = SmartCorridor(route, width_m)

    all_pois, poi_source = acquire_pois(categories, route, width_m, use_live_pois)
    projected = {cat: corridor.split(all_pois.get(cat, []), progress_m) for cat in categories}
    engine, cost_source = make_engine(use_live_osrm, [vehicle, destination] +
                                      [c.poi.coord for kept, _ in projected.values() for c in kept])

    per_category: Dict[str, Dict] = {}
    for cat in categories:
        pois = all_pois.get(cat, [])
        kept, rejected = projected[cat]
        ranked = engine.rank(vehicle, destination, kept, progress_m)
        nearest = radius_search(pois, vehicle)
        radius_pick = engine.evaluate(vehicle, destination, corridor.project(nearest[0][0]), progress_m) if nearest else None
        per_category[cat] = {"pois": pois, "kept": kept, "rejected": rejected, "ranked": ranked, "radius": radius_pick}

    chosen = [d["ranked"][0].candidate for d in per_category.values() if d["ranked"]]
    sequence = greedy_sequence(vehicle, destination, chosen, engine) if chosen else []
    direct_s = engine.backend.route_time_s(vehicle, destination)
    elapsed = time.perf_counter() - started

    ordered = [s.stop for s in sequence if s.stop]
    trip_path = drive_via_stops(route, progress_m, ordered, route_source.startswith("live"), vehicle, destination) if ordered else None
    from_label, to_label = data_loader.place_name(from_name), data_loader.place_name(to_name)
    steps = directions(route, road.steps, progress_m, length_m, ordered, to_label)
    trip = {
        "from": from_label, "to": to_label, "length_m": length_m, "progress_m": progress_m, "width_m": width_m,
        "total_s": sequence[-1].cumulative_s if sequence else direct_s, "direct_s": direct_s,
        "live": route_source.startswith("live") and poi_source.startswith("live"),
        "road_live": route_source.startswith("live"), "pois_live": poi_source.startswith("live"),
        "route_source": route_source, "poi_source": poi_source, "cost_source": cost_source,
        "how": (how_o, how_d), "elapsed": elapsed, "stops": ordered,
    }
    th = THEMES.get(theme, THEMES["Night"])
    fmap = build_folium_map(route, corridor, progress_m, per_category, sequence, trip_path, steps, trip, th)
    return fmap._repr_html_(), trip_panel(trip, per_category, sequence, steps)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
class ZoomBottomRight(MacroElement):
    """Leaflet's zoom buttons in the bottom-right corner, clear of the turn banner."""

    _template = Template("""{% macro script(this, kwargs) %}
        L.control.zoom({position: 'bottomright'}).addTo({{ this._parent.get_name() }});
    {% endmacro %}""")


def poi_pin(category: str, size: int = 30, number: int = 0, glow: bool = False) -> folium.DivIcon:
    """A Waze-style teardrop pin with the category icon (and a number badge for suggested stops)."""
    color = CATEGORY_COLORS.get(category, "#0a84ff")
    badge = ('<div style="position:absolute;top:-8px;right:-8px;width:20px;height:20px;border-radius:50%%;background:#fff;'
             'color:#111;font:800 12px/20px %s;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.4)">%d</div>' % (UI_FONT, number)) if number else ""
    ring = ('<div style="position:absolute;inset:-7px;border-radius:50%%;border:3px solid %s;opacity:.55;'
            'animation:srpulse 1.8s ease-out infinite"></div>' % color) if glow else ""
    html_ = ('<div style="position:relative;width:{s}px;height:{s}px">{ring}'
             '<div style="width:{s}px;height:{s}px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:{c};'
             'border:3px solid #fff;box-shadow:0 3px 10px rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center">'
             '<span style="transform:rotate(45deg);font-size:{f}px;line-height:1">{i}</span></div>{badge}</div>'
             ).format(s=size, c=color, f=int(size * .5), i=CATEGORY_ICONS.get(category, "&#9733;"), ring=ring, badge=badge)
    return folium.DivIcon(html=html_, icon_size=(size, size), icon_anchor=(size // 2, int(size * 1.2)))


def car_icon(heading: float) -> folium.DivIcon:
    """A black muscle car seen from above (Dodge Challenger style), turned to the direction of travel."""
    svg = ('<svg width="30" height="54" viewBox="0 0 30 54" style="transform:rotate({h:.0f}deg);'
           'filter:drop-shadow(0 3px 5px rgba(0,0,0,.55))">'
           '<rect x="2" y="2" width="26" height="50" rx="8" fill="#111" stroke="#fff" stroke-width="1.6"/>'
           '<rect x="11" y="3" width="3" height="48" fill="#3a3a3c"/><rect x="16" y="3" width="3" height="48" fill="#3a3a3c"/>'
           '<path d="M6 17 Q15 12 24 17 L23 24 L7 24 Z" fill="#7fb3d5" opacity=".85"/>'
           '<path d="M7 36 L23 36 L24 42 Q15 45 6 42 Z" fill="#7fb3d5" opacity=".7"/>'
           '<rect x="4" y="3" width="6" height="3" rx="1.5" fill="#fff8c4"/><rect x="20" y="3" width="6" height="3" rx="1.5" fill="#fff8c4"/>'
           '<rect x="4" y="48" width="7" height="2.6" rx="1.3" fill="#ff3b30"/><rect x="19" y="48" width="7" height="2.6" rx="1.3" fill="#ff3b30"/>'
           '</svg>').format(h=heading)
    return folium.DivIcon(html='<div style="width:30px;height:54px">%s</div>' % svg, icon_size=(30, 54), icon_anchor=(15, 27))


def place_pin(kind: str) -> folium.DivIcon:
    if kind == "start":
        html_ = ('<div style="width:22px;height:22px;border-radius:50%;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.45);'
                 'display:flex;align-items:center;justify-content:center"><div style="width:12px;height:12px;border-radius:50%;'
                 'background:#0a84ff"></div></div>')
        return folium.DivIcon(html=html_, icon_size=(22, 22), icon_anchor=(11, 11))
    html_ = ('<div style="width:40px;height:40px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:#fff;'
             'box-shadow:0 3px 10px rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center">'
             '<span style="transform:rotate(45deg);font-size:21px">&#127937;</span></div>')
    return folium.DivIcon(html=html_, icon_size=(40, 40), icon_anchor=(20, 48))


def tip(text: str) -> folium.Tooltip:
    return folium.Tooltip(text, sticky=True)


def name_tag(text: str, th: dict) -> folium.Tooltip:
    return folium.Tooltip('<span style="color:%s">%s</span>' % (th["text"], html.escape(text)), permanent=True,
                          direction="right", offset=(14, -14), className="sr-tag")


def map_css(th: dict) -> str:
    return """<style>
.leaflet-container{background:%(bg)s;font-family:%(font)s}
.leaflet-tile-pane{filter:%(tiles)s}
.leaflet-tooltip{background:%(card)s!important;border:none!important;color:%(text)s!important;border-radius:10px!important;
  box-shadow:0 3px 12px rgba(0,0,0,.35)!important;font:600 12.5px %(font)s!important;padding:6px 10px!important}
.leaflet-tooltip:before{display:none!important}
.sr-tag{padding:4px 9px!important;font-weight:700!important}
.leaflet-bar{border:none!important;box-shadow:0 3px 12px rgba(0,0,0,.35)!important;border-radius:14px!important;overflow:hidden}
.leaflet-bar a{background:%(card)s!important;color:%(text)s!important;border-color:%(line)s!important;width:40px!important;
  height:40px!important;line-height:40px!important;font-size:20px!important}
.leaflet-control-attribution{background:transparent!important;color:%(sub)s!important;font-size:10px}
.leaflet-control-attribution a{color:%(sub)s!important}
.leaflet-bottom.leaflet-right{margin-bottom:96px}
@keyframes srpulse{0%%{transform:scale(.8);opacity:.7}100%%{transform:scale(1.6);opacity:0}}
.sr-banner{position:fixed;top:16px;left:16px;z-index:9999;width:min(420px,calc(100%% - 32px));border-radius:22px;overflow:hidden;
  box-shadow:0 10px 30px rgba(0,0,0,.45);font-family:%(font)s}
.sr-banner .main{display:flex;gap:14px;align-items:center;padding:14px 18px;background:linear-gradient(135deg,#1f8cff,#0a63d8);color:#fff}
.sr-banner .dist{font:800 30px/1 %(font)s;letter-spacing:-.02em}
.sr-banner .road{font:600 17px/1.25 %(font)s;opacity:.95;margin-top:4px}
.sr-banner .then{display:flex;gap:8px;align-items:center;padding:8px 18px;background:#0850a8;color:#dce9ff;font:600 13.5px %(font)s}
.sr-status{position:fixed;top:16px;right:16px;z-index:9999;display:flex;gap:8px;align-items:center;padding:8px 14px;border-radius:999px;
  background:%(card)s;color:%(text)s;box-shadow:0 3px 12px rgba(0,0,0,.35);font:700 12.5px %(font)s}
.sr-status{flex-direction:column;align-items:flex-end;max-width:380px}
.sr-status i{width:9px;height:9px;border-radius:50%%;display:inline-block}
.sr-why{font:500 11px/1.35 %(font)s;color:%(sub)s;text-align:right;margin-top:3px}
.sr-eta{position:fixed;left:50%%;bottom:18px;transform:translateX(-50%%);z-index:9999;display:flex;align-items:center;gap:0;
  background:%(card)s;color:%(text)s;border-radius:24px;box-shadow:0 10px 30px rgba(0,0,0,.4);font-family:%(font)s;overflow:hidden}
.sr-eta div{padding:12px 20px;text-align:center;border-right:1px solid %(line)s}
.sr-eta div:last-child{border-right:none}
.sr-eta b{display:block;font:800 24px/1.05 %(font)s;letter-spacing:-.02em}
.sr-eta span{font:600 11.5px %(font)s;color:%(sub)s;text-transform:uppercase;letter-spacing:.06em}
.sr-eta .plus b{color:#30d158}
.sr-legend{position:fixed;left:16px;bottom:18px;z-index:9998;background:%(card)s;color:%(text)s;border-radius:14px;padding:8px 12px;
  box-shadow:0 3px 12px rgba(0,0,0,.3);font:600 11.5px/1.9 %(font)s}
.sr-legend i{display:inline-block;vertical-align:middle;margin-right:6px}
</style>""" % dict(th, font=UI_FONT)


def build_folium_map(route, corridor, progress_m, per_category, sequence, trip_path, steps: List[Step],
                     trip: dict, th: dict) -> folium.Figure:
    figure = folium.Figure(height=MAP_HEIGHT_PX)
    fmap = folium.Map(location=route[0], zoom_start=12, tiles="OpenStreetMap", zoom_control=False).add_to(figure)
    fmap.add_child(ZoomBottomRight())
    root = fmap.get_root()
    root.header.add_child(folium.Element(map_css(th)))

    # the Smart Corridor, a faint band along the route
    band = folium.FeatureGroup(name="Corridor")
    for p in corridor_circle_points(route):
        folium.Circle(p, radius=corridor.corridor_width_m, stroke=False, fill=True, fill_color=th["band"], fill_opacity=0.10).add_to(band)
    band.add_to(fmap)

    # the route, Waze style: a thick line with a darker casing; the driven part in grey
    done, ahead = split_route(route, progress_m)
    if len(done) > 1 and progress_m > 0:
        folium.PolyLine(done, color=th["done"], weight=8, opacity=0.9, tooltip=tip("Already driven")).add_to(fmap)
    if trip["road_live"]:
        folium.PolyLine(ahead, color=th["casing"], weight=13, opacity=0.95).add_to(fmap)
        folium.PolyLine(ahead, color=th["route"], weight=8, opacity=1, tooltip=tip("Route ahead")).add_to(fmap)
    else:  # the offline sketch joins junctions with straight lines, so show it as approximate
        folium.PolyLine(ahead, color=th["route"], weight=7, opacity=0.9, dash_array="14 10",
                        tooltip=tip("Approximate route (offline sketch of the main roads) - turn on live routing for exact roads")).add_to(fmap)
    if trip_path:
        folium.PolyLine(trip_path, color=TRIP_COLOR, weight=5, opacity=1, dash_array="1 11", line_cap="round",
                        tooltip=tip("Your drive via the stops")).add_to(fmap)

    # candidates: filtered out (small grey dots), inside the corridor (pins), suggested (big numbered pins)
    chosen = {c.poi.id: i + 1 for i, c in enumerate(trip["stops"])}
    for cat, d in per_category.items():
        for c in d["rejected"]:
            folium.CircleMarker(c.poi.coord, radius=5, color="#ffffff", weight=1.5, fill=True, fill_color=FILTERED_COLOR, fill_opacity=0.85,
                                tooltip=tip("%s<br><span style='opacity:.7'>Filtered out: %s &middot; %.0f m from the route</span>"
                                            % (html.escape(c.poi.name), corridor_reason(c, corridor, progress_m) or "excluded", c.cross_track_m))).add_to(fmap)
        detour = {id(r.candidate): r for r in d["ranked"]}
        for c in d["kept"]:
            if c.poi.id in chosen:
                continue
            r = detour.get(id(c))
            folium.Marker(c.poi.coord, icon=poi_pin(cat, 26),
                          tooltip=tip("%s<br><span style='opacity:.7'>+%.1f min &middot; %.0f m from the route</span>"
                                      % (html.escape(c.poi.name), r.detour_min if r else 0, c.cross_track_m))).add_to(fmap)
    for c in trip["stops"]:
        n = chosen[c.poi.id]
        folium.Marker(c.poi.coord, icon=poi_pin(c.poi.category, 40, n, glow=True), z_index_offset=1000,
                      tooltip=name_tag("%d. %s" % (n, c.poi.name), th)).add_to(fmap)

    folium.Marker(route[0], icon=place_pin("start"), tooltip=name_tag(trip["from"], th)).add_to(fmap)
    folium.Marker(route[-1], icon=place_pin("end"), z_index_offset=900, tooltip=name_tag(trip["to"], th)).add_to(fmap)
    heading = bearing_deg(point_along(route, progress_m), point_along(route, progress_m + 80))
    folium.Marker(ahead[0], icon=car_icon(heading), z_index_offset=2000, tooltip=tip("You")).add_to(fmap)

    # overlays: turn banner, status pill, legend, ETA bar
    root.html.add_child(folium.Element(turn_banner(steps) + status_pill(trip) + legend(per_category, th, trip) + eta_bar(trip)))
    pts = list(ahead) + [c.poi.coord for c in trip["stops"]]
    fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]],
                    padding_top_left=(40, 170), padding_bottom_right=(90, 120))
    return figure


def turn_banner(steps: List[Step]) -> str:
    first = steps[0]
    nxt = next((s for s in steps[1:]), None)
    if first.kind == "go":
        main = '<div class="dist">%s</div><div class="road">%s %s</div>' % (km(first.dist_m), first.text, html.escape(first.road))
    else:
        main = '<div class="dist">%s</div><div class="road">%s</div>' % (first.text, html.escape(first.road))
    then = ""
    if nxt is not None:
        what = ("%s &middot; %s" % (nxt.text, html.escape(nxt.road))) if nxt.kind != "go" else "%s %s" % (nxt.text, html.escape(nxt.road))
        then = '<div class="then">Then %s %s</div>' % (arrow_svg(nxt, 18, "#dce9ff"), what)
    return '<div class="sr-banner"><div class="main">%s<div>%s</div></div>%s</div>' % (arrow_svg(first, 48), main, then)


def status_pill(trip: dict) -> str:
    """What the map is showing: live roads and places, or the approximate offline sketch (and why)."""
    if trip["live"]:
        dot, text, note = "#30d158", "LIVE roads and places", ""
    elif trip["road_live"]:
        dot, text, note = "#ffd60a", "LIVE roads &middot; sample places", trip["poi_source"]
    else:
        dot, text, note = "#ff9f0a", "OFFLINE &middot; approximate roads", trip["route_source"]
    reason = note.split(" - ", 1)[1] if " - " in note else ""
    extra = '<div class="sr-why">%s</div>' % html.escape(reason) if reason else ""
    return ('<div class="sr-status"><div><i style="background:%s"></i> %s &middot; %.0f m corridor</div>%s</div>'
            % (dot, text, trip["width_m"], extra))


def legend(per_category, th, trip) -> str:
    dot = '<i style="width:10px;height:10px;border-radius:50%%;background:%s"></i>%s'
    rows = ['<i style="width:18px;height:5px;border-radius:3px;background:%s"></i>Route' % th["route"] if trip["road_live"] else
            '<i style="width:18px;height:0;border-top:4px dashed %s"></i>Approx. route' % th["route"],
            '<i style="width:18px;height:0;border-top:3px dotted %s"></i>Your drive' % TRIP_COLOR,
            dot % (FILTERED_COLOR, "Filtered out")]
    rows += [dot % (CATEGORY_COLORS[c], CATEGORY_LABELS[c]) for c in per_category]
    return '<div class="sr-legend">%s</div>' % "<br>".join(rows)


def eta_bar(trip: dict) -> str:
    arrive = datetime.datetime.now() + datetime.timedelta(seconds=trip["total_s"])
    added = trip["total_s"] - trip["direct_s"]
    cells = [(arrive.strftime("%H:%M"), "arrival"), ("%.0f min" % max(1, round(trip["total_s"] / 60)), "time"),
             (km(trip["length_m"] - trip["progress_m"]), "to go")]
    html_ = "".join('<div><b>%s</b><span>%s</span></div>' % c for c in cells)
    if trip["stops"]:
        html_ += '<div class="plus"><b>+%.1f min</b><span>%d stop%s</span></div>' % (added / 60, len(trip["stops"]), "" if len(trip["stops"]) == 1 else "s")
    return '<div class="sr-eta">%s</div>' % html_


# ---------------------------------------------------------------------------
# Trip panel (right-hand cards)
# ---------------------------------------------------------------------------
def trip_panel(trip, per_category, sequence, steps: List[Step]) -> str:
    total_min = trip["total_s"] / 60
    arrive = (datetime.datetime.now() + datetime.timedelta(seconds=trip["total_s"])).strftime("%H:%M")
    out = ['<div class="sr-card sr-trip"><div class="sr-row"><div><div class="sr-eyebrow">%s &rarr; %s</div>'
           '<div class="sr-big">%.0f min <small>&middot; arrive %s</small></div>'
           '<div class="sr-sub">%s to go &middot; %s</div></div></div></div>'
           % (html.escape(trip["from"]), html.escape(trip["to"]), max(1, round(total_min)), arrive,
              km(trip["length_m"] - trip["progress_m"]),
              ("+%.1f min for %d stop%s" % ((trip["total_s"] - trip["direct_s"]) / 60, len(trip["stops"]), "" if len(trip["stops"]) == 1 else "s"))
              if trip["stops"] else "no stops added")]

    if not trip["road_live"]:
        reason = trip["route_source"].split(" - ", 1)[1] if " - " in trip["route_source"] else "live routing is switched off in Settings"
        out.append('<div class="sr-card sr-warn"><div class="sr-name">Approximate route - not the real roads</div>'
                   '<div class="sr-sub">Live OSRM routing did not work, so this is the offline sketch of Dubai\'s main roads '
                   '(straight lines between junctions). Reason: %s.<br>Check your connection, switch on live routing in '
                   'Settings, or run <code>python check_live.py</code> in the src folder to see what is blocked.</div></div>'
                   % html.escape(reason))

    # suggested stops
    ranked_of = {r.candidate.poi.id: r for d in per_category.values() for r in d["ranked"]}
    if trip["stops"]:
        items = []
        for i, c in enumerate(trip["stops"]):
            r = ranked_of.get(c.poi.id)
            items.append('<div class="sr-stop"><div class="sr-ic" style="background:%s">%s<b>%d</b></div><div class="sr-grow">'
                         '<div class="sr-name">%s</div><div class="sr-sub">%s ahead &middot; %.0f m off the road</div></div>'
                         '<div class="sr-add">+%.1f<small>min</small></div></div>'
                         % (CATEGORY_COLORS[c.poi.category], CATEGORY_ICONS[c.poi.category], i + 1, html.escape(c.poi.name),
                            km(max(0.0, c.along_track_m - trip["progress_m"])), c.cross_track_m, r.detour_min if r else 0))
        out.append('<div class="sr-card"><div class="sr-h">Stops on your way</div>%s</div>' % "".join(items))
    missing = [cat for cat, d in per_category.items() if not d["ranked"]]
    if missing:
        out.append('<div class="sr-card sr-warn"><div class="sr-name">No %s within %.0f m of the route ahead</div>'
                   '<div class="sr-sub">Widen the corridor in Route settings to search further from the road.</div></div>'
                   % (" or ".join(CATEGORY_LABELS[c].lower() for c in missing), trip["width_m"]))

    # directions
    rows = []
    for s in steps:
        if s.kind == "go":
            rows.append('<div class="sr-dir"><div class="sr-arrow">%s</div><div class="sr-grow">%s <b>%s</b></div><div class="sr-km">%s</div></div>'
                        % (arrow_svg(s, 22, "currentColor"), s.text, html.escape(s.road), km(s.dist_m)))
        elif s.kind == "stop":
            rows.append('<div class="sr-dir sr-dstop"><div class="sr-arrow">%s</div><div class="sr-grow"><b>%s</b> &middot; %s</div></div>'
                        % (arrow_svg(s, 22), s.text, html.escape(s.road)))
        else:
            rows.append('<div class="sr-dir"><div class="sr-arrow">%s</div><div class="sr-grow">Arrive at <b>%s</b></div></div>'
                        % (arrow_svg(s, 22), html.escape(s.road)))
    out.append('<div class="sr-card"><div class="sr-h">Directions</div>%s</div>' % "".join(rows))

    # the evidence: corridor vs radius, per stop type (collapsed)
    det = []
    for cat, d in per_category.items():
        det.append('<div class="sr-name" style="margin-top:10px">%s %s &middot; %d found, %d inside the %.0f m corridor</div>'
                   % (CATEGORY_ICONS[cat], CATEGORY_LABELS[cat], len(d["pois"]), len(d["kept"]), trip["width_m"]))
        if d["ranked"]:
            det.append('<table class="sr-t"><tr><th>Stop</th><th>Added</th><th>Off route</th><th>Where</th></tr>%s</table>'
                       % "".join('<tr><td>%s%s</td><td>+%.1f min</td><td>%.0f m</td><td>%s</td></tr>'
                                 % ("&#9733; " if i == 0 else "", html.escape(r.candidate.poi.name), r.detour_min,
                                    r.candidate.cross_track_m, where_text(r.candidate, trip["progress_m"]))
                                 for i, r in enumerate(d["ranked"][:5])))
        rp = d["radius"]
        if rp and (not d["ranked"] or rp.candidate.poi.id != d["ranked"][0].candidate.poi.id):
            why = "behind you" if rp.candidate.along_track_m < trip["progress_m"] else "%.0f m off the route" % rp.candidate.cross_track_m
            det.append('<div class="sr-sub">A radius search (nearest in a straight line) would pick <b>%s</b>: +%.1f min, %s.</div>'
                       % (html.escape(rp.candidate.poi.name), rp.detour_min, why))
    out.append('<details class="sr-card"><summary class="sr-h">Why these stops? Smart Corridor vs radius search</summary>%s</details>' % "".join(det))
    out.append('<div class="sr-src">Route: %s &middot; Stops: %s &middot; Times: %s &middot; Start: %s, destination: %s &middot; '
               'Decision time %.0f ms</div>' % (html.escape(trip["route_source"]), html.escape(trip["poi_source"]),
                                               html.escape(trip["cost_source"]), trip["how"][0], trip["how"][1], trip["elapsed"] * 1000))
    return "".join(out)


# ---------------------------------------------------------------------------
# Interface: CarPlay-style dock, full-height map, panel of big controls
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Interface: SmartRoute running on a CarPlay head unit
# ---------------------------------------------------------------------------
# The whole window is the CarPlay screen: a CarPlay sidebar on the left (clock,
# signal, the SmartRoute app icon, quick destinations, day/night, dashboard
# button), the navigation map filling the screen, and a CarPlay-style tabbed
# list panel (Drive / Where to / Settings) on the right.
PAGE_CSS = """
body, gradio-app { background: #000 !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; font-family: %(font)s !important;
  background: #000 !important; }
footer { display: none !important; }

/* the CarPlay screen fills the window */
#display { height: 100vh; min-height: 560px; flex-wrap: nowrap !important; gap: 0 !important; overflow: hidden;
  background: #000 !important; border: none !important; border-radius: 0 !important; }
#display > div { height: 100%; }

/* CarPlay sidebar */
#sidebar { background: rgba(28,28,30,.96) !important; min-width: 88px !important; max-width: 88px !important; padding: 16px 0 14px !important;
  gap: 12px !important; align-items: center; border-right: 1px solid rgba(255,255,255,.06) !important; border-radius: 0 !important; }
#sidebar .block { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }
#sidebar > * { flex-shrink: 0 !important; }
#status { color: #fff; text-align: center; font-family: %(font)s; }
#clock { display: block; font: 700 21px/1 %(font)s; letter-spacing: -.02em; }
#status small { display: block; margin-top: 4px; color: #aeaeb2; font: 700 11px %(font)s; letter-spacing: .02em; }
.appicon { width: 58px; height: 58px; border-radius: 15px; display: grid; place-items: center; font-size: 27px; color: #fff;
  box-shadow: 0 3px 10px rgba(0,0,0,.5); margin: 0 auto; position: relative; }
.appicon.active::before { content: ""; position: absolute; left: -15px; top: 17px; width: 5px; height: 24px; border-radius: 3px; background: #fff; }
.sr-app { background: linear-gradient(150deg, #64d2ff, #0a84ff 55%, #0040dd); }
.sbtn { min-width: 58px !important; max-width: 58px !important; height: 58px !important; border-radius: 15px !important;
  font-size: 26px !important; border: none !important; padding: 0 !important; color: #fff !important; margin: 0 auto !important;
  box-shadow: 0 3px 10px rgba(0,0,0,.5) !important; }
#s-home { background: linear-gradient(150deg, #5ac8fa, #007aff) !important; }
#s-work { background: linear-gradient(150deg, #ffd60a, #ff9500) !important; }
#s-theme { background: linear-gradient(150deg, #5e5ce6, #2c1f8f) !important; }
.slabel { color: #8e8e93; font: 600 10px %(font)s; text-align: center; margin-top: -9px !important; }
#sidebar .spacer { flex: 1 1 auto !important; }
#dashbtn { width: 46px; height: 46px; border-radius: 12px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
  padding: 11px; background: #2c2c2e; }
#dashbtn i { border-radius: 3px; background: #aeaeb2; }

/* the map fills the screen */
#mapcol { padding: 0 !important; gap: 0 !important; }
#mapcard { background: #0b0b0d !important; border: none !important; border-radius: 0 !important; padding: 0 !important; height: 100vh; min-height: 560px; overflow: hidden; }
#mapcard iframe { border-radius: 0 !important; height: 100vh !important; min-height: 560px; }
.sr-empty { height: %(h)dpx; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #8e8e93;
  font: 600 16px %(font)s; } .sr-empty div { font-size: 54px; }

/* CarPlay list panel with a tab bar */
#panel { background: #111113 !important; border-left: 1px solid rgba(255,255,255,.06) !important; border-radius: 0 !important;
  padding: 12px 14px !important; gap: 0 !important; height: 100vh; overflow-y: auto !important; flex-wrap: nowrap !important; }
#panel > * { flex-shrink: 0 !important; }
#tabs { background: transparent !important; border: none !important; }
#tabs .tab-nav { border: none !important; background: #2c2c2e; border-radius: 14px; padding: 4px; gap: 4px; margin-bottom: 12px;
  display: flex; }
#tabs .tab-nav button { flex: 1; border: none !important; border-radius: 10px !important; color: #aeaeb2 !important;
  font: 700 14px %(font)s !important; padding: 9px 4px !important; background: transparent !important; }
#tabs .tab-nav button.selected { background: #0a84ff !important; color: #fff !important; box-shadow: 0 2px 8px rgba(10,132,255,.4); }
#tabs .tabitem { background: transparent !important; border: none !important; padding: 0 !important; }
#panel .block, #panel .form, #panel fieldset { background: transparent !important; border: none !important; box-shadow: none !important; }
#panel label span, #panel .label-wrap span { color: #aeaeb2 !important; }
#panel input, #panel .wrap-inner, #panel .secondary-wrap { background: #2c2c2e !important; color: #fff !important;
  border-radius: 14px !important; border-color: #3a3a3c !important; font-size: 16px !important; }
#panel .wrap-inner { padding: 6px 10px !important; }
#from-box .wrap-inner { box-shadow: inset 4px 0 0 #0a84ff !important; }
#to-box .wrap-inner { box-shadow: inset 4px 0 0 #ff453a !important; }
#swap { background: #2c2c2e !important; color: #0a84ff !important; border: none !important; border-radius: 14px !important;
  font: 700 15px %(font)s !important; padding: 10px !important; }
#cats .wrap { gap: 8px !important; }
#cats label { background: #2c2c2e !important; border: 2px solid #3a3a3c !important; border-radius: 18px !important;
  padding: 10px 14px !important; color: #f2f2f7 !important; font-weight: 700 !important; font-size: 15px !important; }
#cats label.selected { background: #0a84ff !important; border-color: #64d2ff !important; }
#cats label span { color: inherit !important; } #cats label.selected, #cats label.selected span { color: #fff !important; }
#cats input { display: none !important; }
#go { background: linear-gradient(145deg, #4cd964, #1f9d49) !important; color: #fff !important; font-size: 20px !important;
  font-weight: 800 !important; border-radius: 20px !important; padding: 16px !important; border: none !important;
  box-shadow: 0 8px 22px rgba(48,209,88,.32); letter-spacing: .01em; margin-top: 6px; }
#panel input[type=checkbox] { width: 20px !important; height: 20px !important; border-radius: 6px !important; }
#panel input[type=checkbox]:checked { background-color: #30d158 !important; border-color: #30d158 !important;
  background-image: url("data:image/svg+xml,%%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%%3e%%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%%3e%%3c/svg%%3e") !important; }
#panel .gr-group, #panel .styler { background: transparent !important; }

/* trip cards */
.sr-card { display: block; background: #2c2c2e; border-radius: 20px; padding: 14px 16px; margin-bottom: 10px; color: #f2f2f7;
  font-family: %(font)s; }
.sr-trip { background: linear-gradient(145deg, #0a84ff, #0059c9); }
.sr-trip .sr-sub, .sr-trip .sr-eyebrow { color: #dce9ff; }
.sr-eyebrow { font: 700 12px %(font)s; text-transform: uppercase; letter-spacing: .06em; color: #aeaeb2; }
.sr-big { font: 800 34px/1.1 %(font)s; letter-spacing: -.02em; color: #fff; margin: 4px 0; }
.sr-big small { font: 600 16px %(font)s; opacity: .85; }
.sr-h { font: 800 17px %(font)s; color: #fff; margin-bottom: 8px; cursor: default; }
details summary.sr-h { cursor: pointer; margin-bottom: 0; }
.sr-sub { color: #aeaeb2; font-size: 13px; line-height: 1.4; }
.sr-name { color: #fff; font-weight: 700; font-size: 14.5px; line-height: 1.3; }
.sr-stop, .sr-dir { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-top: 1px solid rgba(255,255,255,.07); }
.sr-stop:first-of-type, .sr-dir:first-of-type { border-top: none; }
.sr-ic { position: relative; flex: none; width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; font-size: 22px; }
.sr-ic b { position: absolute; top: -6px; right: -6px; width: 20px; height: 20px; border-radius: 50%%; background: #fff; color: #111;
  font: 800 12px/20px %(font)s; text-align: center; }
.sr-grow { flex: 1; min-width: 0; color: #e5e5ea; font-size: 14px; }
.sr-add { color: #30d158; font: 800 20px %(font)s; text-align: right; } .sr-add small { font-size: 11px; margin-left: 2px; }
.sr-arrow { flex: none; width: 34px; height: 34px; border-radius: 10px; background: #3a3a3c; display: grid; place-items: center; color: #fff; }
.sr-dstop .sr-arrow { background: #1f9d49; }
.sr-km { color: #aeaeb2; font: 700 13px %(font)s; }
.sr-warn { background: #3a2a10; } .sr-warn .sr-name { color: #ffd60a; }
.sr-msg { background: #3a2a10; } .sr-msg .sr-h { color: #ffd60a; }
.sr-t { width: 100%%; border-collapse: collapse; margin: 6px 0; font-size: 12.5px; }
.sr-t th { color: #8e8e93; text-align: left; font-weight: 600; padding: 4px 6px; }
.sr-t td { color: #e5e5ea; padding: 5px 6px; border-top: 1px solid rgba(255,255,255,.07); }
.sr-src { color: #8e8e93; font: 500 11.5px/1.5 %(font)s; padding: 2px 4px 6px; }
""".replace("%(font)s", UI_FONT).replace("%(h)d", str(MAP_HEIGHT_PX)).replace("%%", "%")

PAGE_JS = """() => {
  document.body.classList.add('dark');
  const tick = () => {
    const el = document.getElementById('clock');
    if (el) el.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hour12: false});
  };
  tick(); setInterval(tick, 5000);
}"""

STATUS = """<div id="status"><span id="clock">--:--</span><small>&#9679;&#9679;&#9679;&#9675; 5G</small></div>"""
APP_ICON = """<div class="appicon sr-app active">&#10148;</div>"""
DASH_BUTTON = """<div id="dashbtn"><i></i><i></i><i></i><i></i></div>"""


def make_ui() -> gr.Blocks:
    places = list(data_loader.PLACES)
    home, office = places[1], places[0]
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="gray")
    cat_choices = [("%s %s" % (html.unescape(CATEGORY_ICONS[c]), CATEGORY_LABELS[c]), c) for c in data_loader.CATEGORIES]
    with gr.Blocks(title="SmartRoute · CarPlay", theme=theme, css=PAGE_CSS, js=PAGE_JS) as demo:
        with gr.Row(elem_id="display", equal_height=True):
            # CarPlay sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=88):
                gr.HTML(STATUS)
                gr.HTML(APP_ICON)
                go_home = gr.Button("\U0001F3E0", elem_id="s-home", elem_classes="sbtn")
                gr.HTML('<div class="slabel">Home</div>')
                go_work = gr.Button("\U0001F3E2", elem_id="s-work", elem_classes="sbtn")
                gr.HTML('<div class="slabel">Work</div>')
                flip = gr.Button("\U0001F317", elem_id="s-theme", elem_classes="sbtn")
                gr.HTML('<div class="slabel">Day/Night</div>')
                gr.HTML("", elem_classes="spacer")
                gr.HTML(DASH_BUTTON)
            # the SmartRoute app: map ...
            with gr.Column(scale=7, min_width=480, elem_id="mapcol"):
                map_html = gr.HTML(elem_id="mapcard")
            # ... and its CarPlay list panel
            with gr.Column(scale=3, min_width=350, elem_id="panel"):
                with gr.Tabs(elem_id="tabs", selected="drive") as tabs:
                    with gr.Tab("Drive", id="drive"):
                        result = gr.HTML()
                    with gr.Tab("Where to", id="where"):
                        origin = gr.Dropdown(places, value=DEFAULT_FROM, label="From", allow_custom_value=True,
                                             elem_id="from-box", info="Pick a place, type a name, or type lat, lon")
                        swap = gr.Button("⇅  Swap", elem_id="swap")
                        destination = gr.Dropdown(places, value=DEFAULT_TO, label="To", allow_custom_value=True, elem_id="to-box")
                        categories = gr.CheckboxGroup(cat_choices, value=["mosque", "supermarket"], label="Stops on the way",
                                                      elem_id="cats")
                        run = gr.Button("Go", variant="primary", elem_id="go")
                    with gr.Tab("Settings", id="settings"):
                        width = gr.Slider(50, 500, value=150, step=10, label="Corridor width (m)",
                                          info="How far from the road a stop may be")
                        progress = gr.Slider(0, 60, value=0, step=0.5, label="Distance already driven (km)",
                                             info="Stops more than 250 m behind you are left out")
                        live_pois = gr.Checkbox(value=True, label="Live places from OpenStreetMap (Overpass)")
                        live_osrm = gr.Checkbox(value=True, label="Live roads and times (OSRM)")
                        look = gr.Radio(["Night", "Day"], value="Night", label="Map style")

        inputs = [origin, destination, categories, width, progress, live_pois, live_osrm, look]
        outputs = [map_html, result]
        to_drive = lambda: gr.Tabs(selected="drive")
        run.click(plan_trip, inputs, outputs).then(to_drive, None, tabs)
        go_home.click(lambda: (office, home), None, [origin, destination]).then(plan_trip, inputs, outputs).then(to_drive, None, tabs)
        go_work.click(lambda: (home, office), None, [origin, destination]).then(plan_trip, inputs, outputs).then(to_drive, None, tabs)
        swap.click(lambda a, b: (b, a), [origin, destination], [origin, destination])
        flip.click(lambda t: "Day" if t == "Night" else "Night", look, look).then(plan_trip, inputs, outputs)
        for control in (width, progress, live_pois, live_osrm, look):
            control.input(plan_trip, inputs, outputs)
        demo.load(plan_trip, inputs, outputs)
    return demo


if __name__ == "__main__":
    make_ui().launch()
