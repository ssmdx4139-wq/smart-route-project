"""The fifty-trip benchmark (Chapter 7).

Compares SmartRoute (Smart Corridor filter, then detour-cost ranking) with a
conventional radius search on the same trips, the same candidate pools and the
same cost model.

Trips are generated procedurally from five straight-line corridors that
approximate Dubai arterial roads, with a fixed random seed so exactly the same
trips are produced on every run (NFR3). Each trip's candidate pool
deliberately contains the three failure patterns from Section 1.2:

  * convenient  - ahead of the vehicle and close to the route
  * behind      - slightly closer in a straight line, but behind the vehicle
  * side_street - very close in a straight line, but far off the route
  * noise (x2)  - scattered at random in the wider search area

Both methods are costed with the offline HeuristicBackend by default, so the
benchmark runs identically without network access. Pass --osrm to cost with a
live OSRM instance instead (Section 7.8).

Run from the src folder:
    python evaluate.py            # prints the summary, writes evaluation_results.json
    python evaluate.py --osrm     # same trips, live OSRM travel times
"""

import argparse
import json
import math
import os
import random
import statistics
import time
from typing import Dict, List, Tuple

from baseline_radius import radius_search
from corridor import POI, SmartCorridor
from costing import DetourCostEngine, HeuristicBackend, OSRMBackend
from geometry import LatLon, from_local_xy, haversine_m, to_local_xy
from sequencer import added_time_s, greedy_sequence

N_TRIPS = 50
SEED = 42
CORRIDOR_WIDTH_M = 150.0
RADIUS_M = 5000.0
CATEGORIES = ["supermarket", "pharmacy", "petrol station", "mosque", "park"]
HERE = os.path.dirname(os.path.abspath(__file__))

# Straight-line approximations of five Dubai arterial roads: (start, end) as (lat, lon).
CORRIDORS: Dict[str, Tuple[LatLon, LatLon]] = {
    "Sheikh Zayed Road": ((25.0443, 55.1170), (25.2270, 55.2860)),
    "Al Khail Road": ((25.0450, 55.1950), (25.1920, 55.3380)),
    "Emirates Road": ((25.0300, 55.2800), (25.2400, 55.4600)),
    "Hessa Street / Al Quoz": ((25.0960, 55.1640), (25.0650, 55.2600)),
    "Al Ain Road": ((25.1800, 55.3400), (25.0600, 55.4700)),
}


def point_on_line(start: LatLon, end: LatLon, along_m: float, offset_m: float) -> LatLon:
    """A point ``along_m`` metres from start towards end, ``offset_m`` metres to the side."""
    ex, ey = to_local_xy(start, end)
    length = math.hypot(ex, ey)
    ux, uy = ex / length, ey / length
    return from_local_xy(start, (ux * along_m + uy * offset_m, uy * along_m - ux * offset_m))


def make_trip(rng: random.Random, index: int) -> Dict:
    """One synthetic trip: a sub-span of a corridor, a category and a candidate pool."""
    corridor_name = rng.choice(list(CORRIDORS))
    c_start, c_end = CORRIDORS[corridor_name]
    c_len = haversine_m(c_start, c_end)
    trip_len = rng.uniform(6000.0, min(14000.0, c_len))
    start_at = rng.uniform(0.0, c_len - trip_len)
    origin = point_on_line(c_start, c_end, start_at, 0.0)
    destination = point_on_line(c_start, c_end, start_at + trip_len, 0.0)
    category = rng.choice(CATEGORIES)
    side = lambda: rng.choice((-1.0, 1.0))  # noqa: E731

    def poi(kind: str, along: float, offset: float) -> POI:
        lat, lon = point_on_line(origin, destination, along, offset)
        return POI("t%02d-%s" % (index, kind), "%s (%s)" % (category.title(), kind.replace("_", " ")),
                   category, lat, lon, {"pattern": kind.rstrip("12")})

    pool = [
        poi("convenient", rng.uniform(1500.0, 4000.0), side() * rng.uniform(20.0, 120.0)),
        poi("behind", -rng.uniform(300.0, 1200.0), side() * rng.uniform(10.0, 100.0)),
        poi("side_street", rng.uniform(0.0, 800.0), side() * rng.uniform(600.0, 1200.0)),
        poi("noise1", rng.uniform(-3000.0, trip_len + 3000.0), rng.uniform(-3000.0, 3000.0)),
        poi("noise2", rng.uniform(-3000.0, trip_len + 3000.0), rng.uniform(-3000.0, 3000.0)),
    ]
    return {"index": index, "corridor": corridor_name, "category": category, "origin": origin,
            "destination": destination, "trip_length_m": trip_len, "pool": pool}


def run_trip(trip: Dict, engine: DetourCostEngine) -> Dict:
    """Run SmartRoute and the radius baseline on one trip, both costed by the same engine."""
    origin, destination = trip["origin"], trip["destination"]
    corridor = SmartCorridor([origin, destination], CORRIDOR_WIDTH_M)

    started = time.perf_counter()
    survivors = corridor.filter(trip["pool"])
    ranked = engine.rank(origin, destination, survivors)
    latency_s = time.perf_counter() - started

    smart = ranked[0] if ranked else None
    radius_hits = radius_search(trip["pool"], origin, RADIUS_M)
    radius = engine.evaluate(origin, destination, corridor.project(radius_hits[0][0])) if radius_hits else None

    # Sequencing is exercised on every trip: the chosen stop as a one-stop sequence.
    sequence_added_s = None
    if smart:
        seq = greedy_sequence(origin, destination, [smart.candidate], engine)
        sequence_added_s = added_time_s(seq, origin, destination, engine)

    smart_s = smart.detour_s if smart else None
    radius_s = radius.detour_s if radius else None
    reduction = (radius_s - smart_s) / radius_s * 100.0 if smart and radius and radius_s > 0 else None
    return {
        "trip": trip["index"],
        "corridor": trip["corridor"],
        "category": trip["category"],
        "trip_length_km": round(trip["trip_length_m"] / 1000.0, 2),
        "candidates": len(trip["pool"]),
        "survivors": len(survivors),
        "smartroute_choice": smart.candidate.poi.tags["pattern"] if smart else None,
        "radius_choice": radius.candidate.poi.tags["pattern"] if radius else None,
        "smartroute_detour_s": round(smart_s, 2) if smart else None,
        "radius_detour_s": round(radius_s, 2) if radius else None,
        "detour_time_reduction_pct": round(reduction, 2) if reduction is not None else None,
        "smartroute_cross_track_m": round(smart.candidate.cross_track_m, 1) if smart else None,
        "sequence_added_time_s": round(sequence_added_s, 2) if sequence_added_s is not None else None,
        "latency_s": latency_s,
        "cost_source": engine.backend.name,
    }


def summarise(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r["detour_time_reduction_pct"] is not None]
    smart = [r["smartroute_detour_s"] for r in ok]
    radius = [r["radius_detour_s"] for r in ok]
    red = [r["detour_time_reduction_pct"] for r in ok]
    lat = [r["latency_s"] for r in rows]
    return {
        "n_trips": len(rows),
        "mean_smartroute_detour_s": round(statistics.mean(smart), 2),
        "mean_radius_detour_s": round(statistics.mean(radius), 2),
        "mean_detour_time_reduction_pct": round(statistics.mean(red), 2),
        "median_detour_time_reduction_pct": round(statistics.median(red), 2),
        "trips_where_smartroute_beat_radius": sum(1 for r in ok if r["smartroute_detour_s"] < r["radius_detour_s"]),
        "mean_latency_s": round(statistics.mean(lat), 7),
        "max_latency_s": round(max(lat), 7),
    }


def breakdowns(rows: List[Dict]) -> Dict:
    def group(key):
        out = {}
        for value in sorted({r[key] for r in rows}):
            reds = [r["detour_time_reduction_pct"] for r in rows if r[key] == value and r["detour_time_reduction_pct"] is not None]
            out[value] = {"trips": len(reds), "mean_reduction_pct": round(statistics.mean(reds), 1),
                          "min_reduction_pct": round(min(reds), 1), "max_reduction_pct": round(max(reds), 1)}
        return out

    def spread(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean_s": round(statistics.mean(vals), 2), "std_s": round(statistics.pstdev(vals), 2),
                "min_s": round(min(vals), 2), "max_s": round(max(vals), 2)}

    def choices(key):
        counts = {}
        for r in rows:
            counts[r[key]] = counts.get(r[key], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    return {
        "by_category": group("category"),
        "by_corridor": group("corridor"),
        "smartroute_spread": spread("smartroute_detour_s"),
        "radius_spread": spread("radius_detour_s"),
        "smartroute_choices": choices("smartroute_choice"),
        "radius_choices": choices("radius_choice"),
    }


# ---------------------------------------------------------------------------
# Figures 7.1-7.3 as plain SVG (no plotting library needed)
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SMART_COLOR, RADIUS_COLOR = "#2a78d6", "#eb6834"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def _svg(width: int, height: int, title: str, body: List[str]) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" %s>'
            '<rect width="100%%" height="100%%" fill="%s"/>'
            '<text x="24" y="32" font-size="17" font-weight="700" fill="%s">%s</text>%s</svg>'
            % (width, height, width, height, FONT, SURFACE, INK, title, "".join(body)))


def _nice_max(v: float) -> float:
    step = 10 ** math.floor(math.log10(max(v, 1e-9)))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * step >= v:
            return m * step
    return 10 * step


def figure_mean_detour(summary: Dict) -> str:
    """Figure 7.1: mean detour time per method (vertical bars)."""
    vals = [("SmartRoute", summary["mean_smartroute_detour_s"], SMART_COLOR),
            ("Radius baseline", summary["mean_radius_detour_s"], RADIUS_COLOR)]
    w, h, left, top, bottom = 560, 380, 70, 70, 320
    top_val = _nice_max(max(v for _, v, _ in vals) * 1.1)
    y = lambda v: bottom - (bottom - top) * v / top_val  # noqa: E731
    body = []
    for i in range(5):
        v = top_val * i / 4
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>' % (left, w - 30, y(v), y(v), GRID if i else BASE))
        body.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end">%g s</text>' % (left - 8, y(v) + 4, MUTED, v))
    bw = 110
    for i, (name, v, color) in enumerate(vals):
        x = left + 70 + i * 200
        body.append('<path d="M%.1f %.1f V%.1f a4 4 0 0 1 4 -4 H%.1f a4 4 0 0 1 4 4 V%.1f Z" fill="%s"/>'
                    % (x, bottom, y(v) + 4, x + bw - 4, bottom, color))
        body.append('<text x="%.1f" y="%.1f" font-size="14" font-weight="700" fill="%s" text-anchor="middle">%.1f s</text>' % (x + bw / 2, y(v) - 8, INK, v))
        body.append('<text x="%.1f" y="%d" font-size="13" fill="%s" text-anchor="middle">%s</text>' % (x + bw / 2, bottom + 22, INK2, name))
    body.append('<text x="24" y="52" font-size="12" fill="%s">Mean detour time added per trip, %d trips, %s costs</text>' % (MUTED, summary["n_trips"], summary.get("cost_source", "")))
    return _svg(w, h, "Figure 7.1  Mean detour time by method", body)


def figure_spread(rows: List[Dict]) -> str:
    """Figure 7.2: every trip's detour time for each method (strip plot with median)."""
    series = [("SmartRoute", [r["smartroute_detour_s"] for r in rows if r["smartroute_detour_s"] is not None], SMART_COLOR),
              ("Radius baseline", [r["radius_detour_s"] for r in rows if r["radius_detour_s"] is not None], RADIUS_COLOR)]
    w, h, left, right, top = 640, 300, 140, 610, 70
    max_v = _nice_max(max(max(v) for _, v, _ in series) * 1.05)
    x = lambda v: left + (right - left) * v / max_v  # noqa: E731
    body = []
    for i in range(6):
        v = max_v * i / 5
        body.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s"/>' % (x(v), x(v), top, h - 50, GRID))
        body.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">%g s</text>' % (x(v), h - 30, MUTED, v))
    rng = random.Random(7)
    for j, (name, vals, color) in enumerate(series):
        cy = top + 45 + j * 90
        body.append('<text x="%d" y="%d" font-size="13" fill="%s" text-anchor="end">%s</text>' % (left - 12, cy + 4, INK2, name))
        for v in vals:
            body.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s" fill-opacity="0.7" stroke="%s" stroke-width="1"/>' % (x(v), cy + rng.uniform(-16, 16), color, SURFACE))
        med = statistics.median(vals)
        body.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s" stroke-width="2"/>' % (x(med), x(med), cy - 24, cy + 24, INK))
        body.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">median %.0f s</text>' % (x(med), cy - 28, INK, med))
    body.append('<text x="24" y="52" font-size="12" fill="%s">One dot per trip; vertical line = median</text>' % MUTED)
    return _svg(w, h, "Figure 7.2  Spread of detour times across all trips", body)


def figure_by_category(extra: Dict) -> str:
    """Figure 7.3: mean detour-time reduction by requested category (horizontal bars)."""
    cats = sorted(extra["by_category"].items(), key=lambda kv: -kv[1]["mean_reduction_pct"])
    w, left, right, top, rowh = 600, 150, 540, 70, 44
    h = top + rowh * len(cats) + 40
    x = lambda v: left + (right - left) * max(0.0, v) / 100.0  # noqa: E731
    body = []
    for v in (0, 25, 50, 75, 100):
        body.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s"/>' % (x(v), x(v), top - 6, h - 34, GRID if v else BASE))
        body.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">%d%%</text>' % (x(v), h - 16, MUTED, v))
    for i, (name, d) in enumerate(cats):
        cy = top + i * rowh
        v = d["mean_reduction_pct"]
        body.append('<path d="M%.1f %.1f H%.1f a4 4 0 0 1 4 4 V%.1f a4 4 0 0 1 -4 4 H%.1f Z" fill="%s"/>'
                    % (x(0), cy, x(v) - 4, cy + 20, x(0), SMART_COLOR))
        body.append('<text x="%d" y="%.1f" font-size="13" fill="%s" text-anchor="end">%s (%d)</text>' % (left - 10, cy + 15, INK2, name.title(), d["trips"]))
        body.append('<text x="%.1f" y="%.1f" font-size="13" font-weight="700" fill="%s">%.1f%%</text>' % (x(v) + 8, cy + 15, INK, v))
    body.append('<text x="24" y="52" font-size="12" fill="%s">Mean reduction in detour time vs the radius baseline (number of trips in brackets)</text>' % MUTED)
    return _svg(w, h, "Figure 7.3  Detour-time reduction by stop category", body)


def write_figures(summary: Dict, rows: List[Dict], extra: Dict, folder: str) -> List[str]:
    os.makedirs(folder, exist_ok=True)
    files = {"figure_7_1_mean_detour.svg": figure_mean_detour(summary),
             "figure_7_2_spread.svg": figure_spread(rows),
             "figure_7_3_by_category.svg": figure_by_category(extra)}
    for name, svg in files.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(svg)
    return [os.path.join(folder, n) for n in files]


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartRoute fifty-trip benchmark")
    parser.add_argument("--osrm", action="store_true", help="cost detours with a live OSRM instance")
    parser.add_argument("--osrm-url", default="https://router.project-osrm.org")
    parser.add_argument("--trips", type=int, default=N_TRIPS)
    args = parser.parse_args()

    backend = OSRMBackend(args.osrm_url) if args.osrm else HeuristicBackend()
    engine = DetourCostEngine(backend)
    rng = random.Random(SEED)
    trips = [make_trip(rng, i + 1) for i in range(args.trips)]
    rows = [run_trip(t, engine) for t in trips]

    summary = summarise(rows)
    summary["cost_source"] = backend.name
    extra = breakdowns(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "cost_source"}, indent=2))
    print("\nCost source: %s   corridor width: %.0f m   seed: %d" % (backend.name, CORRIDOR_WIDTH_M, SEED))

    print("\nBy category:")
    for name, d in sorted(extra["by_category"].items(), key=lambda kv: -kv[1]["mean_reduction_pct"]):
        print("  %-16s %2d trips  mean %5.1f%%  range %5.1f%% to %5.1f%%" % (name, d["trips"], d["mean_reduction_pct"], d["min_reduction_pct"], d["max_reduction_pct"]))
    print("\nBy corridor:")
    for name, d in sorted(extra["by_corridor"].items(), key=lambda kv: -kv[1]["mean_reduction_pct"]):
        print("  %-24s %2d trips  mean %5.1f%%" % (name, d["trips"], d["mean_reduction_pct"]))
    print("\nWhat each method selected:")
    print("  SmartRoute: %s" % ", ".join("%s %d" % kv for kv in extra["smartroute_choices"].items()))
    print("  Radius:     %s" % ", ".join("%s %d" % kv for kv in extra["radius_choices"].items()))
    s, r = extra["smartroute_spread"], extra["radius_spread"]
    print("\nDetour spread: SmartRoute sd %.1f s (min %.1f, max %.1f); radius sd %.1f s (min %.1f, max %.1f)"
          % (s["std_s"], s["min_s"], s["max_s"], r["std_s"], r["min_s"], r["max_s"]))

    out = os.path.join(HERE, "evaluation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "breakdowns": extra, "trips": rows}, f, indent=2)
    figs = write_figures(summary, rows, extra, os.path.join(HERE, "figures"))
    print("\nWrote %s and %d figures in %s" % (os.path.basename(out), len(figs), os.path.join(HERE, "figures")))


if __name__ == "__main__":
    main()
