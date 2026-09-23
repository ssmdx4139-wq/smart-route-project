# SmartRoute

**An intelligent mid-journey navigation optimisation framework** – replacing radius-based
POI search with a direction-aware Smart Corridor engine for active driving routes (Dubai).

Conventional "search along route" ranks stops by straight-line distance from the car, so a
stop behind the driver, or down a side street off the main road, can outrank one sitting
conveniently ahead on the same carriageway. SmartRoute replaces that circle with a
two-stage pipeline:

1. **Smart Corridor** (`corridor.py`) – project every candidate onto the route polyline and
   keep only those within a configurable cross-track distance (`corridor_width_m`, default
   150 m) that are not more than 250 m behind the driver's progress.
2. **Detour costing** (`costing.py`) – rank the survivors by the extra time they add:
   `detour = (t_to + t_from) - direct_time_s + penalty`, with a 90 s U-turn penalty for
   candidates behind the driver. Times come from a live OSRM engine or, offline, a
   deterministic two-speed heuristic (30 km/h under 800 m, 80 km/h otherwise, plus 20 s per leg).

A sequencer (`sequencer.py`) orders up to nine stops (greedy nearest-neighbour or route order, whichever is quicker), and a
Gradio + Folium dashboard (`app.py`) shows the route, the corridor, every candidate and the
recommendation. Everything runs on open data (OpenStreetMap) and free, open-source routing.

## Layout

```
smartroute/
|-- src/
|   |-- geometry.py        # haversine distance, cross-track/along-track projection
|   |-- corridor.py        # Stage 1: the Smart Corridor geometric filter
|   |-- costing.py         # Stage 2: OSRM / heuristic detour-time ranking
|   |-- sequencer.py       # Stage 3: multi-stop sequencing (up to 9 stops)
|   |-- baseline_radius.py # control group: conventional radius-based search
|   |-- data_loader.py     # OSMnx / Overpass / OSRM live data plus an offline sample set
|   |-- dubai_roads.py     # offline approximate network of Dubai's main roads, with sample stops
|   |-- evaluate.py        # the fifty-trip benchmark (Chapter 7)
|   `-- app.py             # the Gradio and Folium interactive dashboard
|-- tests/
|   `-- test_corridor.py   # the unittest suite (Chapter 6)
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.9 or later. The core pipeline, the tests and the benchmark use only the standard
library; the dashboard and the live data paths need the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

`osmnx` (and the geospatial stack it brings) is only needed for `data_loader.load_live_graph`.
If it fails to install, the dashboard still works with just `gradio`, `folium` and `requests`:
`python3 -m pip install "gradio>=4,<5" "huggingface_hub<1.0" folium requests "urllib3<2"`.

## Running

```bash
python3 -m unittest discover -s tests -v   # 10 unit tests, no network needed
cd src
python3 evaluate.py                        # fifty-trip benchmark -> evaluation_results.json, figures/
python3 app.py                             # dashboard at http://127.0.0.1:7860
```

`python3 evaluate.py --osrm` re-runs the same fifty trips with live OSRM travel times
(Section 7.8 / further work).

## If the map says OFFLINE

Real road routes and real places need the live services (OSRM and Overpass). If the map's
status pill says **OFFLINE · approximate roads**, the route is only a sketch of Dubai's main
roads and will cut across junctions. The pill and the Drive tab show the reason; for a full
check run, from the `src` folder:

```bash
python check_live.py
```

It tests each live service and prints OK or the exact error.

## The dashboard

SmartRoute shown as an app open on an Apple CarPlay screen, filling the browser window:

* **CarPlay sidebar** on the left: clock and signal, the SmartRoute app icon (open), and quick
  buttons for Home (Office → Home), Work (Home → Office) and Day/Night map.
* **Full-screen map** with a Waze-style turn banner (next road, distance, and what comes after),
  an ETA bar (arrival time, minutes, distance, time added by stops), teardrop pins with an icon
  for each stop type, numbered pins for the suggested stops, a thick route with an outline, and
  a black muscle-car marker pointing along the route. A pill shows LIVE or OFFLINE data.
* **CarPlay tabbed panel** on the right:
  * *Drive* - the corridor width slider (the trip updates as you move it), trip cards (ETA, stops with the time each adds, directions with left/right turns)
    and a collapsed "Why these stops?" card comparing the Smart Corridor with a radius search.
  * *Where to* - From, Swap, To, stop-type buttons and Go (which jumps back to Drive).
  * *Settings* - distance already driven, live data and map style.

Controls:

* **From / To** – starts on the saved trip Office (Bay Square, Business Bay) → Home
  (Dubai Hills Estate). Pick another Dubai place, type a place name (looked up with OpenStreetMap Nominatim),
  or type coordinates as `25.2, 55.27`. The ⇅ button swaps them.
* **Stops on the way** – up to nine of supermarket, pharmacy, petrol station, mosque, park, cafe,
  restaurant, ATM and EV charging. Live places are cached per map tile in `cache/` for a week.
  With several, the best stop of each type is found and they are visited in whichever is quicker:
  greedy nearest-neighbour order or the order they come along the route (FR7).
* **Choosing a stop yourself** – tap any pin inside the corridor and press *Use this stop*; it replaces
  the automatic pick for that stop type and the order, times and route are recalculated. *Back to
  automatic* on the pin, or *Use automatic stops* under the trip, undoes it.
* **Go / Cancel** – *Go* (on *Where to*, or *Go with these stops* on *Drive*) starts the drive: the map
  then shows only the chosen stops. *Cancel* brings every candidate back so you can choose again;
  your picks are kept.
* **Distance already driven** – where the vehicle is now; stops more than 250 m behind it are
  excluded (FR4).
* **Corridor width** – 50 m to 3 km (default 150 m), half-width either side of the route. With a wide
  corridor the 40 most promising places of each stop type (by the offline estimate) get live drive times.
* **Use live Overpass POI data** (on by default) – real POIs along the route in one Overpass query
  (tries the overpass-api.de servers in turn); falls back to the offline sample if the query fails.
* **Use live OSRM routing** (on by default) – the real road route, road names and travel times;
  falls back to the offline road network and the heuristic cost model if unavailable.

The map shows the route ahead (blue) and already driven (grey), the corridor (faint band),
candidates inside the corridor (category colour) and filtered out (grey), the suggested stops
(numbered stars) and your drive through them (dotted green). The result panel lists the stops
in visiting order with the added time for each and in total, a table per stop type, what a
radius search would have picked instead, and which data and routing sources produced the
numbers, plus road-by-road directions with the stops in place. If nothing survives the
corridor it says so and suggests widening it.

Without internet (or with the live options off), routes run over `dubai_roads.py`, an
approximate hand-built network of Dubai's main roads (Sheikh Zayed Rd, Al Khail Rd, Sheikh
Mohammed bin Zayed Rd, Al Ain Rd, Hessa St, Umm Suqeim Rd, Ras Al Khor Rd, Airport Rd and
connectors), with sample stops generated along each road, so every listed place still gets a
route and stops. Its coordinates are approximate and its stops are illustrative, not real
businesses; the result panel always says which source was used.
