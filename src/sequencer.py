"""Stage 3: greedy multi-stop sequencing (FR7).

When a trip includes more than one requested stop, the visiting order is
chosen with a greedy nearest-neighbour method: from the current position, go
to whichever remaining stop can be reached in the least time, then repeat,
finishing with the leg to the destination. It is easy to explain ("closest
remaining stop next") and, for the two to four stops of a typical commute,
close to the optimum. best_sequence also costs the order the stops come
along the route and keeps whichever is quicker, which matters for longer
lists of up to nine stops.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from corridor import CorridorCandidate
from costing import DetourCostEngine
from geometry import LatLon


@dataclass
class SequencedStop:
    """One leg of a sequenced trip."""

    label: str                             # stop name, or "Destination" for the final leg
    stop: Optional[CorridorCandidate]      # None for the final leg to the destination
    leg_time_s: float                      # time for this leg
    cumulative_s: float                    # running total from the origin


def greedy_sequence(origin: LatLon, destination: LatLon, stops: Sequence[CorridorCandidate],
                    engine: DetourCostEngine) -> List[SequencedStop]:
    """Order ``stops`` greedily by travel time and report each leg and the running total."""
    backend = engine.backend
    remaining = list(stops)
    position = origin
    ordered: List[CorridorCandidate] = []
    while remaining:
        times = [backend.route_time_s(position, s.poi.coord) for s in remaining]
        best = min(range(len(remaining)), key=lambda i: times[i])
        ordered.append(remaining.pop(best))
        position = ordered[-1].poi.coord
    return fixed_sequence(origin, destination, ordered, engine)


def fixed_sequence(origin: LatLon, destination: LatLon, stops: Sequence[CorridorCandidate],
                   engine: DetourCostEngine) -> List[SequencedStop]:
    """Legs and running totals for visiting ``stops`` in the order given."""
    backend = engine.backend
    position, total = origin, 0.0
    sequence: List[SequencedStop] = []
    for stop in stops:
        leg = backend.route_time_s(position, stop.poi.coord)
        total += leg
        sequence.append(SequencedStop(stop.poi.name, stop, leg, total))
        position = stop.poi.coord
    last = backend.route_time_s(position, destination)
    sequence.append(SequencedStop("Destination", None, last, total + last))
    return sequence


def best_sequence(origin: LatLon, destination: LatLon, stops: Sequence[CorridorCandidate],
                  engine: DetourCostEngine) -> List[SequencedStop]:
    """The quicker of the greedy order and the order the stops come along the route.

    With many stops (up to nine) nearest-neighbour can zig-zag back and forth
    along the corridor; visiting them in route order never does, so both are
    costed and the faster one is kept.
    """
    greedy = greedy_sequence(origin, destination, stops, engine)
    along = fixed_sequence(origin, destination, sorted(stops, key=lambda c: c.along_track_m), engine)
    return along if along[-1].cumulative_s < greedy[-1].cumulative_s else greedy


def added_time_s(sequence: Sequence[SequencedStop], origin: LatLon, destination: LatLon,
                 engine: DetourCostEngine) -> float:
    """Extra time the whole sequence adds compared with driving straight to the destination."""
    return sequence[-1].cumulative_s - engine.backend.route_time_s(origin, destination)
