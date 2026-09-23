"""Unit tests for the geometry and Smart Corridor logic (Chapter 6).

Run from the project root:
    python -m unittest discover -s tests -v
No network access or third-party package is needed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from corridor import POI, SmartCorridor  # noqa: E402
from geometry import haversine_m, nearest_projection, route_length_m  # noqa: E402

DUBAI_MARINA = (25.0805, 55.1403)
DOWNTOWN_DUBAI = (25.1972, 55.2744)

# A simple due-east route of about 10 km.
EAST_ROUTE = [(25.2000, 55.2000), (25.2000, 55.2500), (25.2000, 55.3000)]


def poi(pid, lat, lon):
    return POI(pid, "POI " + pid, "supermarket", lat, lon)


class TestGeometry(unittest.TestCase):
    def test_haversine_known_distance(self):
        d = haversine_m(DUBAI_MARINA, DOWNTOWN_DUBAI)
        self.assertGreater(d, 15000)
        self.assertLess(d, 25000)

    def test_zero_distance_same_point(self):
        self.assertAlmostEqual(haversine_m(DOWNTOWN_DUBAI, DOWNTOWN_DUBAI), 0.0)

    def test_projection_on_straight_route(self):
        # about 55 m north of the midpoint of the route
        proj = nearest_projection(EAST_ROUTE, (25.2005, 55.2500))
        self.assertGreater(proj.cross_track_m, 0.0)
        self.assertLess(proj.cross_track_m, 100.0)
        self.assertAlmostEqual(proj.along_track_m, route_length_m(EAST_ROUTE) / 2, delta=50.0)

    def test_route_length_matches_sum_of_segments(self):
        total = haversine_m(EAST_ROUTE[0], EAST_ROUTE[1]) + haversine_m(EAST_ROUTE[1], EAST_ROUTE[2])
        self.assertAlmostEqual(route_length_m(EAST_ROUTE), total, places=6)


class TestSmartCorridor(unittest.TestCase):
    def setUp(self):
        self.corridor = SmartCorridor(EAST_ROUTE, corridor_width_m=150)

    def test_poi_inside_corridor_survives(self):
        near = poi("near", 25.20045, 55.2400)          # about 50 m off the route
        self.assertEqual([c.poi.id for c in self.corridor.filter([near])], ["near"])

    def test_poi_outside_corridor_is_rejected(self):
        far = poi("far", 25.2090, 55.2400)             # about 1 km off the route (side-street decoy)
        self.assertEqual(self.corridor.filter([far]), [])

    def test_behind_poi_excluded_beyond_tolerance(self):
        # About 1 km before the start, but only 20 m from the route's line extended backwards.
        behind = poi("behind", 25.20018, 55.1900)
        proj = nearest_projection(EAST_ROUTE, behind.coord)
        self.assertLess(proj.cross_track_m, 150.0)
        self.assertLess(proj.along_track_m, -self.corridor.behind_tolerance_m)
        self.assertEqual(self.corridor.filter([behind]), [])

    def test_survivors_sorted_by_progress_along_route(self):
        later = poi("later", 25.20030, 55.2800)
        sooner = poi("sooner", 25.19970, 55.2150)
        self.assertEqual([c.poi.id for c in self.corridor.filter([later, sooner])], ["sooner", "later"])

    def test_empty_pool_returns_empty_list(self):
        self.assertEqual(self.corridor.filter([]), [])

    def test_raises_on_single_point_route(self):
        with self.assertRaises(ValueError):
            SmartCorridor([(25.2, 55.2)], corridor_width_m=150)


if __name__ == "__main__":
    unittest.main()
