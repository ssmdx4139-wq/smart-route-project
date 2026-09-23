"""Check that the live services SmartRoute uses can be reached from this computer.

Run from the src folder:
    python check_live.py

It tries live OSRM routing, the OSRM table service, Overpass (places) and
Nominatim (place search), and prints OK or the exact error for each one, so a
failed live mode can be diagnosed instead of silently falling back.
"""

import ssl
import sys
import time

import data_loader
import net

A, B = data_loader.PLACES["Bay Square, Business Bay"], data_loader.PLACES["Dubai Hills Estate"]


def check(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        print("  OK    %-28s %5.1f s  %s  [via %s]" % (name, time.perf_counter() - started, detail, net.last_transport))
        return True
    except Exception as exc:
        print("  FAIL  %-28s %5.1f s  %s: %s" % (name, time.perf_counter() - started, type(exc).__name__, exc))
        return False


def main() -> None:
    import requests
    import urllib3

    print("Python %s | requests %s | urllib3 %s | %s" % (sys.version.split()[0], requests.__version__,
                                                       urllib3.__version__, ssl.OPENSSL_VERSION))
    ok = True
    for server in data_loader.OSRM_SERVERS:
        def route(server=server):
            d = net.get_json("%s/route/v1/driving/%.5f,%.5f;%.5f,%.5f" % (server, A[1], A[0], B[1], B[0]),
                             params={"overview": "false"}, timeout_s=15)["routes"][0]
            return "%.1f km, %.1f min" % (d["distance"] / 1000, d["duration"] / 60)
        ok &= check("OSRM route  " + server.split("/")[2], route)
    ok &= check("OSRM route with road names", lambda: "%d road steps" % len(data_loader.osrm_route([A, B]).steps))
    for url in data_loader.OVERPASS_MIRRORS:
        def overpass(url=url):
            q = '[out:json][timeout:25];nwr["shop"="supermarket"](around:300,%.5f,%.5f);out center 5;' % A
            return "%d places near Bay Square" % len(net.post_form_json(url, {"data": q}, 30).get("elements", []))
        check("Overpass    " + url.split("/")[2], overpass)
    ok &= check("Nominatim place search", lambda: str(data_loader.geocode("Dubai Mall")))
    print("\nAll live services reachable." if ok else
          "\nSome live services failed - copy these lines to Claude so the cause can be fixed.")


if __name__ == "__main__":
    main()
