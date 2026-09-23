"""HTTPS helpers for the live data paths, with a curl fallback.

macOS's built-in Python 3.9 is linked against LibreSSL 2.8.3, which cannot
verify the Let's Encrypt certificate chains used by the OpenStreetMap services
(OSRM, Overpass, Nominatim), so ``requests`` fails there with an SSLError. The
system ``curl`` uses the operating system's own, up-to-date TLS stack, so when
``requests`` hits an SSL error the same request is repeated with curl.
Certificate verification stays on in both paths.
"""

import json
import shutil
import subprocess
from typing import Dict, Optional
from urllib.parse import urlencode

USER_AGENT = "SmartRoute-MSc-project/1.0 (educational demo)"
HEADERS = {"User-Agent": USER_AGENT}

# which transport served the last successful request: "requests" or "curl"
last_transport = "requests"


def _curl(url: str, timeout_s: float, form: Optional[Dict[str, str]] = None):
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not available for the SSL fallback")
    cmd = [curl, "-sS", "--fail", "--compressed", "--max-time", str(int(timeout_s)), "-A", USER_AGENT]
    for key, value in (form or {}).items():
        cmd += ["--data-urlencode", "%s=%s" % (key, value)]
    done = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=timeout_s + 5)
    if done.returncode != 0:
        raise RuntimeError("curl failed (%d): %s" % (done.returncode, done.stderr.strip()[:200]))
    return json.loads(done.stdout)


def get_json(url: str, params: Optional[Dict[str, str]] = None, timeout_s: float = 15.0):
    """GET a JSON document, falling back to curl on an SSL error."""
    global last_transport
    import requests

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout_s)
        resp.raise_for_status()
        last_transport = "requests"
        return resp.json()
    except requests.exceptions.SSLError:
        full = url + ("?" + urlencode(params) if params else "")
        data = _curl(full, timeout_s)
        last_transport = "curl"
        return data


def post_form_json(url: str, form: Dict[str, str], timeout_s: float = 30.0):
    """POST a form and read JSON back, falling back to curl on an SSL error."""
    global last_transport
    import requests

    try:
        resp = requests.post(url, data=form, headers=HEADERS, timeout=timeout_s)
        resp.raise_for_status()
        last_transport = "requests"
        return resp.json()
    except requests.exceptions.SSLError:
        data = _curl(url, timeout_s, form)
        last_transport = "curl"
        return data
