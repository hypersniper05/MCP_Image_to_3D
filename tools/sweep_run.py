"""Sweep Idol's tuning parameters with everything else held fixed.

Splat count is the one thing this metric cannot rank across, so it is pinned at
262144 for every run and only one parameter moves at a time. Writes the job ids
to data/sweep.json for tools/sweep_report.py to score.

  IDOL_IMAGE=http://example.com/object.png \
  IDOL_BASE=http://127.0.0.1:8043 \
  python tools/sweep_run.py

`--image` must be a URL or a path the Idol server can read (not a host-only path).
"""
import argparse
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get("IDOL_BASE", "http://127.0.0.1:8043"))
ap.add_argument("--image", default=os.environ.get("IDOL_IMAGE", ""))
ap.add_argument("--out", default=os.environ.get("IDOL_SWEEP_OUT", "data/sweep.json"))
a = ap.parse_args()
BASE = a.base.rstrip("/")
IMAGE = a.image
if not IMAGE:
    raise SystemExit("set --image or IDOL_IMAGE to a URL or server-side image path")

FIXED = {"mode": "real", "num_gaussians": 262144, "seed": 42}
MATRIX = [("steps", v) for v in (25, 100)] + \
         [("cfg_strength", v) for v in (1.0, 2.0, 5.0, 7.0)] + \
         [("rescale_t", v) for v in (1.0, 2.0, 5.0)]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


results = []
# The baseline is generated too, so the sweep does not depend on a saved job id.
for name, value in [("baseline", None)] + MATRIX:
    params = {**FIXED, "steps": 50, "cfg_strength": 3.0, "rescale_t": 3.0}
    if value is not None:
        params[name] = value
    job = post("/jobs", {"image": IMAGE, **params})["id"]
    print(f"{name}={value if value is not None else '-'} -> {job}", flush=True)
    while True:
        time.sleep(5)
        state = get(f"/jobs/{job}")
        if state["state"] in ("done", "error"):
            break
    print(f"   {state['state']} {state.get('seconds')}s", flush=True)
    results.append({"name": name, "value": value if value is not None else "-",
                    "job": job, "seconds": state.get("seconds"),
                    "state": state["state"], "params": params})

with open(a.out, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1)
print(f"wrote {a.out} with {len(results)} runs")
