"""End-to-end check: one image in, view + download links out.

  python tools/test_e2e.py --image data/test/fox.png
  python tools/test_e2e.py --image https://example.com/x.png --mode real --gaussians 524288
  python tools/test_e2e.py --image fox.png --base http://127.0.0.1:8043 --seed 7

Waits for the model to become ready, submits, polls, and prints the two links.
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import requests


def wait_ready(base: str, timeout: int) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"{base}/health", timeout=10).json()
        except Exception:
            print(f"  waiting for server... ({time.time() - t0:.0f}s)", flush=True)
            time.sleep(5)
            continue
        if h.get("ready"):
            print(f"  server ready: {h['gpu'].get('name', 'no gpu')}")
            return h
        if h.get("state") == "error":
            sys.exit(f"server failed to load the model: {h.get('error')}")
        print(f"  model state: {h.get('state')} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(5)
    sys.exit(f"server not ready after {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path on the server, local path, or URL")
    ap.add_argument("--base", default="http://127.0.0.1:8043")
    ap.add_argument("--mode", default="real", choices=["synthetic", "real"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    wait_ready(base, a.timeout)

    img = a.image
    if img.startswith(("http://", "https://")):
        body = {"image": img}
    elif Path(img).exists():  # upload the bytes, so it works from the host too
        raw = Path(img).read_bytes()
        uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        body = {"image": uri}
    else:
        body = {"image": img}  # assume a path the server can read
    body["params"] = {"mode": a.mode, "seed": a.seed, "steps": a.steps,
                      "num_gaussians": a.gaussians}

    print(f"submitting {a.image} (mode={a.mode} seed={a.seed} "
          f"steps={a.steps} gaussians={a.gaussians})")
    r = requests.post(f"{base}/jobs", json=body, timeout=120)
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:400]}")
    job_id = r.json()["id"]
    print(f"  job {job_id}")

    t0 = time.time()
    last = None
    while time.time() - t0 < a.timeout:
        j = requests.get(f"{base}/jobs/{job_id}", timeout=30).json()
        if j["state"] != last:
            print(f"  {j['state']}", flush=True)
            last = j["state"]
        if j["state"] == "done":
            print(f"\ndone in {j['seconds']}s, {j['num_gaussians']:,} splats")
            print(f"  view     {j['view_url']}")
            print(f"  download {j['download_url']}")
            print(f"  input    {j['input_image_url']}")
            print(f"  camera   {j['camera']}")
            ply = requests.get(j["download_url"], timeout=120).content
            print(f"  ply      {len(ply):,} bytes, header={ply[:11].decode(errors='replace')!r}")
            return
        if j["state"] == "error":
            sys.exit(f"failed: {j['error']}")
        time.sleep(3)
    sys.exit(f"job {job_id} not finished after {a.timeout}s")


if __name__ == "__main__":
    main()
