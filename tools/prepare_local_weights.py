"""Build data/idol-local from the downloaded Model Hub snapshot.

We copy the weights out of the HF cache into a plain folder for two reasons:
  1. `IdolPipeline.from_pretrained()` accepts a local folder, so the container needs
     no Model Hub access at runtime.
  2. The stock config.json asks for `black-forest-labs/FLUX.2-dev`, which is
     license-gated. `FLUX.2-klein-4B` ships the same AutoencoderKLFlux2 VAE
     (same tensors, including the bn.running_mean/var that idol/conditioning.py
     reads) under Apache-2.0 with no gate, so we point the config at it instead.

Run once after tools/prefetch.sh, from any directory:
    python tools/prepare_local_weights.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hf-cache" / "hub" / "models--speridlabs--idol"
DEST = ROOT / "data" / "idol-local"
VAE_REPO = "black-forest-labs/FLUX.2-klein-4B"


def find_snapshot() -> Path:
    snapshots = sorted((CACHE / "snapshots").glob("*"))
    for snap in snapshots:
        if (snap / "config.json").exists() and (snap / "denoiser.safetensors").exists():
            return snap
    sys.exit(f"no complete idol snapshot under {CACHE} - run tools/prefetch.sh first")


def main() -> None:
    snap = find_snapshot()
    print(f"snapshot: {snap.name}")
    DEST.mkdir(parents=True, exist_ok=True)

    for item in snap.iterdir():
        if item.is_symlink():  # cache stores real files on Windows, symlinks elsewhere
            item = item.resolve()
        target = DEST / item.name
        if item.is_dir():
            continue
        if target.exists() and target.stat().st_size == item.stat().st_size:
            print(f"  ok  {item.name}")
            continue
        print(f"  copy {item.name} ({item.stat().st_size / 1e6:.0f} MB)")
        shutil.copy2(item, target)

    cfg_path = DEST / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    before = dict(cfg.get("image_encoders", {}))
    cfg.setdefault("image_encoders", {})["flux2_vae"] = VAE_REPO
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  image_encoders: {before} -> {cfg['image_encoders']}")
    print(f"done: {DEST}")


if __name__ == "__main__":
    main()
