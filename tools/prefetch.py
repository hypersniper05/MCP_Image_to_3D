"""Download the Idol weights and image encoders into data/hf-cache.

This is the cross-platform version of the old shell-only prefetch step. It
uses huggingface_hub directly, so it works the same on Windows, Linux and
macOS. `tools/prefetch.sh` is a thin wrapper around it.

Before the DINOv3 download works, accept the model's terms on Hugging Face and
log in once (`hf auth login`) or set HF_TOKEN.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
HF_HOME = ROOT / "data" / "hf-cache"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

REPOS = [
    ("speridlabs/idol", {"ignore_patterns": ["assets/*"]}),
    ("black-forest-labs/FLUX.2-klein-4B", {"allow_patterns": ["vae/*"]}),
    ("facebook/dinov3-vith16plus-pretrain-lvd1689m", {"ignore_patterns": ["onnx/*"]}),
]


def download(repo_id: str, kwargs: dict) -> None:
    print(f"=== {repo_id} ===", flush=True)
    path = snapshot_download(repo_id, cache_dir=str(HF_HOME), **kwargs)
    print(f"OK {repo_id} -> {path}", flush=True)


def main() -> int:
    try:
        for repo_id, kwargs in REPOS:
            download(repo_id, kwargs)
    except Exception as exc:
        print(f"\nDownload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "If DINOv3 is gated, accept its terms at "
            "https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m "
            "and run `hf auth login` (or set HF_TOKEN).",
            file=sys.stderr,
        )
        return 1

    total = sum(p.stat().st_size for p in HF_HOME.rglob("*") if p.is_file())
    print(f"done: {HF_HOME} ({total / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
