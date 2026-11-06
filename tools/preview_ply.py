"""Render a .ply to PNG without a GPU, to check what Idol actually produced.

An orthographic point projection straight out of the file: no viewer, no WebGL,
no assumptions about up-axis. Run it inside the service image, which has plyfile:

  MSYS_NO_PATHCONV=1 docker compose exec -T mcp-image-to-3d python /app/tools/preview_ply.py /data/objects/<id>.ply /tmp/prev.png
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from plyfile import PlyData


def load(path: str):
    ply = PlyData.read(path)["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    dc = np.stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], axis=1).astype(np.float32)
    op = ply["opacity"].astype(np.float32)
    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1)      # degree-0 SH -> colour
    alpha = 1.0 / (1.0 + np.exp(-op))                          # logit -> opacity
    return xyz, rgb, alpha


def project(xyz, rgb, alpha, axes, res=512, min_alpha=0.1):
    keep = alpha >= min_alpha
    p, c = xyz[keep], rgb[keep]
    if len(p) == 0:
        return np.ones((res, res, 3), np.uint8) * 24
    lo, hi = p.min(0), p.max(0)
    q = (p - lo) / np.maximum(hi - lo, 1e-6)
    u, v = q[:, axes[0]], 1.0 - q[:, axes[1]]
    px = np.clip((u * (res - 1)).astype(int), 0, res - 1)
    py = np.clip((v * (res - 1)).astype(int), 0, res - 1)
    depth = q[:, axes[2]]

    img = np.ones((res, res, 3), np.float32) * 0.09
    order = np.argsort(-depth)                                   # far to near
    for x, y, i in zip(px[order], py[order], np.arange(len(depth))[order]):
        img[y, x] = c[i]
    return (img * 255).astype(np.uint8)


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + "-preview.png"
    xyz, rgb, alpha = load(src)

    print(f"splats      {len(xyz):,}")
    print(f"bounds      min={np.round(xyz.min(0), 3).tolist()} max={np.round(xyz.max(0), 3).tolist()}")
    print(f"opacity     mean={alpha.mean():.3f}  above 0.1: {(alpha >= 0.1).mean() * 100:.1f}%")
    print(f"colour      mean={np.round(rgb.mean(0), 3).tolist()} std={np.round(rgb.std(0), 3).tolist()}")

    side = project(xyz, rgb, alpha, (0, 2, 1))   # looking along -Y (Idol writes up on -Y)
    top = project(xyz, rgb, alpha, (0, 1, 2))    # looking along -Z
    gap = np.ones((side.shape[0], 8, 3), np.uint8) * 200
    Image.fromarray(np.concatenate([side, gap, top], axis=1)).save(out)
    print(f"wrote       {out}  (left: side view, right: top view)")


if __name__ == "__main__":
    main()
