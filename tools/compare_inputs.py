"""Compare Idol output quality across input-image variants.

Runs inside the container against /data/objects.  Every metric is computed from the
written .ply only, so it is cheap and repeatable:

  solid  % of splats at opacity >= 0.5   -- real surface.  higher is better
  haze   % of splats below opacity 0.15  -- floating fog.  lower is better
  bleed  horizontal footprint of the haze divided by that of the solid body.
         > 1.05 means the fog spreads past the object (the "sitting on a cloud" look)
  base   % of the solid splats that sit in the lowest 20% of the object's height,
         and how wide that band is compared to the body.  Thin legs give a small
         share and a narrow band; a filled-in plinth gives a wide band; a cloud
         gives almost no solid splats there at all.
"""
import numpy as np
from plyfile import PlyData

# Job ids from one development experiment. Replace these with your own object ids.
# written .ply: object up is -Y, so larger Y == lower down
CASES = [
    ("3/4  alpha cutout  synth", "90ebeebca60c"),
    ("3/4  white bg      synth", "90a51ba12a4d"),
    ("3/4  grey bg       synth", "6c49c0f8319b"),
    ("3/4  lightgrey bg  synth", "46f9f0471618"),
    ("front  white bg    synth", "a8eaee4ab657"),
    ("front  grey bg     synth", "2a24fe8a8208"),
    ("3/4  white bg      real ", "015521e165b6"),
    ("3/4  alpha cutout  real ", "bc12ebfa7ae0"),
    ("front  white bg    real ", "23c2ceeca0c9"),
]


def load(job):
    v = PlyData.read(f"/data/objects/{job}.ply")["vertex"]
    xyz = np.stack([v[n] for n in ("x", "y", "z")], 1).astype(np.float32)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float32)))
    return xyz, op


def span(p):
    """robust horizontal footprint: 2nd..98th percentile on both horizontal axes"""
    if len(p) < 50:
        return 0.0
    a = np.percentile(p[:, [0, 2]], 98, axis=0) - np.percentile(p[:, [0, 2]], 2, axis=0)
    return float(np.hypot(*a))


print(f"{'input variant':26s} {'solid':>6s} {'haze':>6s} {'bleed':>6s} "
      f"{'base%':>6s} {'base/foot':>9s} {'W':>5s} {'H':>5s} {'D':>5s}")
print("-" * 92)
for label, job in CASES:
    try:
        xyz, op = load(job)
    except FileNotFoundError:
        print(f"{label} missing")
        continue
    solid, haze = op >= 0.5, op < 0.15
    s, h = xyz[solid], xyz[haze]
    y0, y1 = s[:, 1].min(), s[:, 1].max()
    foot = span(s)
    low = s[s[:, 1] > y1 - 0.2 * (y1 - y0)]          # lowest 20% of the body
    base_share = len(low) / max(len(s), 1)
    base_foot = span(low) / foot if foot else 0.0
    bleed = span(h) / foot if foot else 0.0
    ext = s.max(0) - s.min(0)
    print(f"{label} {solid.mean()*100:5.1f}% {haze.mean()*100:5.1f}% {bleed:6.2f} "
          f"{base_share*100:5.1f}% {base_foot:8.2f} "
          f"{ext[0]:5.2f} {ext[1]:5.2f} {ext[2]:5.2f}")
