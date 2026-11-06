"""Score the parameter sweep in data/sweep.json.

Every run in the sweep has the same splat count, which is what makes the opacity
metric meaningful here: it compared badly across presets because more splats always
means more faint ones.  Held fixed, higher solid / lower haze is a real difference.
"""
import json
import numpy as np
from plyfile import PlyData

runs = json.load(open("/data/sweep.json"))
print(f"{'run':26s} {'steps':>5s} {'cfg':>4s} {'rt':>4s} {'time':>6s} {'solid':>7s} {'haze':>6s}  extent")
for r in sorted(runs, key=lambda r: (r["name"], str(r["value"]))):
    p = r["params"]
    v = PlyData.read(f"/data/objects/{r['job']}.ply")["vertex"]
    xyz = np.stack([v[n] for n in ("x", "y", "z")], 1).astype(np.float32)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float32)))
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack(
        [v["f_dc_" + str(i)] for i in range(3)], 1).astype(np.float32), 0, 1)
    s = xyz[op >= 0.5]
    ext = s.max(0) - s.min(0)
    print(f"{r['name'] + '=' + str(r['value']):26s} {p['steps']:5d} {p['cfg_strength']:4.1f} "
          f"{p['rescale_t']:4.1f} {str(r.get('seconds')) + 's':>6s} {(op >= 0.5).mean()*100:6.1f}% "
          f"{(op < 0.15).mean()*100:5.1f}%  {ext[0]:.2f} {ext[1]:.2f} {ext[2]:.2f}  "
          f"rgb {np.round(rgb[op >= 0.5].mean(0), 2).tolist()}")
