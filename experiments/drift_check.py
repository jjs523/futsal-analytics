"""Clock drift check from the players: per 5-minute window, shift cam2 by d grid frames (0.1 s) and find the d at which
the same players (ReID-alike, mutual nearest) coincide best on the pitch. A trend of the best shift over the match = drift.

python experiments/drift_check.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.pipeline.boxes import cam_boxes_from_cache

CACHE = os.path.join(ROOT, "experiments", "cache_full")
meta = json.load(open(os.path.join(CACHE, "meta.json")))
court, rate, start = Court(), float(meta["rate"]), float(meta["start"])
cams = {}
for cam in meta["videos"]:
    cal = timeline_from_json(json.load(open(os.path.join(ROOT, "calib", cam, "calib_timeline.json"), encoding="utf-8")))
    cams[cam] = cam_boxes_from_cache(os.path.join(CACHE, f"{cam}.npz"), cam, cal, float(meta["offsets"][cam]), start, rate, court)
a, b = cams["cam1"], cams["cam2"]
n = 1 + max(max(c.by_k) for c in cams.values())
W, SHIFTS = int(300 * rate), range(-12, 13)


def residual(lo, hi, d, step=3):
    ds = []
    for k in range(lo, hi, step):
        ia, ib = a.boxes_at(k, 0.5), b.boxes_at(k + d, 0.5)
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
        S = a.reid[ia] @ b.reid[ib].T
        r, c = linear_sum_assignment(D - 0.5 * S)
        for i, j in zip(r, c):
            if D[i, j] < 2.5 and S[i, j] > 0.7 and D[i].argmin() == j and D[:, j].argmin() == i:
                ds.append(D[i, j])
    return (float(np.mean(ds)) if len(ds) > 100 else np.nan), len(ds)


rows = []
for lo in range(0, n, W):
    hi = min(lo + W, n)
    res = [(d, *residual(lo, hi, d)) for d in SHIFTS]
    ok = [(d, r, m) for d, r, m in res if np.isfinite(r)]
    if not ok:
        continue
    d_best, r_best, m = min(ok, key=lambda x: x[1])
    r0 = next(r for d, r, _ in res if d == 0)
    # sub-frame best shift: parabola through the best and its neighbours
    rs = {d: r for d, r, _ in res}
    frac = 0.0
    if d_best - 1 in rs and d_best + 1 in rs and np.isfinite(rs[d_best - 1]) and np.isfinite(rs[d_best + 1]):
        y0, y1, y2 = rs[d_best - 1], rs[d_best], rs[d_best + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    best_s = (d_best + frac) / rate
    rows.append(((lo + hi) / 2 / rate, best_s))
    print(f"{(start + lo / rate) / 60:5.1f} min: best cam2 shift {best_s:+.3f} s (mean dist {r_best:.3f} m vs {r0:.3f} m at 0, {m} pairs)", flush=True)
t, s = np.array(rows).T
k, c0 = np.polyfit(t, s, 1)
print(f"fit: shift(t) = {c0:+.3f} s {k * 1e6:+.0f} ppm x t  ({k * 3000:+.3f} s over 50 min)")
