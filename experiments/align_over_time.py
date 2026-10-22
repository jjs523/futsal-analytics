"""Does the cross-camera correction change during the match? Per 5-minute window: raw disagreement, residual after the
whole-match fit, residual after a fit on that window only, and the best extra time shift of cam2 (clock drift check).

python experiments/align_over_time.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.pipeline import align
from futsal.pipeline.boxes import cam_boxes_from_cache

CACHE = os.path.join(ROOT, "experiments", "cache_full")
meta = json.load(open(os.path.join(CACHE, "meta.json")))
court, rate, start = Court(), float(meta["rate"]), float(meta["start"])
cams = {}
for cam in meta["videos"]:
    cal = timeline_from_json(json.load(open(os.path.join(ROOT, "calib", cam, "calib_timeline.json"), encoding="utf-8")))
    cams[cam] = cam_boxes_from_cache(os.path.join(CACHE, f"{cam}.npz"), cam, cal, float(meta["offsets"][cam]), start, rate, court)
a, b = cams["cam1"], cams["cam2"]
glob = align.align_cameras(cams, court)
print("whole match:", {k: glob[k] for k in ("pairs", "median_before_m", "median_after_m")}, flush=True)
W = int(300 * rate)
n = 1 + max(max(c.by_k) for c in cams.values())
for lo in range(0, n, W):
    rng = (lo, min(lo + W, n))
    P, Q = align.matched_pairs(a, b, court, k_range=rng)          # raw positions (cam2, cam1)
    if len(P) < 200:
        print(f"{(start + lo / rate) / 60:5.1f} min: {len(P)} pairs, skipped"); continue
    raw = np.median(np.linalg.norm(P - Q, axis=1))
    Wg = {c: np.asarray(glob["coef"][c]) for c in glob["coef"]}
    F = align._features if hasattr(align, "_features") else None
    gP = P + 0.5 * (align._features(np.clip(P, [-3, -3], [43, 23]), court) @ Wg["cam2"])
    gQ = Q + 0.5 * (align._features(np.clip(Q, [-3, -3], [43, 23]), court) @ Wg["cam1"])
    glob_res = np.median(np.linalg.norm(gP - gQ, axis=1))
    loc = align.align_cameras({"cam1": a, "cam2": b}, court, k_range=rng)
    loc_res = loc["median_after_m"] if loc else float("nan")
    print(f"{(start + lo / rate) / 60:5.1f} min: pairs {len(P):5d}  raw {raw:.3f}  global-fit {glob_res:.3f}  window-fit {loc_res:.3f}", flush=True)
