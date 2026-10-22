"""Which way to apply a measured camera knock? Compare the two cameras' disagreement on the same players before and
after the knock with no correction, +shift and -shift applied to the knocked camera's calibration after the knock.

python experiments/shift_sign.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import Calibration, timeline_from_json
from futsal.pipeline.align import matched_pairs
from futsal.pipeline.boxes import cam_boxes_from_cache

CACHE = os.path.join(ROOT, "experiments", "cache_full")
meta = json.load(open(os.path.join(CACHE, "meta.json")))
court = Court()
base = {c: timeline_from_json(json.load(open(os.path.join(ROOT, meta["calib"][c]), encoding="utf-8"))) for c in meta["videos"]}

# knocks measured with phase correlation (frame 1 s before vs 1.5 s after), in each camera's own clock
KNOCKS = [("cam1", 159.9, (-41.0, -19.0)), ("cam2", 2773.0, (13.8, 1.3))]


def cams_with(cam_knocked, t_knock, shift, sign, before_cal_changes):
    cams = {}
    for c in meta["videos"]:
        cal = base[c]
        if c == cam_knocked and sign != 0:
            dx, dy = shift[0] * sign, shift[1] * sign
            # the tapped calibration (10:00) is valid on one side of the knock; the other side gets the shifted copy
            cal = [(0.0, cal.shifted(dx, dy)), (t_knock, cal)] if before_cal_changes else [(0.0, cal), (t_knock, cal.shifted(dx, dy))]
        cams[c] = cam_boxes_from_cache(os.path.join(CACHE, f"{c}.npz"), c, cal, float(meta["offsets"][c]), float(meta["start"]),
                                       float(meta["rate"]), court)
    return cams


for cam, t, shift in KNOCKS:
    before_side = cam == "cam1"          # cam1 was tapped at 10:00, after its knock at 2:40; cam2 too, before its knock at 46:13
    off = float(meta["offsets"][cam])
    k_knock = int(round((t + off - float(meta["start"])) * float(meta["rate"])))
    lo, hi = (max(0, k_knock - 600), k_knock), (k_knock + 20, k_knock + 620)
    for sign in (0, 1, -1):
        cams = cams_with(cam, t, shift, sign, before_side)
        a, b = cams["cam1"], cams["cam2"]
        res = []
        for rng in (lo, hi):
            P, Q = matched_pairs(a, b, court, k_range=rng) if "k_range" in matched_pairs.__code__.co_varnames else matched_pairs(a, b, court)
            d = np.linalg.norm(P - Q, axis=1) if len(P) else np.array([np.nan])
            res.append(f"k {rng[0]}-{rng[1]}: n={len(P)} median {np.nanmedian(d):.3f} m")
        print(f"{cam} knock at {t}s, sign {sign:+d}: " + " | ".join(res), flush=True)
