"""Team labelling check: joint vs per-camera clustering against the yellow-share rule (cache), and duplicate identities
on the 60 s real clip run by trackview (trackview_v2_test).  python experiments/check_teams.py"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.pipeline import detect
from futsal.pipeline.boxes import assign_teams, cams_from_cache
from futsal.pipeline.run import build_tracks
from harness import Data

d = Data(align=False)
for per in (False, True):
    cams, _ = cams_from_cache(os.path.join(ROOT, "experiments", "cache"), root=ROOT)
    info = assign_teams(cams, per_camera=per)
    row = []
    for c in cams:
        a, y = cams[c].team, d.cams[c].team
        both = (a != "") & (y != "")
        agree = max(np.mean((a[both] == "A") == (y[both] == "N")), np.mean((a[both] == "A") == (y[both] == "Y")))
        row.append(f"{c}: labelled {both.sum()}, agree {agree:.4f}")
    print(f"per_camera={per} mode={info.get('mode')} pairs={info.get('pairs')} pair_agreement={info.get('pair_agreement')} | " + " | ".join(row))

out = os.path.join(ROOT, "trackview_v2_test")
if os.path.isdir(out):
    dets = {c: detect.load(os.path.join(out, f"det_{c}.json"))[0] for c in ("cam1", "cam2")}
    cals = {c: timeline_from_json(json.load(open(os.path.join(ROOT, "calib", c, "calib.json"), encoding="utf-8"))) for c in dets}
    start = 600.0
    sync = {"cam1": {"offset": 0.0 - start, "drift": 0.0}, "cam2": {"offset": 124.268 - start, "drift": 0.0}}
    info = {}
    ts = build_tracks(Court(), dets, cals, sync, fps_out=10.0, ids="v2", info=info)
    n = ts.n_frames
    dup = []
    for i in range(len(ts.players)):
        for j in range(i + 1, len(ts.players)):
            a, b = ts.players[i].xy, ts.players[j].xy
            both = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
            close = both & (np.linalg.norm(np.nan_to_num(a - b), axis=1) < 0.6)
            if close.sum() >= 30:
                dup.append((ts.players[i].id, ts.players[j].id, int(close.sum())))
    print(f"real clip: ids {len(ts.players)} teams {[p.team for p in ts.players]} teams-info mode {info['teams'].get('mode')} "
          f"pairs {info['teams'].get('pairs')}; duplicate pairs (>=3 s within 0.6 m): {dup}")
