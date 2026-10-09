"""Why do two identities sit on one player in the 60 s real clip? List each duplicate identity's boxes per camera with
their team labels, and save crops of both for a visual check.  python experiments/debug_dup.py"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.pipeline import detect
from futsal.pipeline.boxes import build_cam_boxes
from futsal.pipeline.identity import team_vote
from futsal.pipeline.run import guess_frame_wh, track_cams
from futsal.syncview import frames_at

out = os.path.join(ROOT, "trackview_v2_test")
V = {"cam1": r"C:\Users\user\dev\futsal-videos\20261008_170919.mp4", "cam2": r"C:\Users\user\dev\futsal-videos\20261008_171122.mp4"}
OFF = {"cam1": 0.0, "cam2": 124.268}
start, rate, court = 600.0, 10.0, Court()
cams = {}
for c in V:
    dets = detect.load(os.path.join(out, f"det_{c}.json"))[0]
    cal = timeline_from_json(json.load(open(os.path.join(ROOT, "calib", c, "calib.json"), encoding="utf-8")))
    cams[c] = build_cam_boxes(c, dets, cal, OFF[c] - start, 0.0, 0.0, rate, court, guess_frame_wh(dets))
info = {}
ids, slot_teams = track_cams(cams, court, rate, info=info)
print("teams:", {k: v for k, v in info["teams"].items() if k in ("mode", "pairs", "pair_agreement", "counts")})
pos = []
for t in ids:
    a = np.full((700, 2), np.nan)
    for k, p in t.items():
        if 0 <= k < 700:
            a[k] = p.xy
    pos.append(a)
pairs = []
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        both = np.isfinite(pos[i][:, 0]) & np.isfinite(pos[j][:, 0])
        close = both & (np.linalg.norm(np.nan_to_num(pos[i] - pos[j]), axis=1) < 0.6)
        if close.sum() >= 30:
            pairs.append((i, j, int(close.sum())))
print("duplicate pairs (index, index, frames):", pairs)
for i, j, _ in pairs:
    tiles = []
    for q in (i, j):
        t = ids[q]
        by_cam = {}
        for k, p in t.items():
            for cam, b in p.boxes:
                by_cam.setdefault(cam, []).append((k, b))
        desc = {cam: (len(v), {lab: int(sum(cams[cam].team[b] == lab for _, b in v)) for lab in ("A", "B", "")})
                for cam, v in by_cam.items()}
        print(f"  identity {q}: team_vote {team_vote(t, cams)}, slot {slot_teams[q] if slot_teams else '?'}, boxes per camera {desc}")
        row = []
        for cam, v in by_cam.items():
            sel = v[:: max(1, len(v) // 4)][:4]
            for k, b in sel:
                c = cams[cam]
                (_, f), = frames_at(V[cam], [float(c.t[b])])
                x1, y1, x2, y2 = c.xyxy[b]
                crop = f[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * 160 / max(crop.shape[0], 1))), 160))
                cv2.putText(crop, f"{q}{cam[-1]}{c.team[b] or '-'}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                row.append(crop)
        tiles.append(np.hstack(row) if row else np.zeros((160, 60, 3), np.uint8))
    W = max(r.shape[1] for r in tiles)
    img = np.vstack([np.hstack([r, np.zeros((160, W - r.shape[1], 3), np.uint8)]) for r in tiles])
    cv2.imwrite(os.path.join(out, f"dup_{i}_{j}.jpg"), img)
    print("  ->", os.path.join(out, f"dup_{i}_{j}.jpg"))

# colour features of the duplicate's boxes per camera vs the camera's cluster centres
from futsal.pipeline.boxes import UPPER
for q in sorted({i for p in pairs for i in p[:2]}):
    for cam in cams:
        bs = [b for p in ids[q].values() for c_, b in p.boxes if c_ == cam]
        if not bs:
            continue
        up = cams[cam].color[bs, :UPPER]
        up = up / np.maximum(up.sum(1, keepdims=True), 1e-9)
        print(f"identity {q} {cam}: n={len(bs)} mean upper hist {np.round(up.mean(0), 2).tolist()}")
for cam in cams:
    print(cam, "centres", {t: np.round(v, 2).tolist() for t, v in info["teams"]["centres"][cam].items()})
