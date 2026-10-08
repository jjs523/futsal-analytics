"""Crossing audit sheets: python experiments/make_event_audit.py <result name> [--max 30]

For same-team pairs of tracks that come within 1.2 m of each other, one sheet per encounter: row A and row B show each
track's player at -2, -1, +1, +2 s around the closest approach (crop from the camera where the player looks biggest).
A rater answers: did A and B keep their identities (OK), swap (SWAP), or can't tell (UNSURE)?
Writes experiments/audit/<name>/events/ev_XX.jpg + events.json.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, Data, load_tracks
from futsal.syncview import frames_at


def majority_team(t, data):
    labs = [data.cams[c].team[i] for p in t.values() for c, i in p.boxes if data.cams[c].team[i]]
    if not labs:
        return "?"
    y = sum(1 for x in labs if x == "Y")
    return "Y" if y >= len(labs) / 2 else "N"


def best_box(t, k, data, search=6):
    """Largest box of track t at frame k (or the nearest frame within `search`)."""
    for dk in [0] + [s * d for s in range(1, search + 1) for d in (-1, 1)]:
        p = t.get(k + dk)
        if p and p.boxes:
            return max(p.boxes, key=lambda cb: data.cams[cb[0]].h[cb[1]])
    return None


def crop(data, cb, h=150):
    cam, i = cb
    c = data.cams[cam]
    (_, f), = frames_at(data.videos[cam], [float(c.t[i])])
    x1, y1, x2, y2 = c.xyxy[i]
    pad = 0.15 * (y2 - y1)
    im = f[int(max(0, y1 - pad)):int(min(f.shape[0], y2 + pad)), int(max(0, x1 - pad)):int(min(f.shape[1], x2 + pad))]
    if not im.size:
        return np.zeros((h, 60, 3), np.uint8)
    return cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--dist", type=float, default=1.2)
    a = ap.parse_args()
    data = Data()
    tracks = load_tracks(os.path.join(ROOT, "experiments", "results", f"{a.name}.json"))
    tracks = [t for t in tracks if len(t) >= 5 * data.rate]
    teams = [majority_team(t, data) for t in tracks]
    n = data.n
    pos = []
    for t in tracks:
        p = np.full((n, 2), np.nan)
        for k, q in t.items():
            if 0 <= k < n:
                p[k] = q.xy
        pos.append(p)
    enc = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if teams[i] != teams[j] or teams[i] == "?":
                continue
            d = np.linalg.norm(pos[i] - pos[j], axis=1)
            close = np.where(d < a.dist)[0]
            if not len(close):
                continue
            runs = np.split(close, np.where(np.diff(close) > int(2 * data.rate))[0] + 1)
            for r in runs:
                k = int(r[np.argmin(d[r])])
                if k - 2 * data.rate < 0 or k + 2 * data.rate >= n:
                    continue
                enc.append((k, i, j))
    enc.sort()
    step = max(1, len(enc) // a.max) if enc else 1
    chosen = enc[::step][:a.max]
    out_dir = os.path.join(ROOT, "experiments", "audit", a.name, "events")
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for e, (k, i, j) in enumerate(chosen, 1):
        rows = []
        for lab, ti in (("A", i), ("B", j)):
            tiles = []
            for ds in (-2, -1, 1, 2):
                kk = int(k + ds * data.rate)
                cb = best_box(tracks[ti], kk, data)
                im = crop(data, cb) if cb else np.zeros((150, 60, 3), np.uint8)
                canvas = np.zeros((172, max(im.shape[1], 90), 3), np.uint8)
                canvas[22:, :im.shape[1]] = im
                cv2.putText(canvas, f"{lab} {ds:+d}s", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(canvas)
            rows.append(np.hstack(tiles))
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows]
        path = os.path.join(out_dir, f"ev_{e:02d}.jpg")
        cv2.imwrite(path, np.vstack([rows[0], np.full((6, W, 3), 255, np.uint8), rows[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        t_s = data.start + k / data.rate
        index.append({"event": e, "sheet": path, "t": f"{int(t_s // 60)}:{t_s % 60:04.1f}", "k": k, "team": teams[i]})
        print(f"  ev {e}: {index[-1]['t']} team {teams[i]}", flush=True)
    json.dump({"encounters_total": len(enc), "sampled": index}, open(os.path.join(out_dir, "events.json"), "w"), indent=1)
    print(f"{len(enc)} same-team encounters, {len(chosen)} sheets")
