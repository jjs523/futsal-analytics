"""Tracking result video for an experiment result (same layout as futsal.trackview: two camera panels + 2D map + ID events).

python experiments/render_result.py final_d --start-s 600 --duration 180
  --start-s is the cam1 clock (the cache window is 540-900 s). Writes experiments/videos/<name>_<start>.mp4 (+ .csv events).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, Data, load_tracks
from futsal.pipeline.detect import Detection
from futsal.tracks import PlayerTrack, TrackSet
from futsal.trackview import find_events, render, _fmt
from blocks import team_vote

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--start-s", type=float, default=600.0)
    ap.add_argument("--duration", type=float, default=180.0)
    a = ap.parse_args()
    data = Data()
    k0 = int(round((a.start_s - data.start) * data.rate))
    n = int(round(a.duration * data.rate))
    tracks = sorted(load_tracks(os.path.join(ROOT, "experiments", "results", f"{a.name}.json")), key=len, reverse=True)
    players = []
    for t in tracks:
        xy = np.full((n, 2), np.nan)
        for k, p in t.items():
            if k0 <= k < k0 + n:
                xy[k - k0] = p.xy
        if np.isfinite(xy[:, 0]).any():
            team = {"Y": "Y", "N": "N"}.get(team_vote(t, data)[0], "?")
            players.append(PlayerTrack(len(players) + 1, team, xy))
    ts = TrackSet(40.0, 20.0, data.rate, players, start=a.start_s)
    dets = {}
    for cam, c in data.cams.items():
        m = (c.k >= k0) & (c.k < k0 + n) & (c.conf >= 0.3)
        dets[cam] = [Detection(int(c.fid[i]), float(c.t[i]), float(c.foot[i, 0]), float(c.foot[i, 1]), float(c.conf[i]),
                               None, float(c.h[i])) for i in np.flatnonzero(m)]
    cals = {cam: c.cal for cam, c in data.cams.items()}
    offsets = {cam: (data.offsets[cam], 0.0) for cam in data.cams}
    events = find_events(ts, data.court)
    out_dir = os.path.join(ROOT, "experiments", "videos")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{a.name}_{int(a.start_s)}")
    with open(stem + "_events.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time(cam1)", "ID", "kind", "x(m)", "y(m)"])
        for e in events:
            w.writerow([_fmt(a.start_s + e["t"]), e["id"], e["kind"], e["x"], e["y"]])
    frames = render(stem + ".mp4", data.videos, cals, offsets, ts, dets, data.court, a.start_s, events)
    print(f"{len(players)} IDs, {len(events)} mid-pitch events, {frames} frames -> {stem}.mp4")
