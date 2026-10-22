"""Whole-match tracking from the full cache (experiments/cache_full, 10/8 test, ~53 min of overlap) with the src v2 tracker.

python experiments/full_match.py [--render] [--cache experiments/cache_full26] [--out experiments/full_match] [--linker closed|gta]
Writes <out>/tracks.json (TrackSet, web/viewer format), events.csv, summary.json (per-player distance,
speed, time seen) and, with --render, trackview.mp4 (two camera panels + 2D map). Camera knocks are handled with
calibration timelines (experiments/shift_sign.py decided the direction): cam1 was knocked at 2:40 (the tapped calibration
is after it), cam2 at 46:13 (the tapped calibration is before it).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.pipeline.align import corrected_sync
from futsal.pipeline.boxes import cam_boxes_from_cache
from futsal.pipeline.detect import Detection
from futsal.pipeline.run import tracks_to_trackset, track_cams
from futsal.trackview import _fmt, find_events, render

CACHE = os.path.join(ROOT, "experiments", "cache_full26")     # YOLO26x (cache_full: the older YOLO11x run)
OUT = os.path.join(ROOT, "experiments", "full_match")
KNOCKS = {"cam1": [(0.0, (41.0, 19.0)), (159.9, (0.0, 0.0))], "cam2": [(0.0, (0.0, 0.0)), (2773.0, (13.8, 1.3))]}


def timeline(cam: str) -> list[dict]:
    base = timeline_from_json(json.load(open(os.path.join(ROOT, "calib", cam, "calib.json"), encoding="utf-8")))
    return [{"from": t, **base.shifted(dx, dy).to_json()} for t, (dx, dy) in KNOCKS[cam]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--linker", default="closed", choices=("closed", "gta"))
    a = ap.parse_args()
    CACHE, OUT = a.cache, a.out
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    court, rate, start = Court(), float(meta["rate"]), float(meta["start"])
    t0 = time.time()
    cams, cals = {}, {}
    for cam in meta["videos"]:
        tl = timeline(cam)
        json.dump(tl, open(os.path.join(ROOT, "calib", cam, "calib_timeline.json"), "w"), indent=1)
        cals[cam] = timeline_from_json(tl)
        cams[cam] = cam_boxes_from_cache(os.path.join(CACHE, f"{cam}.npz"), cam, cals[cam], float(meta["offsets"][cam]),
                                         start, rate, court)
        print(f"{cam}: {len(cams[cam])} boxes ({time.time() - t0:.0f}s)", flush=True)
    info = {}
    ids, slot_teams = track_cams(cams, court, rate, info=info, linker=a.linker)      # also corrects cam2's clock
    clock = info["clock"]
    if clock is not None:
        print(f"clock: {clock['camera']} offset {clock['offset_s']:+.3f} s, drift {clock['drift'] * 1e6:+.0f} ppm "
              f"(applied {clock['applied']}); shifts {clock['shift_s']}", flush=True)
    st = info["solver"]
    print(f"tracked: {info['tracklets']} tracklets -> {len(ids)} identities; alignment {info['alignment'] and {k: info['alignment'][k] for k in ('pairs', 'median_before_m', 'median_after_m')}}; "
          f"teams {info['teams'].get('mode')}; windows {getattr(st, 'windows', '?')} ({time.time() - t0:.0f}s)", flush=True)
    n = 1 + max(max(c.by_k) for c in cams.values())
    ts = tracks_to_trackset(ids, cams, court, rate, n, slot_teams or None)
    ts.start = start
    ts.compute_stats()
    ts.dump(os.path.join(OUT, "tracks.json"))
    events = find_events(ts, court)
    with open(os.path.join(OUT, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time(cam1)", "ID", "kind", "x(m)", "y(m)"])
        for e in events:
            w.writerow([_fmt(start + e["t"]), e["id"], e["kind"], e["x"], e["y"]])
    seen = {p.id: float(np.isfinite(p.xy[:, 0]).mean()) for p in ts.players}
    summary = {"detector": meta.get("model"), "linker": a.linker, "clock": clock, "window_cam1_s": [start, start + n / rate], "identities": len(ts.players), "events": len(events),
               "events_per_min": round(len(events) / (n / rate / 60), 2), "tracklets": info["tracklets"],
               "alignment": info["alignment"] and {k: info["alignment"][k] for k in ("pairs", "median_before_m", "median_after_m")},
               "teams": {k: v for k, v in info["teams"].items() if k in ("mode", "pairs", "pair_agreement", "counts")},
               "windowed": {k: getattr(st, k) for k in ("windows", "new_ids", "ended", "agreement") if hasattr(st, k)},
               "players": [{"id": p.id, "team": p.team, "seen_share": round(seen[p.id], 3), **p.stats} for p in ts.players]}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    long_ids = sum(1 for v in seen.values() if v > 0.5)
    print(f"{len(ts.players)} identities ({long_ids} seen > half the match), {len(events)} mid-pitch events "
          f"({summary['events_per_min']}/min) -> {OUT} ({time.time() - t0:.0f}s)", flush=True)
    if a.render:
        dets = {}
        for cam, c in cams.items():
            m = c.conf >= 0.3
            dets[cam] = [Detection(int(c.fid[i]), float(c.t[i]), float(c.foot[i, 0]), float(c.foot[i, 1]), float(c.conf[i]), None,
                                   float(c.h[i]), box=tuple(float(v) for v in c.xyxy[i])) for i in np.flatnonzero(m)]
        offsets = {cam: (float(meta["offsets"][cam]), 0.0) for cam in cams}
        if clock is not None and clock["applied"]:
            offsets[clock["camera"]] = corrected_sync(*offsets[clock["camera"]], clock, start)
        frames = render(os.path.join(OUT, "trackview.mp4"), meta["videos"], cals, offsets, ts, dets, court, start, events)
        print(f"rendered {frames} frames ({time.time() - t0:.0f}s)", flush=True)
