"""Audit sheets for the longest tracks of a result: python experiments/make_audit.py <result name> [--top 12] [--n 16]

Writes experiments/audit/<name>/track_<rank>.jpg (crops of the player over time, from the camera where they look
biggest) and experiments/audit/<name>/index.json with each sheet's track length.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, Data, audit_sheet, load_tracks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    data = Data()
    tracks = sorted(load_tracks(os.path.join(ROOT, "experiments", "results", f"{a.name}.json")), key=len, reverse=True)
    out_dir = os.path.join(ROOT, "experiments", "audit", a.name)
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for r, t in enumerate(tracks[:a.top], 1):
        path = os.path.join(out_dir, f"track_{r:02d}.jpg")
        tiles = audit_sheet(t, data, path, n=a.n)
        index.append({"rank": r, "sheet": path, "frames": len(t), "seconds": round(len(t) / data.rate, 1), "tiles": tiles})
        print(f"  track {r}: {len(t) / data.rate:.0f}s, {tiles} crops -> {path}", flush=True)
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
