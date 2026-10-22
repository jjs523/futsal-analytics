"""Cut a window out of the full-match cache: python experiments/slice_cache.py --start 1500 --duration 360 --out experiments/cache_w2
(--start on cam1's clock, like build_cache.py)."""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "experiments", "cache_full"))
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    os.makedirs(a.out, exist_ok=True)
    for cam, off in meta["offsets"].items():
        z = dict(np.load(os.path.join(a.src, f"{cam}.npz")))
        lo, hi = a.start - float(off), a.start + a.duration - float(off)
        m = (z["t"] >= lo) & (z["t"] <= hi)
        ft = z["frame_t"][(z["frame_t"] >= lo) & (z["frame_t"] <= hi)]
        np.savez_compressed(os.path.join(a.out, f"{cam}.npz"), frame_t=ft,
                            **{k: z[k][m] for k in ("fid", "t", "xyxy", "conf", "reid", "color")})
        print(f"{cam}: {int(m.sum())} boxes, {len(ft)} frames")
    json.dump(meta | {"start": a.start, "duration": a.duration}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
