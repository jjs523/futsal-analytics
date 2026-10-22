"""A filtered copy of a detection cache: class-agnostic NMS per source frame and / or a higher confidence floor.

python experiments/filter_cache.py --src experiments/cache_full26 --out experiments/cache_full26_nms07 --nms 0.7
YOLO26 is NMS-free (one-to-one head): on the 10/8 cache it leaves more overlapping boxes than YOLO11 + NMS, and the
conservative tracklet builder cuts at every ambiguous overlap. This checks whether putting NMS back recovers the
YOLO11 continuity. Boxes are kept in cache order, so a filtered cache is a valid cache for every script.
"""
import argparse
import json
import os
import shutil

import numpy as np


def nms(xyxy: np.ndarray, conf: np.ndarray, thr: float) -> np.ndarray:
    """Greedy class-agnostic NMS; returns a keep mask."""
    order = np.argsort(-conf, kind="stable")
    keep = np.zeros(len(conf), bool)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    while len(order):
        i = order[0]
        keep[i] = True
        rest = order[1:]
        x1 = np.maximum(xyxy[i, 0], xyxy[rest, 0]); y1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        x2 = np.minimum(xyxy[i, 2], xyxy[rest, 2]); y2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nms", type=float, default=None, help="IoU threshold (None: no NMS)")
    ap.add_argument("--min-conf", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    for cam in meta["videos"]:
        z = dict(np.load(os.path.join(a.src, f"{cam}.npz")))
        n = len(z["t"])
        keep = np.ones(n, bool)
        if a.min_conf is not None:
            keep &= z["conf"] >= a.min_conf
        if a.nms is not None:
            fid = z["fid"]
            order = np.argsort(fid, kind="stable")
            cuts = np.flatnonzero(np.diff(fid[order])) + 1
            for idx in np.split(order, cuts):
                idx = idx[keep[idx]]
                if len(idx) > 1:
                    keep[idx[~nms(z["xyxy"][idx].astype(float), z["conf"][idx], a.nms)]] = False
        out = {k: (v[keep] if (isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n) else v) for k, v in z.items()}
        np.savez_compressed(os.path.join(a.out, f"{cam}.npz"), **out)
        print(f"{cam}: kept {keep.sum()} of {n} boxes", flush=True)
    meta["filter"] = {"src": a.src, "nms": a.nms, "min_conf": a.min_conf}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
