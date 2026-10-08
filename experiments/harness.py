"""Shared harness for tracking experiments on the cached real footage (experiments/build_cache.py).

A tracker is a function `track(data: Data) -> list[Track]`, where a Track is {frame k: TrackPoint}.
Frame k is the common 10 Hz grid on cam1's clock: k = 0 at the window start. A TrackPoint holds the pitch position
and the boxes it was built from as (camera, box index) pairs, so metrics and visual audits can go back to the pixels.

    python experiments/harness.py baseline          # run one registered tracker and print label-free metrics
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.homography import timeline_from_json, calibration_at

CACHE = os.path.join(ROOT, "experiments", "cache")
DETECTOR_JITTER_PX = 3.0


@dataclass
class TrackPoint:
    xy: np.ndarray                          # (2,) metres
    boxes: list[tuple[str, int]] = field(default_factory=list)   # (camera, index into Data.cams[camera] arrays)


Track = dict  # {k: TrackPoint}


class CamData:
    """All boxes of one camera, flat arrays indexed by box index."""

    def __init__(self, name: str, npz: dict, cal, offset: float, start: float, rate: float, court: Court):
        self.name = name
        self.fid = npz["fid"].astype(int)
        self.t = npz["t"].astype(float)
        self.xyxy = npz["xyxy"].astype(float)
        self.conf = npz["conf"].astype(float)
        self.reid = npz["reid"].astype(np.float32)
        self.color = npz["color"].astype(float) / 255.0
        self.frame_t = npz["frame_t"].astype(float)
        self.cal = cal
        foot = np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]], 1)
        self.foot = foot
        self.h = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.w = self.xyxy[:, 2] - self.xyxy[:, 0]
        c = calibration_at(cal, float(np.median(self.t))) if len(self.t) else cal
        self.xy = c.to_pitch(foot) if len(foot) else np.zeros((0, 2))
        self.sigma = DETECTOR_JITTER_PX * c.metres_per_pixel(foot) if len(foot) else np.zeros(0)
        # common grid on cam1's clock: k = round((t_cam + offset - start) * rate)
        # one grid frame per source frame: times that land exactly between two grid frames (60 fps at x.x5 s) can
        # round two consecutive source frames onto the same k; push the later one to the next free k
        k_of = {}
        last = None
        for f, tt in sorted({(int(f), float(tt)) for f, tt in zip(self.fid, self.t)}):
            kk = int(np.round((tt + offset - start) * rate))
            if last is not None and kk <= last:
                kk = last + 1
            k_of[f] = last = kk
        self.k = np.array([k_of[int(f)] for f in self.fid], int)
        self.in_court = court.contains(self.xy, 1.0) if len(self.xy) else np.zeros(0, bool)
        # team from the bib: one team wears yellow bibs (upper-body hue bins 1-2 = OpenCV hue 15-45), the other none.
        # 'Y' / 'N' when clear, '' when the box is small or the colour is in between.
        up = self.color[:, :15] if len(self.color) else np.zeros((0, 15))
        yel = (up[:, 1] + up[:, 2]) / np.maximum(up.sum(1), 1e-9)
        self.yellow = yel
        self.team = np.where(self.h < 30, "", np.where(yel >= 0.15, "Y", np.where(yel < 0.05, "N", "")))
        self.by_k: dict[int, np.ndarray] = {}
        for i in np.argsort(self.k, kind="stable"):
            self.by_k.setdefault(int(self.k[i]), []).append(int(i))
        self.by_k = {k: np.array(v) for k, v in self.by_k.items()}
        # sample index -> frame time, for re-reading pixels
        self.k_of_fid = {}

    def boxes_at(self, k: int, min_conf: float = 0.0, court_only: bool = True) -> np.ndarray:
        idx = self.by_k.get(k, np.zeros(0, int))
        if len(idx) == 0:
            return idx
        m = self.conf[idx] >= min_conf
        if court_only:
            m &= self.in_court[idx]
        return idx[m]


class Data:
    def __init__(self, cache: str = CACHE, align: bool = True):
        meta = json.load(open(os.path.join(cache, "meta.json"), encoding="utf-8"))
        self.meta = meta
        self.court = Court(40, 20)
        self.rate = float(meta["rate"])
        self.start = float(meta["start"])
        self.duration = float(meta["duration"])
        self.n = int(round(self.duration * self.rate))
        self.videos = meta["videos"]
        self.offsets = {c: float(o) for c, o in meta["offsets"].items()}
        self.cams: dict[str, CamData] = {}
        for cam in self.videos:
            cal = timeline_from_json(json.load(open(os.path.join(ROOT, meta["calib"][cam]), encoding="utf-8")))
            npz = dict(np.load(os.path.join(cache, f"{cam}.npz")))
            self.cams[cam] = CamData(cam, npz, cal, self.offsets[cam], self.start, self.rate, self.court)

        self.alignment = None
        if align:
            self.align_cameras()

    def align_cameras(self, min_conf: float = 0.5, max_d: float = 2.5, min_sim: float = 0.7) -> dict:
        """Cross-camera self-calibration from the players themselves: confident cam1/cam2 boxes that are mutual
        nearest on the pitch and look alike (ReID) are the same person, so their foot points should coincide.
        Fit a cubic correction field each way and move both cameras to the midpoint. Keeps the raw positions
        in CamData.xy_raw."""
        from scipy.optimize import linear_sum_assignment
        cams = list(self.cams.values())
        if len(cams) != 2:
            return {}
        a, b = cams
        P, Q = [], []
        for k in range(self.n):
            ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k, min_conf)
            if not len(ia) or not len(ib):
                continue
            D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
            S = a.reid[ia] @ b.reid[ib].T
            r, c = linear_sum_assignment(D - 0.5 * S)
            for i, j in zip(r, c):
                if D[i, j] < max_d and S[i, j] > min_sim and D[i].argmin() == j and D[:, j].argmin() == i:
                    P.append(b.xy[ib[j]]); Q.append(a.xy[ia[i]])
        P, Q = np.array(P), np.array(Q)
        L, Wd = self.court.length, self.court.width

        def feats(X):
            x, y = X[:, 0] / L, X[:, 1] / Wd
            return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], 1)

        def fit(src, dst):
            F = feats(src)
            W = np.linalg.lstsq(F, dst - src, rcond=None)[0]
            for _ in range(3):
                r = np.linalg.norm(src + F @ W - dst, axis=1)
                keep = r < np.percentile(r, 85)
                W = np.linalg.lstsq(F[keep], (dst - src)[keep], rcond=None)[0]
            return W
        Wba, Wab = fit(P, Q), fit(Q, P)            # cam2 -> cam1 frame, cam1 -> cam2 frame
        before = float(np.median(np.linalg.norm(P - Q, axis=1)))
        for cam, W in ((a, Wab), (b, Wba)):
            cam.xy_raw = cam.xy.copy()
            if len(cam.xy):
                cam.xy = cam.xy + 0.5 * (feats(cam.xy) @ W)
                cam.in_court = self.court.contains(cam.xy, 1.0)
        Pm = P + 0.5 * (feats(P) @ Wba)
        Qm = Q + 0.5 * (feats(Q) @ Wab)
        after = float(np.median(np.linalg.norm(Pm - Qm, axis=1)))
        self.alignment = {"pairs": len(P), "median_before_m": round(before, 3), "median_after_m": round(after, 3),
                          "W_cam1": Wab.tolist(), "W_cam2": Wba.tolist()}
        return self.alignment

    def cam_time(self, cam: str, k: int) -> float:
        """Camera clock time of grid frame k."""
        return self.start + k / self.rate - self.offsets[cam]


# ---------------------------------------------------------------------------------------------------------------
# Label-free metrics
# ---------------------------------------------------------------------------------------------------------------

def _seen(tr: Track) -> np.ndarray:
    return np.array(sorted(tr))


def metrics(tracks: list[Track], data: Data, edge_m: float = 2.5, settle_s: float = 2.0, team=None) -> dict:
    n, rate, court = data.n, data.rate, data.court
    minutes = data.duration / 60.0
    tracks = [t for t in tracks if t]
    lens = np.array([len(t) for t in tracks]) if tracks else np.zeros(0)
    # births / deaths in the middle of the pitch (not at the window edges, not near the lines)
    events = 0
    for t in tracks:
        ks = _seen(t)
        for kind, k in (("new", ks[0]), ("lost", ks[-1])):
            if (kind == "new" and k < settle_s * rate) or (kind == "lost" and k > n - 1 - settle_s * rate):
                continue
            x, y = t[k].xy
            if min(x, y, court.length - x, court.width - y) < edge_m:
                continue
            events += 1
    # teleports: speed > 10 m/s between points at most 3 frames apart
    tele, steps = 0, 0
    for t in tracks:
        ks = _seen(t)
        for a, b in zip(ks, ks[1:]):
            if b - a <= 3:
                steps += 1
                if np.linalg.norm(t[b].xy - t[a].xy) / ((b - a) / rate) > 10.0:
                    tele += 1
    # box usage: every box at most once per frame per camera
    used: dict[tuple[str, int], int] = {}
    for t in tracks:
        for p in t.values():
            for cb in p.boxes:
                used[cb] = used.get(cb, 0) + 1
    double = sum(1 for v in used.values() if v > 1)
    good_boxes = sum(int((c.conf >= 0.5)[c.in_court & (c.k >= 0) & (c.k < n)].sum()) for c in data.cams.values())
    good_used = sum(1 for (cam, i) in used if data.cams[cam].conf[i] >= 0.5 and data.cams[cam].in_court[i])
    # duplicates: two tracks within 0.8 m of each other for >= 2 s together
    dup_pairs = 0
    pos = []
    for t in tracks:
        a = np.full((n, 2), np.nan)
        for k, p in t.items():
            if 0 <= k < n:
                a[k] = p.xy
        pos.append(a)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            both = np.isfinite(pos[i][:, 0]) & np.isfinite(pos[j][:, 0])
            if both.sum() < 2 * rate:
                continue
            close = np.linalg.norm(pos[i][both] - pos[j][both], axis=1) < 0.8
            if close.sum() >= 2 * rate:
                dup_pairs += 1
    # appearance breaks: within a track, 2 s chunks of ReID; a consecutive-chunk cosine similarity below 0.55
    breaks = 0
    for t in tracks:
        chunks = {}
        for k, p in t.items():
            for cam, i in p.boxes:
                c = data.cams[cam]
                if c.h[i] >= 60 and c.conf[i] >= 0.5:
                    chunks.setdefault(k // int(2 * rate), []).append(c.reid[i])
        keys = sorted(chunks)
        means = [np.mean(chunks[q], axis=0) for q in keys]
        means = [m / max(np.linalg.norm(m), 1e-9) for m in means]
        for q in range(1, len(means)):
            if keys[q] - keys[q - 1] <= 2 and float(means[q] @ means[q - 1]) < 0.55:
                breaks += 1
    # team impurity: share of a track's clearly-labelled boxes whose bib colour disagrees with the track's majority
    wrong = total = 0
    for t in tracks:
        labs = [data.cams[cam].team[i] for p in t.values() for cam, i in p.boxes if data.cams[cam].team[i]]
        if labs:
            y = sum(1 for x in labs if x == "Y")
            wrong += min(y, len(labs) - y)
            total += len(labs)
    concurrent = np.array([sum(1 for t in tracks if k in t) for k in range(n)])
    return {
        "ids": len(tracks),
        "ids_over_half": int((lens > 0.5 * n).sum()),
        "ids_over_20s": int((lens >= 20 * rate).sum()),
        "frames_in_top10": round(float(np.sort(lens)[::-1][:10].sum() / (10 * n)), 3) if len(lens) else 0.0,
        "mean_len_s": round(float(lens.mean() / rate), 1) if len(lens) else 0.0,
        "events_per_min": round(events / minutes, 2),
        "teleports_per_min": round(tele / minutes, 2),
        "appearance_breaks_per_min": round(breaks / minutes, 2),
        "team_impurity": round(wrong / max(total, 1), 4),
        "dup_pairs": dup_pairs,
        "double_used_boxes": double,
        "box_coverage": round(good_used / max(good_boxes, 1), 3),
        "concurrent_mean": round(float(concurrent.mean()), 2),
    }


# ---------------------------------------------------------------------------------------------------------------
# Visual audit sheets: crops of one track at evenly spaced times, from the camera where the player looks biggest
# ---------------------------------------------------------------------------------------------------------------

def audit_sheet(track: Track, data: Data, out_jpg: str, n: int = 16, crop_h: int = 160) -> int:
    import cv2
    from futsal.syncview import frames_at
    pts = []
    for k in sorted(track):
        best = None
        for cam, i in track[k].boxes:
            c = data.cams[cam]
            if best is None or c.h[i] > data.cams[best[0]].h[best[1]]:
                best = (cam, i)
        if best is not None:
            pts.append((k, best))
    if not pts:
        return 0
    sel = [pts[int(round(q))] for q in np.linspace(0, len(pts) - 1, min(n, len(pts)))]
    tiles = []
    for cam in data.cams:
        want = [(k, i) for k, (c, i) in sel if c == cam]
        if not want:
            continue
        c = data.cams[cam]
        # one short seek per crop: frames_at reads forward from the earliest to the latest time, so a single call
        # over crops spread across minutes would decode the whole window
        frames = [frames_at(data.videos[cam], [float(c.t[i])])[0] for _, i in want]
        for (k, i), (_, f) in zip(want, frames):
            x1, y1, x2, y2 = c.xyxy[i]
            pad = 0.15 * (y2 - y1)
            crop = f[int(max(0, y1 - pad)):int(min(f.shape[0], y2 + pad)), int(max(0, x1 - pad)):int(min(f.shape[1], x2 + pad))]
            if not crop.size:
                continue
            s = crop_h / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), crop_h))
            canvas = np.zeros((crop_h + 22, max(crop.shape[1], 70), 3), np.uint8)
            canvas[22:, :crop.shape[1]] = crop
            cv2.putText(canvas, f"{(data.start + k / data.rate) // 60:.0f}:{(data.start + k / data.rate) % 60:04.1f}{cam[-1]}",
                        (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append((k, canvas))
    tiles = [t for _, t in sorted(tiles, key=lambda kt: kt[0])]
    rows, row, w = [], [], 0
    for t in tiles:
        if w + t.shape[1] > 1400 and row:
            rows.append(row); row, w = [], 0
        row.append(t); w += t.shape[1]
    if row:
        rows.append(row)
    W = max(sum(t.shape[1] for t in r) for r in rows)
    img = np.vstack([np.hstack(r + [np.zeros((r[0].shape[0], W - sum(t.shape[1] for t in r), 3), np.uint8)]) for r in rows])
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    cv2.imwrite(out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(tiles)


def save_tracks(tracks: list[Track], path: str) -> None:
    out = [{str(k): [round(float(p.xy[0]), 3), round(float(p.xy[1]), 3), [[c, int(i)] for c, i in p.boxes]]
            for k, p in sorted(t.items())} for t in tracks if t]
    json.dump(out, open(path, "w"), separators=(",", ":"))


def load_tracks(path: str) -> list[Track]:
    raw = json.load(open(path))
    return [{int(k): TrackPoint(np.array(v[:2], float), [(c, int(i)) for c, i in v[2]]) for k, v in t.items()} for t in raw]


TRACKERS = {}


def register(name):
    def deco(fn):
        TRACKERS[name] = fn
        return fn
    return deco


if __name__ == "__main__":
    import importlib
    import time
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    name = sys.argv[1]
    mod = sys.argv[2] if len(sys.argv) > 2 else "trackers"
    importlib.import_module(mod)
    import harness as _h               # trackers register into the importable module, not __main__
    data = Data(align=os.environ.get("FUTSAL_NOALIGN") is None)
    if data.alignment:
        print(f"camera alignment: {data.alignment['pairs']} pairs, median {data.alignment['median_before_m']} -> "
              f"{data.alignment['median_after_m']} m", file=sys.stderr)
    t0 = time.time()
    tracks = _h.TRACKERS[name](data)
    dt = time.time() - t0
    os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)
    save_tracks(tracks, os.path.join(ROOT, "experiments", "results", f"{name}.json"))
    m = metrics(tracks, data) | {"runtime_s": round(dt, 1)}
    print(json.dumps({name: m}, ensure_ascii=False))
