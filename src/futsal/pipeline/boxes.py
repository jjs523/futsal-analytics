"""Per-camera player boxes on a common frame grid: the input of the box-level tracker.

The older pipeline (run.to_observations) keeps only a foot point per detection and fuses cameras frame by frame.
The box-level tracker needs more: the whole box (overlaps, crops, image-space motion), the ReID embedding and the
colour histograms, and a way to go back from any track point to the pixels it was built from. CamBoxes keeps all
boxes of one camera as flat arrays indexed by box index; a TrackPoint refers to boxes as (camera, box index).

Frame grid: k = round((t + offset + drift * t - start) * rate) on the reference clock, so k = 0 is the window start.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..court import Court
from ..homography import Calibration, CalibrationTimeline, calibration_at, timeline_from_json
from .detect import FEAT_LEN, HUE_BINS, Detection

DETECTOR_JITTER_PX = 3.0      # foot-point noise of the detector (same as run.DETECTOR_JITTER_PX)
COURT_MARGIN_M = 1.0          # in_court keeps players just outside the lines (throw-ins, keeper behind the line)
UPPER = HUE_BINS + 3          # upper-body half of the appearance histogram (pipeline.detect.appearance)


@dataclass
class TrackPoint:
    xy: np.ndarray                                                  # (2,) metres
    boxes: list[tuple[str, int]] = field(default_factory=list)      # (camera, box index into CamBoxes arrays)


Track = dict[int, TrackPoint]     # {grid frame k: TrackPoint}


@dataclass
class CamBoxes:
    """All boxes of one camera, flat arrays indexed by box index.

    `reid` / `color` rows of boxes that have no embedding / histogram are zero: a zero ReID row has cosine
    similarity 0 with everything (never a confident match) and a zero histogram is skipped by assign_teams."""
    name: str
    fid: np.ndarray               # (n,) int   source frame index (as stored by the detector)
    t: np.ndarray                 # (n,) camera clock, s
    k: np.ndarray                 # (n,) int   grid frame on the reference clock
    xyxy: np.ndarray              # (n, 4) pixels (calibrated resolution)
    conf: np.ndarray              # (n,)
    reid: np.ndarray | None       # (n, d) float32, L2-normalised (to the float16 precision it is stored with)
    color: np.ndarray | None      # (n, 30) in [0, 1], upper + lower body histograms
    foot: np.ndarray              # (n, 2) pixels, bottom centre of the box
    h: np.ndarray                 # (n,) box height, px
    w: np.ndarray                 # (n,) box width, px
    xy: np.ndarray                # (n, 2) pitch metres (camera-aligned once align.align_cameras ran)
    xy_raw: np.ndarray            # (n, 2) pitch metres straight from the homography
    sigma: np.ndarray             # (n,) foot-point uncertainty, metres
    in_court: np.ndarray          # (n,) bool, inside the pitch + COURT_MARGIN_M
    team: np.ndarray              # (n,) str 'A' / 'B' / '' (see assign_teams)
    cal: Calibration | CalibrationTimeline
    frame_wh: tuple[int, int] = (1920, 1080)
    by_k: dict[int, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def boxes_at(self, k: int, min_conf: float = 0.0, court_only: bool = True) -> np.ndarray:
        """Box indices of grid frame k (ascending)."""
        idx = self.by_k.get(k, np.zeros(0, int))
        if len(idx) == 0:
            return idx
        m = self.conf[idx] >= min_conf
        if court_only:
            m &= self.in_court[idx]
        return idx[m]


def grid_frames(fid: np.ndarray, t: np.ndarray, offset_s: float, drift: float, start_s: float, rate: float) -> np.ndarray:
    """Grid frame k of every box: one grid frame per source frame.

    Source times that land exactly between two grid frames (60 fps at x.x5 s) can round two consecutive source
    frames onto the same k; the later one is pushed to the next free k, so a k never mixes two source frames."""
    k_of: dict[tuple[int, float], int] = {}
    last = None
    for f, tt in sorted({(int(f), float(tt)) for f, tt in zip(fid, t)}, key=lambda ft: (ft[1], ft[0])):
        kk = int(np.round((tt + offset_s + drift * tt - start_s) * rate))
        if last is not None and kk <= last:
            kk = last + 1
        k_of[(f, tt)] = last = kk
    return np.array([k_of[(int(f), float(tt))] for f, tt in zip(fid, t)], int).reshape(-1)


def _pitch(cal: Calibration | CalibrationTimeline, t: np.ndarray, foot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Foot pixels -> pitch metres and their sigma, with the calibration in force at each box's time
    (a knocked tripod gives a timeline)."""
    xy, sig = np.zeros((len(foot), 2)), np.zeros(len(foot))
    if not len(foot):
        return xy, sig
    if isinstance(cal, Calibration):
        groups = [(cal, np.ones(len(foot), bool))]
    else:
        which = np.array([id(calibration_at(cal, float(tt))) for tt in t])
        groups = [(c, which == id(c)) for _, c in cal]
    for c, m in groups:
        if m.any():
            xy[m] = c.to_pitch(foot[m])
            sig[m] = DETECTOR_JITTER_PX * c.metres_per_pixel(foot[m])
    return xy, sig


def _index_by_k(k: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i in np.argsort(k, kind="stable"):
        out.setdefault(int(k[i]), []).append(int(i))
    return {kk: np.array(v, int) for kk, v in out.items()}


def _cam_boxes(name: str, fid, t, xyxy, conf, reid, color, cal, offset_s: float, drift: float, start_s: float,
               rate: float, court: Court, frame_wh=(1920, 1080), foot=None) -> CamBoxes:
    fid = np.asarray(fid).astype(int).reshape(-1)
    t = np.asarray(t, float).reshape(-1)
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    conf = np.asarray(conf, float).reshape(-1)
    if foot is None:
        foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)
    xy, sigma = _pitch(cal, t, foot)
    k = grid_frames(fid, t, offset_s, drift, start_s, rate)
    return CamBoxes(name=name, fid=fid, t=t, k=k, xyxy=xyxy, conf=conf, reid=reid, color=color, foot=foot,
                    h=xyxy[:, 3] - xyxy[:, 1], w=xyxy[:, 2] - xyxy[:, 0], xy=xy, xy_raw=xy.copy(), sigma=sigma,
                    in_court=court.contains(xy, COURT_MARGIN_M) if len(xy) else np.zeros(0, bool),
                    team=np.full(len(t), "", dtype="<U1"), cal=cal, frame_wh=(int(frame_wh[0]), int(frame_wh[1])),
                    by_k=_index_by_k(k))


def build_cam_boxes(name: str, dets: list[Detection], cal: Calibration | CalibrationTimeline, offset_s: float,
                    drift: float, start_s: float, rate: float, court: Court, frame_wh=(1920, 1080)) -> CamBoxes:
    """Detections of one camera -> CamBoxes on the grid k = round((t + offset + drift*t - start) * rate).

    Detections saved before boxes were kept (no `box`) get a box rebuilt from the foot point and height with a
    typical player aspect (w = 0.4 h); the foot point itself stays exact. ReID rows are used as stored (the
    embedder normalised them), not renormalised, so this matches the research cache box for box."""
    n = len(dets)
    foot = np.array([[d.u, d.v] for d in dets], float).reshape(-1, 2)
    xyxy = np.zeros((n, 4))
    for i, d in enumerate(dets):
        if d.box is not None:
            xyxy[i] = d.box
        else:
            hw = 0.2 * d.h_px
            xyxy[i] = (d.u - hw, d.v - d.h_px, d.u + hw, d.v)
    reid = None
    dims = {len(d.reid) for d in dets if d.reid is not None}
    if dims:
        if len(dims) > 1:
            raise ValueError(f"{name}: ReID embeddings of different sizes {sorted(dims)}")
        reid = np.zeros((n, dims.pop()), np.float32)
        for i, d in enumerate(dets):
            if d.reid is not None and np.all(np.isfinite(d.reid)):
                reid[i] = d.reid
    color = None
    if any(d.feat is not None for d in dets):
        color = np.zeros((n, FEAT_LEN))
        for i, d in enumerate(dets):
            if d.feat is not None:
                color[i] = np.asarray(d.feat, float) / 255.0
    return _cam_boxes(name, [d.frame for d in dets], [d.t for d in dets], xyxy, [d.conf for d in dets], reid, color,
                      cal, offset_s, drift, start_s, rate, court, frame_wh, foot=foot)


def cam_boxes_from_cache(npz_path: str, name: str, cal: Calibration | CalibrationTimeline, offset: float, start: float,
                         rate: float, court: Court, drift: float = 0.0, frame_wh=(1920, 1080)) -> CamBoxes:
    """CamBoxes from a research detection cache (experiments/build_cache.py: fid, t, xyxy, conf, reid float16,
    color uint8), without the video. ReID is used as stored, like the research harness."""
    z = np.load(npz_path)
    n = len(z["t"])
    reid = z["reid"].astype(np.float32).reshape(n, -1) if "reid" in z and z["reid"].size else None
    color = z["color"].astype(float).reshape(n, -1) / 255.0 if "color" in z and z["color"].size else None
    return _cam_boxes(name, z["fid"], z["t"], z["xyxy"], z["conf"], reid, color, cal, offset, drift, start, rate, court,
                      frame_wh)


def cams_from_cache(cache_dir: str, root: str = ".", court: Court | None = None) -> tuple[dict[str, CamBoxes], dict]:
    """All cameras of a research cache directory (meta.json + <cam>.npz); calibration paths in meta.json are
    relative to `root` (the repository). Returns (cams, meta); no alignment, no teams."""
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    court = court or Court()
    cams = {}
    for cam in meta["videos"]:
        with open(os.path.join(root, meta["calib"][cam]), encoding="utf-8") as f:
            cal = timeline_from_json(json.load(f))
        cams[cam] = cam_boxes_from_cache(os.path.join(cache_dir, f"{cam}.npz"), cam, cal, float(meta["offsets"][cam]),
                                         float(meta["start"]), float(meta["rate"]), court)
    return cams, meta


# ---------------------------------------------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------------------------------------------

def _kmeans2(F: np.ndarray, rng: np.random.Generator, n_init: int = 5, iters: int = 100) -> np.ndarray:
    """Two-centre k-means (k-means++ seeding, best of n_init by inertia)."""
    best, best_cost = None, np.inf
    for _ in range(n_init):
        c0 = F[rng.integers(len(F))]
        d0 = ((F - c0) ** 2).sum(1)
        c1 = F[rng.choice(len(F), p=d0 / d0.sum())] if d0.sum() > 0 else F[rng.integers(len(F))]
        C = np.stack([c0, c1])
        for _ in range(iters):
            a = ((F[:, None] - C[None]) ** 2).sum(2).argmin(1)
            if (a == 0).all() or (a == 1).all():
                break
            new = np.stack([F[a == j].mean(0) for j in range(2)])
            if np.allclose(new, C):
                break
            C = new
        cost = ((F[:, None] - C[None]) ** 2).sum(2).min(1).sum()
        if cost < best_cost:
            best, best_cost = C, cost
    return best


def _cross_camera_pairs(a: CamBoxes, b: CamBoxes, min_conf: float = 0.5, max_d: float = 2.5, min_sim: float = 0.7,
                        step: int = 5) -> np.ndarray:
    """(i, j) box pairs that are the same person in cameras a and b: mutual nearest on the pitch (raw positions, the
    cameras are not aligned yet), within `max_d`, and alike by ReID. Every `step`-th frame is enough to vote."""
    from scipy.optimize import linear_sum_assignment
    if a.reid is None or b.reid is None:
        return np.zeros((0, 2), int)
    out = []
    for k in sorted(set(a.by_k) & set(b.by_k))[::step]:
        ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k, min_conf)
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy_raw[ia][:, None] - b.xy_raw[ib][None], axis=2)
        S = a.reid[ia] @ b.reid[ib].T
        r, c = linear_sum_assignment(D - 0.5 * S)
        for i, j in zip(r, c):
            if D[i, j] < max_d and S[i, j] > min_sim and D[i].argmin() == j and D[:, j].argmin() == i:
                out.append((ia[i], ib[j]))
    return np.array(out, int).reshape(-1, 2)


def assign_teams(cams: dict[str, CamBoxes], min_h: float = 40, min_conf: float = 0.4, label_min_h: float = 30,
                 margin: float = 0.95, max_fit: int = 20000, seed: int = 0, per_camera: bool = True,
                 min_pairs: int = 50) -> dict:
    """Split the boxes into two teams by upper-body colour and write cam.team ('A' / 'B' / '').

    per_camera (default): each phone gets its own two colour clusters, because phones render colours differently
    (a yellow bib looked lime in one phone of the 10/8 test and fell into the no-bib cluster of a joint fit, which
    then split one player into two identities). Which cluster of camera 2 is which of camera 1 is decided by the
    players both cameras see at the same moment (mutual nearest on the pitch and alike by ReID): the same person is
    on the same team whatever the colours look like. Without ReID or with fewer than `min_pairs` such pairs the
    joint fit below is used.

    Bibs or shirts are the one thing every player of a team shares, so two clusters of the upper-body histogram
    (shares per hue / black / grey / white bin) are the two teams. The clusters are fitted on confident, big,
    in-court boxes only: spectators and tiny boxes have mixed colours and would pull the centres. Every box at
    least `label_min_h` px tall is then labelled with its nearer centre unless the two distances are within
    `margin` of each other (''). A is the larger cluster. Returns a summary (fit size, centres, label counts),
    or {'n_fit': n} with every team '' when there is too little colour to fit.

    On the research window (one team in yellow bibs, the other without) this agrees with the yellow-share rule on
    99 % of the boxes both label, and labels 98 % of the boxes that rule labels."""
    for c in cams.values():
        c.team = np.full(len(c), "", dtype="<U1")
    rng = np.random.default_rng(seed)
    feats = {}
    for name, c in cams.items():
        if c.color is None or not len(c):
            continue
        up = c.color[:, :UPPER]
        s = up.sum(1)
        valid = s > 0
        F = up / np.maximum(s, 1e-9)[:, None]
        feats[name] = (F, valid, valid & (c.h >= min_h) & (c.conf >= min_conf) & c.in_court)

    def fit(F_fit: np.ndarray) -> np.ndarray:
        if len(F_fit) > max_fit:
            F_fit = F_fit[rng.choice(len(F_fit), max_fit, replace=False)]
        return _kmeans2(F_fit, rng)

    def label(name: str, C: np.ndarray) -> np.ndarray:
        """Cluster index per box, -1 where the colour is unclear or the box too small."""
        F, valid, _ = feats[name]
        D = np.sqrt(((F[:, None] - C[None]) ** 2).sum(2))
        ok = valid & (D.min(1) < margin * D.max(1)) & (cams[name].h >= label_min_h)
        return np.where(ok, D.argmin(1), -1)

    labels: dict[str, np.ndarray] = {}
    centres: dict[str, np.ndarray] = {}
    info: dict = {"mode": "joint"}
    names = list(feats)
    if per_camera and len(names) == 2 and all(feats[n][2].sum() >= 20 for n in names):
        for n in names:
            centres[n] = fit(feats[n][0][feats[n][2]])
            labels[n] = label(n, centres[n])
        P = _cross_camera_pairs(cams[names[0]], cams[names[1]])
        la = labels[names[0]][P[:, 0]] if len(P) else np.zeros(0, int)
        lb = labels[names[1]][P[:, 1]] if len(P) else np.zeros(0, int)
        both = (la >= 0) & (lb >= 0)
        if both.sum() >= min_pairs:
            same = int((la[both] == lb[both]).sum())
            diff = int(both.sum()) - same
            if diff > same:                              # camera 2's clusters are numbered the other way round
                labels[names[1]] = np.where(labels[names[1]] >= 0, 1 - labels[names[1]], -1)
                centres[names[1]] = centres[names[1]][::-1]
            info = {"mode": "per_camera", "pairs": int(both.sum()), "pair_agreement": round(max(same, diff) / both.sum(), 4)}
    if info["mode"] == "joint":
        F_fit = np.vstack([feats[n][0][feats[n][2]] for n in names]) if names else np.zeros((0, UPPER))
        if len(F_fit) < 20:
            return {"n_fit": int(len(F_fit))}
        C = fit(F_fit)
        for n in names:
            centres[n] = C
            labels[n] = label(n, C)
    # A is the larger team (over the confident in-court boxes), so labels do not depend on cluster numbering
    n0 = sum(int(((labels[n] == 0) & feats[n][2]).sum()) for n in names)
    n1 = sum(int(((labels[n] == 1) & feats[n][2]).sum()) for n in names)
    a_idx = 0 if n0 >= n1 else 1
    counts = {}
    for n in names:
        lab = labels[n]
        cams[n].team = np.where(lab < 0, "", np.where(lab == a_idx, "A", "B")).astype("<U1")
        counts[n] = {t: int((cams[n].team == t).sum()) for t in ("A", "B", "")}
    order = (lambda C: C) if a_idx == 0 else (lambda C: C[::-1])
    return info | {"n_fit": int(sum(feats[n][2].sum() for n in names)), "counts": counts,
                   "centres": {n: {"A": order(centres[n])[0].round(4).tolist(), "B": order(centres[n])[1].round(4).tolist()}
                               for n in names}}

