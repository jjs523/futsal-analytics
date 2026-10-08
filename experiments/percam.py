"""Per-camera image-space tracklets (research_plan.md V3 step 1 and V5), the input of the cross-view pairing and the linkers.

Every TrackPoint here holds exactly one box of one camera, and its xy is that box's (cross-camera aligned) pitch position.
Two sources:
  conservative_tracklets()   custom Deep-EIoU-style builder that cuts whenever it is unsure (purity before length)
  boxmot_tracklets()         off-the-shelf boxmot trackers fed with the cached OSNet embeddings and a dummy image

    python experiments/harness.py pc_conservative percam
"""
from __future__ import annotations

import contextlib
import io
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from harness import CamData, Data, Track, TrackPoint, register

REID_WEIGHTS = Path(r"C:\Users\user\dev\models\osnet_ain_x1_0_msmt17.pt")
IMG_HW = (1080, 1920)


# ---------------------------------------------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------------------------------------------

def expand(boxes: np.ndarray, e: float) -> np.ndarray:
    """Deep-EIoU expansion: (w, h) -> ((1 + 2E) w, (1 + 2E) h) around the same centre. At 10 Hz a running player's
    box barely overlaps its previous one; the expanded boxes still do, while far-apart players stay apart."""
    w = (boxes[:, 2] - boxes[:, 0])[:, None]
    h = (boxes[:, 3] - boxes[:, 1])[:, None]
    return boxes + e * np.hstack([-w, -h, w, h])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def frame_overlap(cam: CamData, idx: np.ndarray) -> np.ndarray:
    """Largest IoU of each box with any other box of the same camera and frame."""
    if len(idx) < 2:
        return np.zeros(len(idx))
    m = iou_matrix(cam.xyxy[idx], cam.xyxy[idx])
    np.fill_diagonal(m, 0.0)
    return m.max(1)


# ---------------------------------------------------------------------------------------------------------------
# Conservative Deep-EIoU-style builder
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _Live:
    box: np.ndarray                     # last observed box
    xy: np.ndarray                      # last pitch position
    sigma: float                        # its foot-point sigma, m
    last: int                           # last grid frame
    emb: np.ndarray | None              # EMA of reliable OSNet features
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))   # image-space centre velocity, px per sample
    points: dict = field(default_factory=dict)


@dataclass
class CutStats:
    """Why tracklets ended; for tuning the purity / fragmentation trade-off."""
    overlap: int = 0
    margin: int = 0
    gap: int = 0
    app_veto: int = 0
    pitch_veto: int = 0
    matched: int = 0
    by_pass: dict = field(default_factory=dict)


def conservative_tracklets(data: Data, cam: str, min_conf: float = 0.3, low_conf: float = 0.1, app_gate: float = 0.3,
                           eiou_schedule: tuple[float, ...] = (0.7, 0.85, 1.0), cut_iou: float = 0.5, margin: float = 0.1,
                           max_gap: int = 3, vmax: float = 9.0, veto_sigma: float = 2.0, eiou_min: float = 0.3,
                           app_min_h: float = 40.0, vel_damp: float = 0.0, ema: float = 0.9, stats: CutStats | None = None) -> list[Track]:
    """Per-camera tracklets that end whenever continuing could mix two people.

    Association per grid frame: Hungarian passes over expanded IoU with E stepped through `eiou_schedule` (a match
    found with a small expansion is the safer one), then a ByteTrack pass of left-over tracks against low-confidence
    boxes. A pair is valid only if EIoU >= eiou_min, the pitch step is <= vmax * dt + 1 m + veto_sigma * the combined
    foot-point sigma (far-side boxes move ~0.4 m per pixel; veto_sigma=0 is the plain rule), and - when both the track's
    EMA feature and the box are reliable (h >= app_min_h) - 1 - cos <= app_gate. Cost = min(1 - EIoU, 1 - cos)
    (Deep-EIoU), or 1 - EIoU without a reliable appearance.
    Cuts (the track ends, the box starts a new tracklet): the box overlaps another box of this camera with
    IoU > cut_iou (such boxes are emitted as single-sample tracklets), the best-vs-second-best cost margin in its row
    or column is < margin, or > max_gap samples are missing. Prediction is the last box shifted by vel_damp times
    the EMA image velocity (0 = last box; no Kalman, constant velocity overshoots at 10 Hz)."""
    c = data.cams[cam]
    st = stats if stats is not None else CutStats()
    live: list[_Live] = []
    done: list[_Live] = []
    dt = 1.0 / data.rate

    def close(t: _Live) -> None:
        done.append(t)

    def new(i: int, k: int) -> _Live:
        emb = c.reid[i].astype(np.float64) if c.h[i] >= app_min_h else None
        return _Live(c.xyxy[i].copy(), c.xy[i].copy(), float(c.sigma[i]), k, emb, points={k: i})

    def extend(t: _Live, i: int, k: int) -> None:
        dk = k - t.last
        ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        t.vel = 0.5 * t.vel + 0.5 * (ctr(c.xyxy[i]) - ctr(t.box)) / dk
        t.box, t.xy, t.sigma, t.last = c.xyxy[i].copy(), c.xy[i].copy(), float(c.sigma[i]), k
        t.points[k] = i
        if c.h[i] >= app_min_h and c.conf[i] >= 0.5:
            f = c.reid[i].astype(np.float64)
            t.emb = f if t.emb is None else ema * t.emb + (1 - ema) * f
            t.emb /= max(np.linalg.norm(t.emb), 1e-9)

    def costs(tracks: list[_Live], dets: np.ndarray, k: int, e: float) -> tuple[np.ndarray, np.ndarray]:
        """(cost, why-invalid) matrices; invalid entries are np.inf. why: 0 ok, 1 eiou, 2 pitch, 3 appearance."""
        pred = np.array([t.box + vel_damp * (k - t.last) * np.r_[t.vel, t.vel] for t in tracks])
        eiou = iou_matrix(expand(pred, e), expand(c.xyxy[dets], e))
        cost = 1.0 - eiou
        why = np.where(eiou < eiou_min, 1, 0)
        step = np.linalg.norm(np.array([t.xy for t in tracks])[:, None] - c.xy[dets][None], axis=2)
        reach = vmax * dt * np.array([k - t.last for t in tracks], float)[:, None] + 1.0
        reach = reach + veto_sigma * np.hypot(np.array([t.sigma for t in tracks])[:, None], c.sigma[dets][None])
        why = np.where((why == 0) & (step > reach), 2, why)
        temb = [t.emb if t.emb is not None else np.zeros(c.reid.shape[1]) for t in tracks]
        dapp = 1.0 - np.array(temb) @ c.reid[dets].astype(np.float64).T
        reliable = np.array([t.emb is not None for t in tracks])[:, None] & (c.h[dets] >= app_min_h)[None]
        why = np.where((why == 0) & reliable & (dapp > app_gate), 3, why)
        cost = np.where(reliable, np.minimum(cost, dapp), cost)
        return np.where(why == 0, cost, np.inf), why

    def assign(tracks: list[_Live], dets: np.ndarray, k: int, e: float, tag: str) -> tuple[list[int], list[int]]:
        """One Hungarian pass. Returns (unmatched track positions, unmatched det positions); matched tracks are
        extended or, when ambiguous / overlapped, closed with their box starting a new tracklet."""
        if not tracks or not len(dets):
            return list(range(len(tracks))), list(range(len(dets)))
        cost, why = costs(tracks, dets, k, e)
        finite = np.isfinite(cost)
        r, cidx = linear_sum_assignment(np.where(finite, cost, 1e6))
        used_t, used_d = set(), set()
        for ti, di in zip(r, cidx):
            if not finite[ti, di]:
                continue
            row = np.delete(cost[ti], di)
            col = np.delete(cost[:, di], ti)
            second = min(row.min() if len(row) else np.inf, col.min() if len(col) else np.inf)
            used_t.add(ti); used_d.add(di)
            t, i = tracks[ti], int(dets[di])
            if overlapped[i]:
                st.overlap += 1
                close(t)
                done.append(new(i, k))                  # an overlapped box is a tracklet of its own
                continue
            if second - cost[ti, di] < margin:
                st.margin += 1
                close(t)
                live_new.append(new(i, k))
                continue
            extend(t, i, k)
            live_new.append(t)
            st.matched += 1
            st.by_pass[tag] = st.by_pass.get(tag, 0) + 1
        # veto bookkeeping: a track that had only vetoed candidates (and no match)
        for ti in range(len(tracks)):
            if ti not in used_t and len(dets):
                w = why[ti][why[ti] > 1]
                if len(w) and not finite[ti].any():
                    if (w == 3).any():
                        st.app_veto += 1
                    elif (w == 2).any():
                        st.pitch_veto += 1
        return [i for i in range(len(tracks)) if i not in used_t], [j for j in range(len(dets)) if j not in used_d]

    for k in range(data.n):
        idx = c.boxes_at(k, low_conf)
        ov = frame_overlap(c, idx)
        overlapped = {int(i): bool(o > cut_iou) for i, o in zip(idx, ov)}
        high = idx[c.conf[idx] >= min_conf]
        low = idx[c.conf[idx] < min_conf]
        # tracks past the gap limit end here
        keep = []
        for t in live:
            if k - t.last - 1 > max_gap:
                st.gap += 1
                close(t)
            else:
                keep.append(t)
        live_new: list[_Live] = []
        tracks, dets = keep, high
        for e in eiou_schedule:
            ut, ud = assign(tracks, dets, k, e, f"E{e}")
            tracks, dets = [tracks[i] for i in ut], dets[ud]
        ut, _ = assign(tracks, low, k, eiou_schedule[-1], "low")   # low boxes never start tracklets (ByteTrack)
        tracks = [tracks[i] for i in ut]
        for i in dets:                                   # unmatched confident boxes start tracklets
            if overlapped[int(i)]:
                st.overlap += 1
                done.append(new(int(i), k))
            else:
                live_new.append(new(int(i), k))
        live = live_new + tracks                         # unmatched tracks coast
    done += live
    out = []
    for t in sorted(done, key=lambda t: min(t.points)):
        out.append({k: TrackPoint(c.xy[i].copy(), [(cam, int(i))]) for k, i in sorted(t.points.items())})
    return out


# ---------------------------------------------------------------------------------------------------------------
# boxmot trackers on cached detections + embeddings
# ---------------------------------------------------------------------------------------------------------------

class _CachedReid:
    """Stands in for boxmot's ReID backend: no model is loaded, get_features() serves the cached OSNet features of the
    current frame and counts how often a tracker asks (a tracker that honours update(embs=...) never asks)."""

    def __init__(self, weights=None, device=None, half=False):
        self.weights = weights
        self.calls = 0
        self.embs = np.zeros((0, 512))
        self.model = self

    def get_features(self, xyxy: np.ndarray, img: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert len(xyxy) == len(self.embs), "tracker asked for features of a different box set"
        return self.embs.copy()


_IDENTITY_WARP = np.eye(2, 3)


def _unfreeze(self) -> None:
    """KalmanFilterXYSR.unfreeze (OC-SORT observation-centric re-update) made to run here: boxmot 12 calls float() on
    1-element arrays, which NumPy 2.5 rejects, and unpacks 4 values from HybridSort's 5-d [x, y, s, score, r]
    measurements. Same interpolation otherwise: virtual boxes linear in centre / width / height (and score)."""
    from collections import deque
    from copy import deepcopy
    if self.attr_saved is None:
        return
    new_history = deepcopy(list(self.history_obs))
    self.__dict__ = self.attr_saved
    self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
    seen = np.flatnonzero([d is not None for d in new_history])
    i1, i2 = seen[-2], seen[-1]
    b1, b2 = (np.asarray(new_history[i], float).ravel() for i in (i1, i2))
    wh = lambda b: (np.sqrt(b[2] * b[-1]), np.sqrt(b[2] / b[-1]))
    (w1, h1), (w2, h2) = wh(b1), wh(b2)
    gap = i2 - i1
    for i in range(gap):
        f = (i + 1) / gap
        x, y = b1[0] + f * (b2[0] - b1[0]), b1[1] + f * (b2[1] - b1[1])
        w, h = w1 + f * (w2 - w1), h1 + f * (h2 - h1)
        z = [x, y, w * h, w / h] if len(b1) == 4 else [x, y, w * h, b1[3] + f * (b2[3] - b1[3]), w / h]
        self.update(np.array(z).reshape((-1, 1)))
        if i != gap - 1:
            self.predict()
            self.history_obs.pop()
    self.history_obs.pop()


@contextlib.contextmanager
def _patched_kf():
    from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
    orig = KalmanFilterXYSR.unfreeze
    KalmanFilterXYSR.unfreeze = _unfreeze
    try:
        yield
    finally:
        KalmanFilterXYSR.unfreeze = orig

_DEFAULTS = {   # 30 fps defaults rescaled to 10 Hz: ages / buffers / delta_t divided by 3, min_hits 1
    "bytetrack": dict(min_conf=0.1, track_thresh=0.4, match_thresh=0.8, track_buffer=20, frame_rate=30),
    "ocsort": dict(min_conf=0.1, det_thresh=0.4, max_age=20, min_hits=1, asso_threshold=0.3, delta_t=1,
                   inertia=0.2, use_byte=True),
    "hybridsort": dict(det_thresh=0.4, max_age=20, min_hits=1, iou_threshold=0.3, delta_t=1, inertia=0.2,
                       longterm_reid_weight=0.0, TCM_first_step_weight=0.0, use_byte=False),
    "deepocsort": dict(det_thresh=0.4, max_age=20, min_hits=1, iou_threshold=0.3, delta_t=1, inertia=0.2,
                       w_association_emb=1.25, alpha_fixed_emb=0.95, aw_param=0.5, cmc_off=True),
    # (the plan's aw_param 1.0 divides by 1 - aw_param in boxmot's compute_aw_max_metric)
    "boosttrack": dict(max_age=20, min_hits=1, det_thresh=0.4, iou_threshold=0.3, use_ecc=False, with_reid=True),
    "botsort": dict(track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.5, track_buffer=20,
                    match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25, cmc_method="ecc", frame_rate=30,
                    with_reid=True),
}
_CLASSES = {
    "bytetrack": "boxmot.trackers.bytetrack.bytetrack.ByteTrack",
    "ocsort": "boxmot.trackers.ocsort.ocsort.OcSort",
    "hybridsort": "boxmot.trackers.hybridsort.hybridsort.HybridSort",
    "deepocsort": "boxmot.trackers.deepocsort.deepocsort.DeepOcSort",
    "boosttrack": "boxmot.trackers.boosttrack.boosttrack.BoostTrack",
    "botsort": "boxmot.trackers.botsort.botsort.BotSort",
}
_REID = {"hybridsort", "deepocsort", "boosttrack", "botsort"}
_DET_IND_BROKEN = {"hybridsort"}        # its det_ind column holds the detection score; map its rows by box IoU


def make_boxmot(tracker: str, **params):
    """Instantiate a boxmot tracker with 10 Hz defaults, its ReID backend replaced by _CachedReid and camera-motion
    compensation off (fixed tripods): cmc_off / use_ecc=False where the tracker has a switch, and cmc.apply stubbed
    to an identity warp everywhere else (BotSort always runs it; HybridSort only if its ECC flag is set)."""
    import importlib
    mod_name, cls_name = _CLASSES[tracker].rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    kw = _DEFAULTS[tracker] | params
    if tracker in _REID:
        kw = dict(reid_weights=REID_WEIGHTS, device="cpu", half=False) | kw
    real = getattr(mod, "ReidAutoBackend", None)
    if real is not None:
        mod.ReidAutoBackend = _CachedReid               # no model load, no CUDA_VISIBLE_DEVICES side effect
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            trk = getattr(mod, cls_name)(**kw)
    finally:
        if real is not None:
            mod.ReidAutoBackend = real
    reid = None
    for attr in ("model", "reid_model"):
        if isinstance(getattr(trk, attr, None), _CachedReid):
            reid = getattr(trk, attr)
    if hasattr(trk, "cmc") and trk.cmc is not None:
        trk.cmc.apply = lambda img, dets=None: _IDENTITY_WARP.copy()
    if tracker == "hybridsort":
        trk.ECC = False
    return trk, reid


@dataclass
class BoxmotStats:
    rows: int = 0
    det_ind_ok: int = 0
    box_fallback: int = 0
    unmapped: int = 0
    reused: int = 0
    reid_calls: int = 0


def boxmot_tracklets(data: Data, cam: str, tracker: str = "hybridsort", min_conf: float = 0.1,
                     stats: BoxmotStats | None = None, **params) -> list[Track]:
    """Run a boxmot tracker over the grid frames of one camera with the cached embeddings (update(dets, img, embs))
    and a dummy zero image. Output rows [x1,y1,x2,y2,id,conf,cls,det_ind] are mapped back to box indices by det_ind
    (the detection the tracker associated, even if its Kalman output box drifted off it); rows without a valid det_ind
    fall back to the best-overlapping box with IoU > 0.5 (HybridSort stores the score as det_ind and also ignores embs,
    so its features come in through _CachedReid.get_features)."""
    c = data.cams[cam]
    st = stats if stats is not None else BoxmotStats()
    trk, reid = make_boxmot(tracker, **params)
    img = np.zeros((*IMG_HW, 3), np.uint8)
    tracks: dict[int, Track] = {}
    sink = io.StringIO()
    for k in range(data.n):
        idx = c.boxes_at(k, min_conf)
        dets = np.hstack([c.xyxy[idx], c.conf[idx][:, None], np.zeros((len(idx), 1))]) if len(idx) else np.empty((0, 6))
        embs = c.reid[idx].astype(np.float64) if len(idx) else np.empty((0, c.reid.shape[1]))
        if reid is not None:
            reid.embs = embs
        with contextlib.redirect_stdout(sink), _patched_kf():   # HybridSort prints on every BYTE correction
            out = trk.update(dets, img, embs)
        if sink.tell() > 1 << 20:
            sink.seek(0); sink.truncate()
        out = np.asarray(out)
        if out.size == 0 or not len(idx):
            continue
        out = out.reshape(-1, out.shape[-1])
        ious = iou_matrix(out[:, :4].astype(float), c.xyxy[idx])
        taken = set()
        for r, row in enumerate(out):
            st.rows += 1
            j = int(round(row[7])) if np.isfinite(row[7]) else -1
            # det_ind is the detection the tracker actually associated this frame; trust it even when the output box
            # is a Kalman box that has drifted (IoU <= 0.5), because the IoU argmax can then be a neighbour's box
            if tracker not in _DET_IND_BROKEN and 0 <= j < len(idx) and abs(row[7] - j) < 1e-6:
                st.det_ind_ok += 1
            else:
                j = int(ious[r].argmax())
                if ious[r, j] <= 0.5:
                    st.unmapped += 1
                    continue
                st.box_fallback += 1
            if j in taken:
                st.reused += 1
                continue
            taken.add(j)
            i = int(idx[j])
            tracks.setdefault(int(row[4]), {})[k] = TrackPoint(c.xy[i].copy(), [(cam, i)])
    st.reid_calls += reid.calls if reid is not None else 0
    return sorted(tracks.values(), key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# Shared helpers and registered variants
# ---------------------------------------------------------------------------------------------------------------

_CACHE: "weakref.WeakKeyDictionary[Data, dict]" = weakref.WeakKeyDictionary()   # not id(data): ids are reused after GC


def per_camera(data: Data, source: str = "conservative", **params) -> dict[str, list[Track]]:
    """{cam: tracklets} for a tracklet source, memoised per Data object and parameter set within one run."""
    cache = _CACHE.setdefault(data, {})
    key = (source, tuple(sorted(params.items())))
    if key not in cache:
        if source == "conservative":
            cache[key] = {cam: conservative_tracklets(data, cam, **params) for cam in data.cams}
        else:
            cache[key] = {cam: boxmot_tracklets(data, cam, source, **params) for cam in data.cams}
    return cache[key]


def check_single_view(tracks: list[Track], data: Data) -> dict:
    """Invariants of a per-camera tracklet set: one box per point, xy = that box's position, every box used once."""
    used, bad = set(), 0
    for t in tracks:
        for k, p in t.items():
            if len(p.boxes) != 1:
                bad += 1
                continue
            cam, i = p.boxes[0]
            c = data.cams[cam]
            if c.k[i] != k or not np.allclose(p.xy, c.xy[i]) or (cam, i) in used:
                bad += 1
            used.add((cam, i))
    return {"points": sum(len(t) for t in tracks), "violations": bad}


def _concat(per: dict[str, list[Track]]) -> list[Track]:
    return [t for cam in per for t in per[cam]]


@register("pc_conservative")
def pc_conservative(data: Data) -> list[Track]:
    """Both cameras' conservative tracklets, no cross-view merge (players seen twice are double-counted)."""
    return _concat(per_camera(data, "conservative"))


@register("pc_conservative_vel")
def pc_conservative_vel(data: Data) -> list[Track]:
    return _concat(per_camera(data, "conservative", vel_damp=0.5))


for _name in ("hybridsort", "deepocsort", "bytetrack", "ocsort", "boosttrack", "botsort"):
    register(f"pc_{_name}")(lambda data, _n=_name: _concat(per_camera(data, _n)))
