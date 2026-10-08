"""Short, pure tracklets: per-camera image-space tracking, then cross-view pairing of the two cameras' tracklets.

Purity before length: both stages cut whenever continuing could mix two people, and leave the long-range joining
to a linker that sees whole tracklets (appearance, team, reachability; pipeline.identity). Every box of the
selected frames ends up in exactly one output tracklet, so nothing is lost to an early decision.

    conservative_tracklets   one camera: Deep-EIoU-style association that cuts when unsure; one box per point
    pair_views               two cameras: fuse a cam-1 and a cam-2 tracklet over the samples where they agree
    build_tracklets          both stages with the defaults tuned on the 2026-10-08 test match
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .boxes import CamBoxes, Track, TrackPoint


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


def frame_overlap(cam: CamBoxes, idx: np.ndarray) -> np.ndarray:
    """Largest IoU of each box with any other box of the same camera and frame."""
    if len(idx) < 2:
        return np.zeros(len(idx))
    m = iou_matrix(cam.xyxy[idx], cam.xyxy[idx])
    np.fill_diagonal(m, 0.0)
    return m.max(1)


def default_frames(cams: dict[str, CamBoxes]) -> range:
    """Grid frames 0 .. last box of any camera. Boxes before the reference start (k < 0) are left out, as
    run.to_observations does."""
    last = max((int(c.k.max()) for c in cams.values() if len(c.k)), default=-1)
    return range(0, last + 1)


# ---------------------------------------------------------------------------------------------------------------
# Per-camera conservative builder
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _Live:
    box: np.ndarray                     # last observed box
    xy: np.ndarray                      # last pitch position
    sigma: float                        # its foot-point sigma, m
    last: int                           # last grid frame
    emb: np.ndarray | None              # EMA of reliable ReID features
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


def conservative_tracklets(cams: dict[str, CamBoxes], cam: str, rate: float, frames: Iterable[int] | None = None,
                           min_conf: float = 0.3, low_conf: float = 0.1, app_gate: float = 0.3,
                           eiou_schedule: tuple[float, ...] = (0.7, 0.85, 1.0), cut_iou: float = 0.5,
                           margin: float = 0.1, max_gap: int = 3, vmax: float = 9.0, veto_sigma: float = 2.0,
                           eiou_min: float = 0.3, app_min_h: float = 40.0, vel_damp: float = 0.0, ema: float = 0.9,
                           stats: CutStats | None = None) -> list[Track]:
    """Per-camera tracklets of camera `cam` that end whenever continuing could mix two people. Only in-court boxes
    of the grid frames `frames` (default: default_frames(cams)) take part; `rate` is the grid rate in Hz.

    Association per grid frame: Hungarian passes over expanded IoU with E stepped through `eiou_schedule` (a match
    found with a small expansion is the safer one), then a ByteTrack pass of left-over tracks against low-confidence
    boxes. A pair is valid only if EIoU >= eiou_min, the pitch step is <= vmax * dt + 1 m + veto_sigma * the combined
    foot-point sigma (far-side boxes move ~0.4 m per pixel; veto_sigma=0 is the plain rule), and - when both the track's
    EMA feature and the box are reliable (h >= app_min_h) - 1 - cos <= app_gate. Cost = min(1 - EIoU, 1 - cos)
    (Deep-EIoU), or 1 - EIoU without a reliable appearance (or without ReID at all).
    Cuts (the track ends, the box starts a new tracklet): the box overlaps another box of this camera with
    IoU > cut_iou (such boxes are emitted as single-sample tracklets), the best-vs-second-best cost margin in its row
    or column is < margin, or > max_gap samples are missing. Prediction is the last box shifted by vel_damp times
    the EMA image velocity (0 = last box; no Kalman: constant velocity overshoots at 10 Hz).
    Each TrackPoint holds exactly one box and that box's (aligned) pitch position."""
    from scipy.optimize import linear_sum_assignment
    c = cams[cam]
    st = stats if stats is not None else CutStats()
    has_reid = c.reid is not None
    live: list[_Live] = []
    done: list[_Live] = []
    dt = 1.0 / rate
    frames = default_frames(cams) if frames is None else frames

    def new(i: int, k: int) -> _Live:
        emb = c.reid[i].astype(np.float64) if has_reid and c.h[i] >= app_min_h else None
        return _Live(c.xyxy[i].copy(), c.xy[i].copy(), float(c.sigma[i]), k, emb, points={k: i})

    def extend(t: _Live, i: int, k: int) -> None:
        dk = k - t.last
        ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        t.vel = 0.5 * t.vel + 0.5 * (ctr(c.xyxy[i]) - ctr(t.box)) / dk
        t.box, t.xy, t.sigma, t.last = c.xyxy[i].copy(), c.xy[i].copy(), float(c.sigma[i]), k
        t.points[k] = i
        if has_reid and c.h[i] >= app_min_h and c.conf[i] >= 0.5:
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
        if has_reid:
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
                done.append(t)
                done.append(new(i, k))                  # an overlapped box is a tracklet of its own
                continue
            if second - cost[ti, di] < margin:
                st.margin += 1
                done.append(t)
                live_new.append(new(i, k))
                continue
            extend(t, i, k)
            live_new.append(t)
            st.matched += 1
            st.by_pass[tag] = st.by_pass.get(tag, 0) + 1
        for ti in range(len(tracks)):                   # veto bookkeeping: only vetoed candidates and no match
            if ti not in used_t and len(dets):
                w = why[ti][why[ti] > 1]
                if len(w) and not finite[ti].any():
                    if (w == 3).any():
                        st.app_veto += 1
                    elif (w == 2).any():
                        st.pitch_veto += 1
        return [i for i in range(len(tracks)) if i not in used_t], [j for j in range(len(dets)) if j not in used_d]

    for k in frames:
        idx = c.boxes_at(k, low_conf)
        ov = frame_overlap(c, idx)
        overlapped = {int(i): bool(o > cut_iou) for i, o in zip(idx, ov)}
        high = idx[c.conf[idx] >= min_conf]
        low = idx[c.conf[idx] < min_conf]
        keep = []                                        # tracks past the gap limit end here
        for t in live:
            if k - t.last - 1 > max_gap:
                st.gap += 1
                done.append(t)
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
    return [{k: TrackPoint(c.xy[i].copy(), [(cam, int(i))]) for k, i in sorted(t.points.items())}
            for t in sorted(done, key=lambda t: min(t.points))]


# ---------------------------------------------------------------------------------------------------------------
# Cross-view pairing
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class XviewStats:
    candidates: int = 0             # pairs sharing >= min_shared samples
    accepted: int = 0               # ... passing the distance / inside / team gates
    rejected_team: int = 0
    windows: int = 0                # distinct sets of simultaneously active accepted pairs
    paired_samples: int = 0         # (cam-1, cam-2) sample pairs fused
    ambiguous_samples: int = 0      # left unpaired because a swapped assignment was within tie_m
    cuts: int = 0                   # tracklet splits at partner changes
    conflict_tracklets: int = 0     # (cut_on_partner_change=False) tracklets that had to be cut anyway
    medians: list = field(default_factory=list)


def _team(cams: dict[str, CamBoxes], boxes: list[tuple[str, int]], min_labels: int = 3, share: float = 0.7) -> str:
    """Majority team of a tracklet: 'A' / 'B' when >= share of its labelled boxes agree, else ''."""
    labs = [cams[cam].team[i] for cam, i in boxes if cams[cam].team[i]]
    if len(labs) < min_labels:
        return ""
    a = sum(1 for x in labs if x == "A") / len(labs)
    return "A" if a >= share else "B" if a <= 1 - share else ""


def _single(p: TrackPoint) -> tuple[str, int]:
    if len(p.boxes) != 1:
        raise ValueError("pair_views expects single-view tracklets (one box per TrackPoint)")
    return p.boxes[0]


def _solve(cost: dict[tuple, float], unpaired: float, forbid: tuple | None = None) -> tuple[dict[tuple, float], float]:
    """Min-cost partial matching: every tracklet may stay unpaired at cost `unpaired` (>= any valid pair cost, so a
    valid pair always beats leaving both sides alone); `forbid` removes one pair but keeps its tracklets in the
    problem, so totals stay comparable. Returns ({(a, b): cost}, total cost)."""
    from scipy.optimize import linear_sum_assignment
    A = sorted({a for a, _ in cost}); B = sorted({b for _, b in cost})
    na, nb = len(A), len(B)
    big = 1e6
    M = np.full((na + nb, nb + na), big)
    for (a, b), c in cost.items():
        if (a, b) != forbid:
            M[A.index(a), B.index(b)] = c
    M[np.arange(na), nb + np.arange(na)] = unpaired
    M[na + np.arange(nb), np.arange(nb)] = unpaired
    M[na:, nb:] = 0.0
    r, c = linear_sum_assignment(M)
    total = float(M[r, c].sum())
    return {(A[i], B[j]): float(M[i, j]) for i, j in zip(r, c) if i < na and j < nb and M[i, j] < big}, total


def pair_views(t1: list[Track], t2: list[Track], cams: dict[str, CamBoxes], min_shared: int = 10, gate_m: float = 1.0,
               min_inside: float = 0.7, cut_on_partner_change: bool = True, tie_m: float = 0.1,
               sigma_gate: float = 0.0, stats: XviewStats | None = None) -> list[Track]:
    """Fuse one camera's single-view tracklets t1 with the other camera's t2.

    Candidates share >= min_shared samples; cost = median pitch distance over the shared samples. A pair is accepted
    if the median is <= gate (gate_m, or sigma_gate x the median combined foot-point sigma if larger), >= min_inside
    of the shared samples lie within 2 x gate, and the majority teams do not contradict. Then, for every window
    of frames with the same set of active accepted pairs, a min-cost partial matching picks at most one partner per
    tracklet (mutual best); an assigned pair is dropped for that window if forbidding it costs < tie_m more in total
    (a near-equal swap, typical of two players on the camera diagonal). A tracklet whose partner changes is split there
    (cut_on_partner_change), so each output tracklet has at most one partner per camera. Deciding per tracklet pair
    (evidence over >= 1 s) instead of per frame means one bad frame cannot fuse two players.
    Fused points carry both boxes and the inverse-variance (sigma) mean position. Every input box ends up in exactly
    one output tracklet."""
    st = stats if stats is not None else XviewStats()
    tr = {("1", i): t for i, t in enumerate(t1)} | {("2", i): t for i, t in enumerate(t2)}
    box = {key: {k: _single(p) for k, p in t.items()} for key, t in tr.items()}
    team = {key: _team(cams, list(b.values())) for key, b in box.items()}
    xy = lambda cb: cams[cb[0]].xy[cb[1]]
    sig = lambda cb: float(cams[cb[0]].sigma[cb[1]])

    # --- candidate pairs over shared samples
    at: dict[int, tuple[list, list]] = {}
    for key, b in box.items():
        for k in b:
            at.setdefault(k, ([], []))[0 if key[0] == "1" else 1].append(key)
    shared: dict[tuple, list] = {}
    for k, (A, B) in at.items():
        if not A or not B:
            continue
        P = np.array([xy(box[a][k]) for a in A]); Q = np.array([xy(box[b][k]) for b in B])
        D = np.linalg.norm(P[:, None] - Q[None], axis=2)
        S = np.hypot(np.array([sig(box[a][k]) for a in A])[:, None], np.array([sig(box[b][k]) for b in B])[None])
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                if D[i, j] < 6 * gate_m + 3 * sigma_gate * S[i, j]:     # far pairs can never pass the gates
                    shared.setdefault((a, b), []).append((k, D[i, j], S[i, j]))
    cost: dict[tuple, float] = {}
    for (a, b), rows in shared.items():
        n_both = len(set(box[a]) & set(box[b]))
        if n_both < min_shared:
            continue
        st.candidates += 1
        d = np.full(n_both, np.inf)
        d[:len(rows)] = [r[1] for r in rows]                     # samples outside the coarse cut count as outside
        gate = max(gate_m, sigma_gate * float(np.median([r[2] for r in rows])))
        med = float(np.median(d))
        if med > gate or (d <= 2 * gate).mean() < min_inside:
            continue
        if team[a] and team[b] and team[a] != team[b]:
            st.rejected_team += 1
            continue
        cost[(a, b)] = med
        st.medians.append(med)
    st.accepted = len(cost)

    # --- per-window mutual-best assignment
    partner: dict[tuple, dict[int, tuple]] = {key: {} for key in tr}
    memo: dict[frozenset, tuple[set, int]] = {}
    unpaired = gate_m if sigma_gate <= 0 else max(cost.values(), default=gate_m)
    for k, (A, B) in sorted(at.items()):
        act = frozenset(p for p in ((a, b) for a in A for b in B) if p in cost)
        if not act:
            continue
        if act not in memo:
            st.windows += 1
            sub = {p: cost[p] for p in act}
            best, total = _solve(sub, unpaired)
            keep = set()
            for p in best:
                if tie_m > 0 and len(sub) > 1:
                    _, alt = _solve(sub, unpaired, forbid=p)
                    if alt - total < tie_m:
                        continue
                keep.add(p)
            memo[act] = (keep, len(best) - len(keep))
        keep, amb = memo[act]
        st.ambiguous_samples += amb
        for a, b in keep:
            partner[a][k] = b; partner[b][k] = a
            st.paired_samples += 1

    # --- split at partner changes
    def segments(key, cut: bool) -> list[tuple[list[int], tuple | None]]:
        segs, cur, p = [], [], None
        for k in sorted(box[key]):
            q = partner[key].get(k)
            if cut and q is not None and p is not None and q != p:
                segs.append((cur, p)); cur = []
                st.cuts += 1
            cur.append(k)
            p = q if q is not None else p
        segs.append((cur, p))
        return segs

    must_cut = set()
    if not cut_on_partner_change:                                # cut only where whole-tracklet fusion would collide
        parent = {key: key for key in tr}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for key in tr:
            if key[0] == "1":
                for b in set(partner[key].values()):
                    parent[find(key)] = find(b)
        comps: dict = {}
        for key in tr:
            comps.setdefault(find(key), []).append(key)
        for members in comps.values():
            for side in ("1", "2"):
                ks = [k for m in members if m[0] == side for k in box[m]]
                if len(ks) != len(set(ks)):
                    must_cut.update(members)
        st.conflict_tracklets = len(must_cut)

    seg_of: dict[tuple, dict[int, int]] = {}
    seg_keys: list[tuple] = []                                   # (tracklet key, frames)
    for key in tr:
        seg_of[key] = {}
        for frames, _ in segments(key, cut_on_partner_change or key in must_cut):
            for k in frames:
                seg_of[key][k] = len(seg_keys)
            seg_keys.append((key, frames))

    # --- union mutually paired segments, emit fused tracklets
    root_of = list(range(len(seg_keys)))

    def root(x: int) -> int:
        while root_of[x] != x:
            root_of[x] = root_of[root_of[x]]; x = root_of[x]
        return x
    for a in (key for key in tr if key[0] == "1"):
        for k, b in partner[a].items():
            root_of[root(seg_of[a][k])] = root(seg_of[b][k])
    groups: dict[int, list[int]] = {}
    for s in range(len(seg_keys)):
        groups.setdefault(root(s), []).append(s)
    out: list[Track] = []
    for members in groups.values():
        pts: dict[int, list[tuple[str, int]]] = {}
        for s in members:
            key, frames = seg_keys[s]
            for k in frames:
                pts.setdefault(k, []).append(box[key][k])
        t: Track = {}
        for k in sorted(pts):
            bs = sorted(pts[k])
            if len({cam for cam, _ in bs}) != len(bs):
                raise AssertionError(f"two boxes of one camera at frame {k}")
            w = np.array([1.0 / max(sig(cb), 1e-6) ** 2 for cb in bs])
            t[k] = TrackPoint((np.array([xy(cb) for cb in bs]) * w[:, None]).sum(0) / w.sum(), bs)
        out.append(t)
    return sorted(out, key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# Both stages
# ---------------------------------------------------------------------------------------------------------------

_PERCAM_ARGS = set(inspect.signature(conservative_tracklets).parameters) - {"cams", "cam", "rate", "frames", "stats"}
_PAIR_ARGS = set(inspect.signature(pair_views).parameters) - {"t1", "t2", "cams", "stats"}


def build_tracklets(cams: dict[str, CamBoxes], rate: float = 10.0, frames: Iterable[int] | None = None,
                    **params) -> list[Track]:
    """Conservative per-camera tracklets of every camera, then (two cameras) pair_views between them.
    `params` go to conservative_tracklets or pair_views by name (e.g. margin=0.15, cut_on_partner_change=False);
    `frames` defaults to default_frames(cams). With one camera its single-view tracklets are returned as they are."""
    unknown = set(params) - _PERCAM_ARGS - _PAIR_ARGS
    if unknown:
        raise TypeError(f"build_tracklets() got unknown parameters: {sorted(unknown)}")
    if len(cams) > 2:
        raise NotImplementedError("pair_views pairs two cameras")
    frames = list(default_frames(cams) if frames is None else frames)
    pc = {k: v for k, v in params.items() if k in _PERCAM_ARGS}
    pv = {k: v for k, v in params.items() if k in _PAIR_ARGS}
    per = {cam: conservative_tracklets(cams, cam, rate, frames, **pc) for cam in cams}
    if len(per) < 2:
        return [t for ts in per.values() for t in ts]
    c1, c2 = list(cams)
    return pair_views(per[c1], per[c2], cams, **pv)
