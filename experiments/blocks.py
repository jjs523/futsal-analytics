"""Shared building blocks B1-B6 of experiments/research_plan.md, operating on harness Tracks ({k: TrackPoint}).

Every function is pure (takes Data / Tracks, returns new objects) except for two per-Data caches that are filled on
first use and keyed weakly on the Data object: the per-box pitch covariances (B2) and the default clean-crop masks
(B4). Both depend only on the cached detections, so caching them cannot change a result.

    B1 team_vote, split_team_flips            bib colour per tracklet, split where it flips for good
    B2 covariance, box_covariances, fuse      anisotropic pitch covariance from the homography Jacobian
    B3 reachable                              can one person get from the end of A to the start of B
    B4 clean_masks, track_embedding,          robust ReID from crops that show one whole person
       app_distance, clean_feats
    B5 overlap, cannot_link                   hard constraints for any linker
    B6 fill_gaps, smooth                      cosmetic gap filling once identities are fixed
       dbscan_split                           GTA-style purity splitter on clean-crop ReID
"""
from __future__ import annotations

import weakref

import numpy as np

from harness import Data, Track, TrackPoint
from futsal.homography import calibration_at

FRAME_WH = (1920, 1080)          # both phones record 1920x1080 (checked on the source videos)
HOMOGRAPHY_FLOOR_M = 0.15        # B2: absorbs homography / alignment error that pixel noise does not explain
NO_BOX_VAR = 0.5 ** 2            # variance (m^2) assumed for interpolated points that carry no boxes

_COV_CACHE: "weakref.WeakKeyDictionary[Data, dict[str, np.ndarray]]" = weakref.WeakKeyDictionary()
_CLEAN_CACHE: "weakref.WeakKeyDictionary[Data, dict[str, np.ndarray]]" = weakref.WeakKeyDictionary()


def _keys(track: Track) -> np.ndarray:
    return np.array(sorted(track), int)


def _sub(track: Track, ks) -> Track:
    return {int(k): track[int(k)] for k in ks}


# ---------------------------------------------------------------------------------------------------------------
# B1. Team vote
# ---------------------------------------------------------------------------------------------------------------

def _labelled(track: Track, data: Data, min_h: float, min_conf: float):
    """(k, weight, is_yellow) of every box of the track whose bib colour is clear and that is big enough to trust."""
    out = []
    for k, p in track.items():
        for cam, i in p.boxes:
            c = data.cams[cam]
            if c.h[i] >= min_h and c.conf[i] >= min_conf and c.team[i]:
                out.append((k, float(c.h[i]), c.team[i] == "Y"))
    return out


def team_vote(track: Track, data: Data, min_h: float = 40, min_conf: float = 0.4) -> tuple[str, float]:
    """'Y' / 'N' / 'U' and the box-height-weighted yellow share of the track's clearly labelled boxes.
    Big boxes count more because the bib hue of a 40 px far-side player is unreliable. NaN share when nothing
    is labelled (the vote is then 'U')."""
    lab = _labelled(track, data, min_h, min_conf)
    if not lab:
        return "U", float("nan")
    w = np.array([x[1] for x in lab])
    y = np.array([x[2] for x in lab], float)
    share = float((w * y).sum() / w.sum())
    return ("Y" if share >= 0.7 else "N" if share <= 0.3 else "U"), share


def _sample_shares(track: Track, data: Data, min_h: float = 40, min_conf: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample weighted yellow share, only for samples with at least one labelled box."""
    acc: dict[int, list[float]] = {}
    for k, w, y in _labelled(track, data, min_h, min_conf):
        a = acc.setdefault(k, [0.0, 0.0])
        a[0] += w * y
        a[1] += w
    ks = np.array(sorted(acc), int)
    return ks, np.array([acc[k][0] / acc[k][1] for k in ks])


def split_team_flips(tracks: list[Track], data: Data, run: int = 15, share: float = 0.8) -> list[Track]:
    """B1 split: cut a track where its bib colour flips for good, i.e. the `run` labelled samples before the cut are
    >= `share` one team and the `run` after are >= `share` the other. One-off flips (a teammate's bib occluding
    the player, a bad crop) never reach `share` over 1.5 s, so they do not cut."""
    out = []
    for tr in tracks:
        ks, sh = _sample_shares(tr, data)
        if len(ks) < 2 * run:
            out.append(tr)
            continue
        y = (sh >= 0.5).astype(float)
        c = np.concatenate([[0.0], np.cumsum(y)])
        j = np.arange(run, len(y) - run + 1)                    # cut between labelled samples j-1 and j
        before = (c[j] - c[j - run]) / run
        after = (c[j + run] - c[j]) / run
        score = np.maximum(np.minimum(before, 1 - after), np.minimum(1 - before, after))
        ok = score >= share
        cuts = []
        q = 0
        while q < len(j):                                       # best position inside each contiguous candidate stretch
            if not ok[q]:
                q += 1
                continue
            r = q
            while r + 1 < len(j) and ok[r + 1]:
                r += 1
            best = q + int(np.argmax(score[q:r + 1]))
            if not cuts or j[best] - cuts[-1] >= run:
                cuts.append(int(j[best]))
            q = r + 1
        if not cuts:
            out.append(tr)
            continue
        all_k = _keys(tr)
        bounds = [all_k[0]] + [int(ks[cj]) for cj in cuts] + [all_k[-1] + 1]
        for lo, hi in zip(bounds, bounds[1:]):
            piece = _sub(tr, all_k[(all_k >= lo) & (all_k < hi)])
            if piece:
                out.append(piece)
    return out


# ---------------------------------------------------------------------------------------------------------------
# B2. Anisotropic pitch covariance
# ---------------------------------------------------------------------------------------------------------------

def _pitch_jacobian(cal, uv: np.ndarray) -> np.ndarray:
    """(n, 2, 2) d(pitch)/d(pixel) by central differences on the raw homography."""
    e = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cols = [(cal.to_pitch(uv + d) - cal.to_pitch(uv - d)) / 2.0 for d in e]
    return np.stack(cols, axis=2)


def box_covariances(data: Data, cam: str, sx: float = 0.03, sy: float = 0.05, min_px: float = 1.0) -> np.ndarray:
    """(n_boxes, 2, 2) pitch covariance of every box of `cam` (cached per Data for the default parameters).
    Pixel noise of the foot point grows with the box (0.03 w across, 0.05 h along: feet are vaguer vertically),
    mapped through the local Jacobian, which stretches it along the viewing ray on the far side."""
    default = (sx, sy, min_px) == (0.03, 0.05, 1.0)
    if default and cam in _COV_CACHE.get(data, {}):
        return _COV_CACHE[data][cam]
    c = data.cams[cam]
    R = np.zeros((len(c.t), 2, 2))
    if len(c.t):
        cals = [calibration_at(c.cal, float(t)) for t in c.t]
        groups: dict[int, list[int]] = {}
        for i, cal in enumerate(cals):
            groups.setdefault(id(cal), []).append(i)
        for idx in groups.values():
            idx = np.array(idx)
            J = _pitch_jacobian(cals[idx[0]], c.foot[idx])
            px = np.stack([np.maximum(sx * c.w[idx], min_px), np.maximum(sy * c.h[idx], min_px)], 1) ** 2
            R[idx] = np.einsum("nij,nj,nkj->nik", J, px, J)
        R += HOMOGRAPHY_FLOOR_M ** 2 * np.eye(2)
    if default:
        _COV_CACHE.setdefault(data, {})[cam] = R
    return R


def covariance(data: Data, cam: str, idx: int) -> np.ndarray:
    """2x2 pitch covariance (m^2) of one box (B2)."""
    return box_covariances(data, cam)[int(idx)]


def fuse(data: Data, boxes: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-covariance fusion of several boxes' (aligned) pitch positions: the camera that is precise along a
    direction dominates along that direction, instead of one scalar weight per camera."""
    if not boxes:
        raise ValueError("fuse() needs at least one box")
    P = np.zeros((2, 2))
    b = np.zeros(2)
    for cam, i in boxes:
        Ri = np.linalg.inv(covariance(data, cam, i))
        P += Ri
        b += Ri @ data.cams[cam].xy[i]
    cov = np.linalg.inv(P)
    return cov @ b, cov


def point_cov(data: Data, p: TrackPoint) -> np.ndarray:
    """Covariance of a TrackPoint: fused from its boxes, or a generous isotropic default for interpolated points."""
    return fuse(data, p.boxes)[1] if p.boxes else NO_BOX_VAR * np.eye(2)


def mahalanobis2(d: np.ndarray, R: np.ndarray) -> float:
    """Squared Mahalanobis length of offset d under covariance R (gate: < 9.21 = chi2(2) 99%)."""
    return float(d @ np.linalg.solve(R, d))


# ---------------------------------------------------------------------------------------------------------------
# B3 / B5. Reachability and cannot-links
# ---------------------------------------------------------------------------------------------------------------

def overlap(a: Track, b: Track) -> int:
    """Number of frames both tracks claim."""
    return len(a.keys() & b.keys())


def _junctions(a: Track, b: Track):
    """Every hand-over between a and b on the merged timeline: (k1, point1, k2, point2) for consecutive samples owned
    by different tracks. For a ending before b starts this is the single pair (end of a, start of b); shared frames
    give pairs with k1 == k2."""
    ev = sorted([(k, 0) for k in a] + [(k, 1) for k in b])
    src = (a, b)
    for (k1, o1), (k2, o2) in zip(ev, ev[1:]):
        if o1 != o2:
            yield k1, src[o1][k1], k2, src[o2][k2]


def _reach_ok(p1: TrackPoint, p2: TrackPoint, dt: float, data: Data, vmax: float, slack: float) -> bool:
    lam = float(np.linalg.eigvalsh(point_cov(data, p1) + point_cov(data, p2))[-1])
    return float(np.linalg.norm(p2.xy - p1.xy)) <= vmax * dt + slack + 3.0 * np.sqrt(lam)


def reachable(a: Track, b: Track, data: Data, vmax: float = 8.0, slack: float = 1.0) -> bool:
    """B3: ||p_B - p_A|| <= vmax dt + slack + 3 sqrt(lambda_max(R_A + R_B)) at the hand-over from a to b (a ends
    before b starts). If the two interleave, every hand-over on the merged timeline must pass, so the same test
    serves for joining multi-part tracks."""
    if not a or not b:
        return True
    return all(_reach_ok(p1, p2, (k2 - k1) / data.rate, data, vmax, slack) for k1, p1, k2, p2 in _junctions(a, b))


def cannot_link(a: Track, b: Track, data: Data, team_a: str | None = None, team_b: str | None = None,
                vmax: float = 8.0, slack: float = 1.0, max_overlap: int = 1) -> bool:
    """B5 hard constraint: True if a and b cannot be the same person (time overlap > max_overlap samples, Y vs N
    bibs, or an impossible run between them). 'U' is compatible with both teams."""
    if overlap(a, b) > max_overlap:
        return True
    ta = team_a if team_a is not None else team_vote(a, data)[0]
    tb = team_b if team_b is not None else team_vote(b, data)[0]
    if {ta, tb} == {"Y", "N"}:
        return True
    return not reachable(a, b, data, vmax, slack)


# ---------------------------------------------------------------------------------------------------------------
# B4. Clean crops and tracklet appearance
# ---------------------------------------------------------------------------------------------------------------

def _iou_matrix(b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def clean_masks(data: Data, min_h: float = 60, min_conf: float = 0.5, max_iou: float = 0.1, border: float = 0.02,
                other_conf: float = 0.25) -> dict[str, np.ndarray]:
    """B4: per camera, a bool per box that is True when the crop shows one whole person: big and confident enough,
    IoU < max_iou with every other box (conf >= other_conf) of the same camera and frame, and at least
    border * frame width away from every image edge. Only such crops give a ReID vector of one identity."""
    key = (min_h, min_conf, max_iou, border, other_conf)
    default = key == (60, 0.5, 0.1, 0.02, 0.25)
    if default and data in _CLEAN_CACHE:
        return _CLEAN_CACHE[data]
    W, H = FRAME_WH
    m = border * W
    out = {}
    for cam, c in data.cams.items():
        x = c.xyxy
        ok = (c.h >= min_h) & (c.conf >= min_conf) & (x[:, 0] >= m) & (x[:, 1] >= m) & (x[:, 2] <= W - m) & (x[:, 3] <= H - m)
        for idx in c.by_k.values():
            if len(idx) < 2 or not ok[idx].any():
                continue
            iou = _iou_matrix(x[idx])
            np.fill_diagonal(iou, 0.0)
            iou[:, c.conf[idx] < other_conf] = 0.0
            ok[idx] &= iou.max(1) < max_iou
        out[cam] = ok
    if default:
        _CLEAN_CACHE[data] = out
    return out


def clean_feats(track: Track, data: Data, clean: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ks, feats): one ReID vector per frame that has a clean box, taken from the camera where the box is larger
    (a fused point usually has a near view and a far view; the near crop carries the identity)."""
    clean = clean_masks(data) if clean is None else clean
    ks, rows = [], []
    for k in sorted(track):
        best = None
        for cam, i in track[k].boxes:
            if clean[cam][i] and (best is None or data.cams[cam].h[i] > data.cams[best[0]].h[best[1]]):
                best = (cam, i)
        if best is not None:
            ks.append(k)
            rows.append(data.cams[best[0]].reid[best[1]])
    if not rows:
        return np.zeros(0, int), np.zeros((0, 512), np.float32)
    return np.array(ks, int), np.stack(rows).astype(np.float32)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def medoid(feats: np.ndarray, max_n: int = 500) -> np.ndarray:
    """The feature with the highest summed cosine similarity to the others (on an evenly spaced subsample)."""
    f = feats[np.linspace(0, len(feats) - 1, min(max_n, len(feats))).astype(int)] if len(feats) > max_n else feats
    return f[int(np.argmax((f @ f.T).sum(1)))]


def track_embedding(track: Track, data: Data, clean: dict[str, np.ndarray] | None = None, max_medoids: int = 10,
                    min_crops: int = 3) -> dict | None:
    """B4 appearance of a track: L2-normalised mean of its clean crops plus up to `max_medoids` medoids, one per
    contiguous time chunk, so a pose / lighting change along the track is represented instead of averaged away.
    None when fewer than `min_crops` clean crops exist (far-side tracklets: link them by motion and team only)."""
    ks, f = clean_feats(track, data, clean)
    if len(f) < min_crops:
        return None
    m = min(max_medoids, max(1, len(f) // min_crops))
    meds = np.stack([medoid(chunk) for chunk in np.array_split(f, m)])
    return {"mean": normalise(f.mean(0)), "medoids": meds, "n": len(f), "span": (int(ks[0]), int(ks[-1]))}


def app_distance(e1: dict | None, e2: dict | None) -> float:
    """B4: min(1 - cos(means), 20th percentile of pairwise medoid cosine distances), in [0, 2]; 0.5 (uninformative)
    when either side has no clean crops. The percentile lets two tracks match on the poses they share."""
    if e1 is None or e2 is None:
        return 0.5
    d_mean = 1.0 - float(e1["mean"] @ e2["mean"])
    d_med = float(np.percentile(1.0 - e1["medoids"] @ e2["medoids"].T, 20))
    return float(np.clip(min(d_mean, d_med), 0.0, 2.0))


def dbscan_labels(feats: np.ndarray, eps: float = 0.55, min_samples: int = 5) -> np.ndarray:
    """DBSCAN (cosine) cluster label per crop; -1 is noise."""
    from sklearn.cluster import DBSCAN
    if len(feats) < min_samples:
        return np.full(len(feats), -1)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute").fit(feats).labels_


def dbscan_split(track: Track, data: Data, clean: dict[str, np.ndarray] | None = None, eps: float = 0.55,
                 min_samples: int = 5, max_clusters: int = 3, min_len_s: float = 5) -> list[Track]:
    """GTA-style purity splitter: cluster the track's clean crops; if they form more than one identity, give every
    frame the cluster of its nearest (in time) clean crop and cut into time runs. Runs shorter than 1 s are absorbed
    into a neighbour, so crop-level noise cannot shred the track; pieces stay contiguous in time and are left to the
    linker to rejoin."""
    if len(track) < min_len_s * data.rate:
        return [track]
    ks, f = clean_feats(track, data, clean)
    lab = dbscan_labels(f, eps, min_samples)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    if len(ids) <= 1:
        return [track]
    keep = ids[np.argsort(-counts)[:max_clusters]]
    cent = normalise(np.stack([f[lab == q].mean(0) for q in keep]))
    lab = np.where(np.isin(lab, keep), lab, keep[np.argmax(f @ cent.T, axis=1)])   # noise -> nearest kept cluster
    all_k = _keys(track)
    pos = np.clip(np.searchsorted(ks, all_k), 0, len(ks) - 1)
    prev = np.clip(pos - 1, 0, len(ks) - 1)
    near = np.where(np.abs(ks[prev] - all_k) < np.abs(ks[pos] - all_k), prev, pos)
    frame_lab = lab[near]
    runs = []                                                   # [label, first index, last index] into all_k
    for q, l in enumerate(frame_lab):
        if runs and runs[-1][0] == l:
            runs[-1][2] = q
        else:
            runs.append([int(l), q, q])

    def span(r):
        return all_k[r[2]] - all_k[r[1]] + 1

    while len(runs) > 1:
        q = min(range(len(runs)), key=lambda r: span(runs[r]))
        if span(runs[q]) >= data.rate:
            break
        left, right = runs[q - 1] if q > 0 else None, runs[q + 1] if q + 1 < len(runs) else None
        if left is not None and right is not None and left[0] == right[0]:
            left[2] = right[2]
            del runs[q:q + 2]
            continue
        tgt = left if right is None or (left is not None and span(left) >= span(right)) else right
        tgt[1], tgt[2] = min(tgt[1], runs[q][1]), max(tgt[2], runs[q][2])
        del runs[q]
    return [_sub(track, all_k[r[1]:r[2] + 1]) for r in runs]


# ---------------------------------------------------------------------------------------------------------------
# B6. Gap filling and smoothing (after identities are fixed)
# ---------------------------------------------------------------------------------------------------------------

def _end_velocity(track: Track, ks: np.ndarray, q: int, rate: float, side: int, span: int = 5) -> np.ndarray | None:
    """Least-squares velocity at sample ks[q] from the samples within `span` frames on one side (-1 before, +1 after)."""
    k0 = ks[q]
    sel = ks[(ks <= k0) & (ks >= k0 - span)] if side < 0 else ks[(ks >= k0) & (ks <= k0 + span)]
    if len(sel) < 2:
        return None
    t = (sel - k0) / rate
    P = np.stack([track[int(k)].xy for k in sel])
    tc = t - t.mean()
    return (tc[:, None] * (P - P.mean(0))).sum(0) / (tc ** 2).sum()


def fill_gaps(track: Track, rate: float, max_gap_s: float = 3.0, vmax: float = 8.0) -> Track:
    """B6: fill gaps up to max_gap_s with a cubic Hermite curve through the two endpoints using their velocities
    (clamped to vmax; the chord velocity when an endpoint has no neighbours). Filled points have boxes=[], which
    is how metrics tell them apart. Longer gaps stay empty: identity known, position unknown."""
    out = dict(track)
    ks = _keys(track)
    for q in range(len(ks) - 1):
        a, b = int(ks[q]), int(ks[q + 1])
        if b - a <= 1 or b - a > max_gap_s * rate:
            continue
        T = (b - a) / rate
        p0, p1 = track[a].xy, track[b].xy
        chord = (p1 - p0) / T
        v0 = _end_velocity(track, ks, q, rate, -1)
        v1 = _end_velocity(track, ks, q + 1, rate, +1)
        v0 = chord if v0 is None else v0
        v1 = chord if v1 is None else v1
        v0, v1 = (v * min(1.0, vmax / max(np.linalg.norm(v), 1e-9)) for v in (v0, v1))
        for k in range(a + 1, b):
            s = (k - a) / (b - a)
            h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
            out[k] = TrackPoint(h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1, [])
    return dict(sorted(out.items()))


def smooth(track: Track, window: int = 7, order: int = 2) -> Track:
    """Savitzky-Golay smoothing of positions over each run of consecutive frames (runs shorter than the window are
    left as they are); boxes are kept."""
    from scipy.signal import savgol_filter
    ks = _keys(track)
    out = {}
    if not len(ks):
        return out
    cuts = np.flatnonzero(np.diff(ks) > 1) + 1
    for run in np.split(ks, cuts):
        P = np.stack([track[int(k)].xy for k in run])
        if len(run) >= window:
            P = savgol_filter(P, window, order, axis=0)
        for k, xy in zip(run, P):
            out[int(k)] = TrackPoint(np.asarray(xy, float), list(track[int(k)].boxes))
    return out
