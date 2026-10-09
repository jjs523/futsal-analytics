"""Identity building blocks shared by the tracklet linkers, on Tracks ({grid frame k: TrackPoint}).

Every TrackPoint keeps the (camera, box index) pairs it was built from, so these functions can always go back to
the boxes of `cams` (pipeline.boxes.CamBoxes): their size, confidence, team label, ReID vector and calibration.

    team_vote, split_team_flips               team (bib colour) per tracklet, split where it flips for good
    box_covariances, covariance, fuse,        anisotropic pitch covariance from the homography Jacobian:
      point_cov, mahalanobis2                 a far-side foot point is vague along the viewing ray only
    overlap, reachable, cannot_link           hard constraints for any linker
    clean_masks, clean_feats,                 robust ReID from crops that show one whole person
      track_embedding, app_distance
    dbscan_split                              purity splitter on clean-crop ReID
    fill_gaps, smooth                         cosmetic gap filling once identities are fixed

Teams are the generic labels of boxes.assign_teams: 'A' / 'B' per box ('' = unclear), and 'A' / 'B' / 'U'
(undecided) per track. Everything here is symmetric in A and B.

Per-box covariances and clean-crop masks are cached per CamBoxes object (and parameter set): they depend only on
the boxes and the calibration, which never change after boxes.build_cam_boxes (alignment moves `xy` only).
"""
from __future__ import annotations

import weakref
from typing import Callable

import numpy as np

from ..homography import calibration_at
from .boxes import CamBoxes, Track, TrackPoint

HOMOGRAPHY_FLOOR_M = 0.15        # absorbs homography / alignment error that pixel noise does not explain
NO_BOX_VAR = 0.5 ** 2            # variance (m^2) assumed for interpolated points that carry no boxes
EDGES = ("left", "top", "right", "bottom")

_CACHE: dict[tuple, tuple[weakref.ref, np.ndarray]] = {}


def _cached(cam: CamBoxes, key: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """compute() memoised on the CamBoxes object itself (not on id() alone: ids are reused after GC)."""
    ck = (id(cam),) + key
    hit = _CACHE.get(ck)
    if hit is not None and hit[0]() is cam:
        return hit[1]
    val = compute()
    try:
        _CACHE[ck] = (weakref.ref(cam, lambda _r, ck=ck: _CACHE.pop(ck, None)), val)
    except TypeError:                     # not weak-referenceable: just do not cache
        pass
    return val


def _keys(track: Track) -> np.ndarray:
    return np.array(sorted(track), int)


def _sub(track: Track, ks) -> Track:
    return {int(k): track[int(k)] for k in ks}


# ---------------------------------------------------------------------------------------------------------------
# Team vote
# ---------------------------------------------------------------------------------------------------------------

def _labelled(track: Track, cams: dict[str, CamBoxes], min_h: float, min_conf: float):
    """(k, weight, is_A) of every box of the track whose team colour is clear and that is big enough to trust."""
    out = []
    for k, p in track.items():
        for cam, i in p.boxes:
            c = cams[cam]
            if c.h[i] >= min_h and c.conf[i] >= min_conf and c.team[i]:
                out.append((k, float(c.h[i]), c.team[i] == "A"))
    return out


def team_vote(track: Track, cams: dict[str, CamBoxes], min_h: float = 40, min_conf: float = 0.4,
              camera_conflict: bool = True, min_cam_labels: int = 10) -> tuple[str, float]:
    """'A' / 'B' / 'U' and the box-height-weighted share of 'A' among the track's clearly labelled boxes.
    Big boxes count more because the shirt colour of a 40 px far-side player is unreliable. NaN share when nothing
    is labelled (the vote is then 'U').

    camera_conflict: when each of two cameras has >= `min_cam_labels` labelled boxes and they clearly disagree (one
    camera >= 0.7 'A', the other <= 0.3), the vote is 'U' whatever the weighted share says — the cameras render the
    shirt differently, and a confident wrong team would forbid the right identity (see tracklets.pair_views)."""
    lab = _labelled(track, cams, min_h, min_conf)
    if not lab:
        return "U", float("nan")
    w = np.array([x[1] for x in lab])
    a = np.array([x[2] for x in lab], float)
    share = float((w * a).sum() / w.sum())
    if camera_conflict:
        per_cam: dict[str, list[bool]] = {}
        for p in track.values():
            for cam, i in p.boxes:
                c = cams[cam]
                if c.h[i] >= min_h and c.conf[i] >= min_conf and c.team[i]:
                    per_cam.setdefault(cam, []).append(c.team[i] == "A")
        shares = [float(np.mean(v)) for v in per_cam.values() if len(v) >= min_cam_labels]
        if len(shares) >= 2 and max(shares) >= 0.7 and min(shares) <= 0.3:
            return "U", share
    return ("A" if share >= 0.7 else "B" if share <= 0.3 else "U"), share


def _sample_shares(track: Track, cams: dict[str, CamBoxes], min_h: float = 40,
                   min_conf: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample weighted 'A' share, only for samples with at least one labelled box."""
    acc: dict[int, list[float]] = {}
    for k, w, a in _labelled(track, cams, min_h, min_conf):
        s = acc.setdefault(k, [0.0, 0.0])
        s[0] += w * a
        s[1] += w
    ks = np.array(sorted(acc), int)
    return ks, np.array([acc[k][0] / acc[k][1] for k in ks])


def split_team_flips(tracks: list[Track], cams: dict[str, CamBoxes], run: int = 15, share: float = 0.8) -> list[Track]:
    """Cut a track where its team colour flips for good, i.e. the `run` labelled samples before the cut are
    >= `share` one team and the `run` after are >= `share` the other. One-off flips (a teammate's shirt occluding
    the player, a bad crop) never reach `share` over 1.5 s, so they do not cut. Every frame is kept."""
    out = []
    for tr in tracks:
        ks, sh = _sample_shares(tr, cams)
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
# Anisotropic pitch covariance
# ---------------------------------------------------------------------------------------------------------------

def _pitch_jacobian(cal, uv: np.ndarray) -> np.ndarray:
    """(n, 2, 2) d(pitch)/d(pixel) by central differences on the raw homography."""
    e = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cols = [(cal.to_pitch(uv + d) - cal.to_pitch(uv - d)) / 2.0 for d in e]
    return np.stack(cols, axis=2)


def _box_covariances(c: CamBoxes, sx: float, sy: float, min_px: float) -> np.ndarray:
    R = np.zeros((len(c.t), 2, 2))
    if not len(c.t):
        return R
    cals = [calibration_at(c.cal, float(t)) for t in c.t]
    groups: dict[int, list[int]] = {}
    for i, cal in enumerate(cals):
        groups.setdefault(id(cal), []).append(i)
    for idx in groups.values():
        idx = np.array(idx)
        J = _pitch_jacobian(cals[idx[0]], c.foot[idx])
        px = np.stack([np.maximum(sx * c.w[idx], min_px), np.maximum(sy * c.h[idx], min_px)], 1) ** 2
        R[idx] = np.einsum("nij,nj,nkj->nik", J, px, J)
    return R + HOMOGRAPHY_FLOOR_M ** 2 * np.eye(2)


def box_covariances(cams: dict[str, CamBoxes], cam: str, sx: float = 0.03, sy: float = 0.05,
                    min_px: float = 1.0) -> np.ndarray:
    """(n_boxes, 2, 2) pitch covariance (m^2) of every box of `cam`.
    Pixel noise of the foot point grows with the box (0.03 w across, 0.05 h along: feet are vaguer vertically),
    mapped through the local Jacobian, which stretches it along the viewing ray on the far side. The camera
    alignment is a small smooth correction, so the raw homography's Jacobian is used."""
    c = cams[cam]
    return _cached(c, ("cov", sx, sy, min_px), lambda: _box_covariances(c, sx, sy, min_px))


def covariance(cams: dict[str, CamBoxes], cam: str, idx: int) -> np.ndarray:
    """2x2 pitch covariance (m^2) of one box."""
    return box_covariances(cams, cam)[int(idx)]


def fuse(cams: dict[str, CamBoxes], boxes: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-covariance fusion of several boxes' (aligned) pitch positions: the camera that is precise along a
    direction dominates along that direction, instead of one scalar weight per camera."""
    if not boxes:
        raise ValueError("fuse() needs at least one box")
    P = np.zeros((2, 2))
    b = np.zeros(2)
    for cam, i in boxes:
        Ri = np.linalg.inv(covariance(cams, cam, i))
        P += Ri
        b += Ri @ cams[cam].xy[i]
    cov = np.linalg.inv(P)
    return cov @ b, cov


def point_cov(cams: dict[str, CamBoxes], p: TrackPoint) -> np.ndarray:
    """Covariance of a TrackPoint: fused from its boxes, or a generous isotropic default for interpolated points."""
    return fuse(cams, p.boxes)[1] if p.boxes else NO_BOX_VAR * np.eye(2)


def mahalanobis2(d: np.ndarray, R: np.ndarray) -> float:
    """Squared Mahalanobis length of offset d under covariance R (gate: < 9.21 = chi2(2) 99%)."""
    return float(d @ np.linalg.solve(R, d))


# ---------------------------------------------------------------------------------------------------------------
# Reachability and cannot-links
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


def _reach_ok(p1: TrackPoint, p2: TrackPoint, dt: float, cams: dict[str, CamBoxes], vmax: float, slack: float) -> bool:
    lam = float(np.linalg.eigvalsh(point_cov(cams, p1) + point_cov(cams, p2))[-1])
    return float(np.linalg.norm(p2.xy - p1.xy)) <= vmax * dt + slack + 3.0 * np.sqrt(lam)


def reachable(a: Track, b: Track, cams: dict[str, CamBoxes], rate: float, vmax: float = 8.0, slack: float = 1.0) -> bool:
    """||p_B - p_A|| <= vmax dt + slack + 3 sqrt(lambda_max(R_A + R_B)) at the hand-over from a to b (a ends before
    b starts; `rate` is the grid rate in Hz). If the two interleave, every hand-over on the merged timeline must
    pass, so the same test serves for joining multi-part tracks. Shared frames count as hand-overs with dt = 0
    whatever the cameras (a linker that fuses cross-camera shared frames needs its own test there)."""
    if not a or not b:
        return True
    return all(_reach_ok(p1, p2, (k2 - k1) / rate, cams, vmax, slack) for k1, p1, k2, p2 in _junctions(a, b))


def cannot_link(a: Track, b: Track, cams: dict[str, CamBoxes], rate: float, team_a: str | None = None,
                team_b: str | None = None, vmax: float = 8.0, slack: float = 1.0, max_overlap: int = 1) -> bool:
    """Hard constraint: True if a and b cannot be the same person (time overlap > max_overlap samples, A vs B
    teams, or an impossible run between them). 'U' is compatible with both teams."""
    if overlap(a, b) > max_overlap:
        return True
    ta = team_a if team_a is not None else team_vote(a, cams)[0]
    tb = team_b if team_b is not None else team_vote(b, cams)[0]
    if {ta, tb} == {"A", "B"}:
        return True
    return not reachable(a, b, cams, rate, vmax, slack)


# ---------------------------------------------------------------------------------------------------------------
# Clean crops and tracklet appearance
# ---------------------------------------------------------------------------------------------------------------

def _iou_matrix(b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def _clean_mask(c: CamBoxes, min_h: float, min_conf: float, max_iou: float, border: float, other_conf: float,
                exempt: tuple[str, ...]) -> np.ndarray:
    W, H = c.frame_wh
    m = {e: 0.0 if e in exempt else border * W for e in EDGES}      # border in units of the frame WIDTH on all edges
    x = c.xyxy
    ok = ((c.h >= min_h) & (c.conf >= min_conf) & (x[:, 0] >= m["left"]) & (x[:, 1] >= m["top"])
          & (x[:, 2] <= W - m["right"]) & (x[:, 3] <= H - m["bottom"]))
    for idx in c.by_k.values():
        if len(idx) < 2 or not ok[idx].any():
            continue
        iou = _iou_matrix(x[idx])
        np.fill_diagonal(iou, 0.0)
        iou[:, c.conf[idx] < other_conf] = 0.0
        ok[idx] &= iou.max(1) < max_iou
    return ok


def clean_masks(cams: dict[str, CamBoxes], min_h: float = 60, min_conf: float = 0.5, max_iou: float = 0.1,
                border: float = 0.02, other_conf: float = 0.25,
                border_exempt: dict[str, tuple[str, ...]] | None = None) -> dict[str, np.ndarray]:
    """Per camera, a bool per box that is True when the crop shows one whole person: big and confident enough,
    IoU < max_iou with every other box (conf >= other_conf) of the same camera and frame, and at least
    border * frame width away from every image edge. Only such crops give a ReID vector of one identity.

    border_exempt {camera: edges} drops the margin on some edges ('left', 'top', 'right', 'bottom'; the box must
    still lie inside the image). Pass the top edge of a camera aimed low: on the test match cam2 cuts 86% of the
    heads at the top edge, and the 2% rule left it with 11% clean crops instead of cam1's 76%, i.e. almost no
    identity evidence from that camera."""
    for cam, edges in (border_exempt or {}).items():
        bad = set(edges) - set(EDGES)
        if bad:
            raise ValueError(f"unknown image edges for {cam}: {sorted(bad)}")
    out = {}
    for cam, c in cams.items():
        exempt = tuple(sorted(set((border_exempt or {}).get(cam, ()))))
        key = ("clean", min_h, min_conf, max_iou, border, other_conf, exempt)
        out[cam] = _cached(c, key, lambda c=c, exempt=exempt: _clean_mask(c, min_h, min_conf, max_iou, border,
                                                                          other_conf, exempt))
    return out


def _reid_dim(cams: dict[str, CamBoxes]) -> int:
    return next((c.reid.shape[1] for c in cams.values() if c.reid is not None), 0)


def clean_feats(track: Track, cams: dict[str, CamBoxes],
                clean: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ks, feats): one ReID vector per frame that has a clean box, taken from the camera where the box is larger
    (a fused point usually has a near view and a far view; the near crop carries the identity). Cameras without
    ReID contribute nothing."""
    clean = clean_masks(cams) if clean is None else clean
    ks, rows = [], []
    for k in sorted(track):
        best = None
        for cam, i in track[k].boxes:
            if cams[cam].reid is not None and clean[cam][i] and (best is None or cams[cam].h[i] > cams[best[0]].h[best[1]]):
                best = (cam, i)
        if best is not None:
            ks.append(k)
            rows.append(cams[best[0]].reid[best[1]])
    if not rows:
        return np.zeros(0, int), np.zeros((0, _reid_dim(cams)), np.float32)
    return np.array(ks, int), np.stack(rows).astype(np.float32)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def medoid(feats: np.ndarray, max_n: int = 500) -> np.ndarray:
    """The feature with the highest summed cosine similarity to the others (on an evenly spaced subsample)."""
    f = feats[np.linspace(0, len(feats) - 1, min(max_n, len(feats))).astype(int)] if len(feats) > max_n else feats
    return f[int(np.argmax((f @ f.T).sum(1)))]


def track_embedding(track: Track, cams: dict[str, CamBoxes], clean: dict[str, np.ndarray] | None = None,
                    max_medoids: int = 10, min_crops: int = 3) -> dict | None:
    """Appearance of a track: L2-normalised mean of its clean crops plus up to `max_medoids` medoids, one per
    contiguous time chunk, so a pose / lighting change along the track is represented instead of averaged away.
    None when fewer than `min_crops` clean crops exist (far-side tracklets: link them by motion and team only)."""
    ks, f = clean_feats(track, cams, clean)
    if len(f) < min_crops:
        return None
    m = min(max_medoids, max(1, len(f) // min_crops))
    meds = np.stack([medoid(chunk) for chunk in np.array_split(f, m)])
    return {"mean": normalise(f.mean(0)), "medoids": meds, "n": len(f), "span": (int(ks[0]), int(ks[-1]))}


def app_distance(e1: dict | None, e2: dict | None) -> float:
    """min(1 - cos(means), 20th percentile of pairwise medoid cosine distances), in [0, 2]; 0.5 (uninformative)
    when either side has no clean crops. The percentile lets two tracks match on the poses they share."""
    if e1 is None or e2 is None:
        return 0.5
    d_mean = 1.0 - float(e1["mean"] @ e2["mean"])
    d_med = float(np.percentile(1.0 - e1["medoids"] @ e2["medoids"].T, 20))
    return float(np.clip(min(d_mean, d_med), 0.0, 2.0))


def dbscan_labels(feats: np.ndarray, eps: float = 0.55, min_samples: int = 5) -> np.ndarray:
    """DBSCAN (cosine) cluster label per crop; -1 is noise."""
    from sklearn.cluster import DBSCAN   # optional dependency, only for this splitter
    if len(feats) < min_samples:
        return np.full(len(feats), -1)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute").fit(feats).labels_


def dbscan_split(track: Track, cams: dict[str, CamBoxes], rate: float, clean: dict[str, np.ndarray] | None = None,
                 eps: float = 0.55, min_samples: int = 5, max_clusters: int = 3, min_len_s: float = 5) -> list[Track]:
    """GTA-style purity splitter: cluster the track's clean crops; if they form more than one identity, give every
    frame the cluster of its nearest (in time) clean crop and cut into time runs. Runs shorter than 1 s are absorbed
    into a neighbour, so crop-level noise cannot shred the track; pieces stay contiguous in time and are left to the
    linker to rejoin. Note: on OSNet x1.0 features the default eps 0.55 practically never splits (clean-crop
    distances are compressed: within a track 0.15-0.31, between tracks median 0.52); use eps 0.2-0.3 with
    min_samples 15 there."""
    if len(track) < min_len_s * rate:
        return [track]
    ks, f = clean_feats(track, cams, clean)
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
        if span(runs[q]) >= rate:
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
# Gap filling and smoothing (after identities are fixed)
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
    """Fill gaps up to max_gap_s with a cubic Hermite curve through the two endpoints using their velocities
    (clamped to vmax; the chord velocity when an endpoint has no neighbours). Filled points have boxes=[], which
    is how later stages tell them apart. Longer gaps stay empty: identity known, position unknown."""
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
