"""Tracklet sources: the input of the linkers. Goal: tracklets that are long AND pure.

The conservative per-camera builder (percam.py) is pure but short (~60% singletons from overlapped boxes); the boxmot
trackers are long but mix people. This module measures sources with a fast label-free summary and builds better ones.

    summary(tracklets, data) -> dict           fast source metrics (no O(T^2) loops), use instead of harness.metrics
    best_source(data, name) -> list[Track]     the registered sources, for the linkers

    python experiments/harness.py src_relink sources
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import numpy as np

from harness import Data, Track, TrackPoint, register
from blocks import clean_feats, clean_masks, dbscan_labels, medoid, team_vote

_PARTNERS: "weakref.WeakKeyDictionary[Data, dict[str, np.ndarray]]" = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------------------------------------------
# Fast source summary
# ---------------------------------------------------------------------------------------------------------------

def _team_wrong(t: Track, data: Data) -> tuple[int, int]:
    """(minority, total) clearly labelled boxes of a track: harness' team impurity numerator / denominator."""
    labs = [data.cams[cam].team[i] for p in t.values() for cam, i in p.boxes if data.cams[cam].team[i]]
    if not labs:
        return 0, 0
    y = sum(1 for x in labs if x == "Y")
    return min(y, len(labs) - y), len(labs)


def _app_breaks(t: Track, data: Data, chunk: int) -> int:
    """harness.metrics' appearance breaks of one track: consecutive 2 s chunk means of ReID (h >= 60, conf >= 0.5)
    with cosine similarity < 0.55."""
    chunks: dict[int, list[np.ndarray]] = {}
    for k, p in t.items():
        for cam, i in p.boxes:
            c = data.cams[cam]
            if c.h[i] >= 60 and c.conf[i] >= 0.5:
                chunks.setdefault(k // chunk, []).append(c.reid[i])
    keys = sorted(chunks)
    means = [np.mean(chunks[q], axis=0) for q in keys]
    means = [m / max(np.linalg.norm(m), 1e-9) for m in means]
    return sum(1 for q in range(1, len(means)) if keys[q] - keys[q - 1] <= 2 and float(means[q] @ means[q - 1]) < 0.55)


def _speed_jumps(t: Track, rate: float, win: int = 5, jump: float = 6.0) -> int:
    """metrics2.physics' speed jumps of one track (|v_after - v_before| > jump over `win` frames, runs count once)."""
    if len(t) < 2 * win + 1:
        return 0
    ks = np.array(sorted(t), int)
    lo = ks[0]
    P = np.full((ks[-1] - lo + 1, 2), np.nan)
    for k in ks:
        P[k - lo] = t[k].xy
    vb = (P[win:-win] - P[:-2 * win]) * rate / win
    va = (P[2 * win:] - P[win:-win]) * rate / win
    flag = np.linalg.norm(va - vb, axis=1) > jump                 # NaN (a missing end) compares False
    flag &= np.isfinite(P[win:-win, 0])
    return int((flag & ~np.r_[False, flag[:-1]]).sum())


def summary(tracks: list[Track], data: Data, part: str = "all", long_s: float = 2.0, eps: float = 0.3,
            min_samples: int = 15, ref: list[Track] | None = None) -> dict:
    """Label-free source metrics in O(points): size (tracklets, singletons, length-weighted mean length, share of
    points in tracklets >= long_s), purity proxies (team impurity overall and on tracklets >= long_s, appearance
    breaks / min, speed jumps / min, length-weighted ReID dispersion to the medoid, share of tracklets >= 5 s whose
    clean crops form > 1 DBSCAN cluster at the strict eps that phase 1 found able to detect splices) and bookkeeping
    (box coverage of conf >= 0.5 in-court boxes, double-used boxes, share of points carrying two cameras), and the
    swaps witnessed by the other camera (xref_swaps against `ref`, default the conservative per-camera tracklets).
    part = 'first' / 'second' clips to that half first."""
    from percam import per_camera
    from metrics2 import clip, part_range
    lo, hi = part_range(data, part)
    T = clip([t for t in tracks if t], lo, hi)
    rate = data.rate
    minutes = (hi - lo) / rate / 60.0
    lens = np.array([len(t) for t in T], float)
    pts = lens.sum()
    clean = clean_masks(data)
    wrong = total = wrong_l = total_l = breaks = jumps = 0
    disp, disp_w = [], []
    long5 = multi = 0
    used: dict[tuple[str, int], int] = {}
    two = 0
    for t, n in zip(T, lens):
        w, tot = _team_wrong(t, data)
        wrong += w; total += tot
        if n >= long_s * rate:
            wrong_l += w; total_l += tot
        breaks += _app_breaks(t, data, int(2 * rate))
        jumps += _speed_jumps(t, rate)
        for p in t.values():
            two += len(p.boxes) > 1
            for cb in p.boxes:
                used[cb] = used.get(cb, 0) + 1
        if n >= 2 * rate:
            _, f = clean_feats(t, data, clean)
            if len(f) >= 5:
                disp.append(float(np.mean(1.0 - f @ medoid(f)))); disp_w.append(n)
            if n >= 5 * rate and len(f) >= min_samples:
                long5 += 1
                lab = dbscan_labels(f, eps, min_samples)
                multi += len(np.unique(lab[lab >= 0])) > 1
    if ref is None:
        ref = [t for ts in per_camera(data, "conservative").values() for t in ts]
    good = sum(int(((c.conf >= 0.5) & c.in_court & (c.k >= lo) & (c.k < hi)).sum()) for c in data.cams.values())
    good_used = sum(1 for (cam, i) in used if data.cams[cam].conf[i] >= 0.5 and data.cams[cam].in_court[i])
    return {
        "tracklets": len(T),
        "singletons": int((lens == 1).sum()),
        "lw_mean_len_s": round(float((lens ** 2).sum() / max(pts, 1) / rate), 2),
        "share_ge2s": round(float(lens[lens >= long_s * rate].sum() / max(pts, 1)), 3),
        "share_ge10s": round(float(lens[lens >= 10 * rate].sum() / max(pts, 1)), 3),
        "team_impurity": round(wrong / max(total, 1), 4),
        "team_impurity_ge2s": round(wrong_l / max(total_l, 1), 4),
        "app_breaks_per_min": round(breaks / minutes, 2),
        "speed_jumps_per_min": round(jumps / minutes, 2),
        "reid_dispersion": round(float(np.average(disp, weights=disp_w)), 4) if disp else float("nan"),
        "multi_cluster_share": round(multi / max(long5, 1), 3),
        "box_coverage": round(good_used / max(good, 1), 3),
        "double_used_boxes": sum(1 for v in used.values() if v > 1),
        "fused_share": round(float(two / max(pts, 1)), 3),
        "xref_swaps_per_min": round(xref_swaps(T, data, ref)["xref_swaps"] / minutes, 2),
    }


# ---------------------------------------------------------------------------------------------------------------
# Cross-reference swap audit: the other camera as an independent witness
# ---------------------------------------------------------------------------------------------------------------

def xview_partners(data: Data, min_conf: float = 0.3, gate_m: float = 1.0, margin_m: float = 0.7) -> dict[str, np.ndarray]:
    """{cam: partner box index in the other camera, -1 if none} for boxes whose cross-view match is unambiguous in
    that frame: Hungarian on aligned pitch distance, d < gate_m, every other candidate in its row and column at least
    margin_m farther, bib colours not contradicting. Two phones at diagonal corners rarely see the same occlusion,
    so the other view tells who a box is independently of this view's tracking."""
    from scipy.optimize import linear_sum_assignment
    (ca, a), (cb, b) = data.cams.items()
    out = {ca: np.full(len(a.k), -1), cb: np.full(len(b.k), -1)}
    for k in range(data.n):
        ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k, min_conf)
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
        ta, tb = a.team[ia][:, None], b.team[ib][None]
        D = np.where((ta != "") & (tb != "") & (ta != tb), 1e3, D)
        r, c = linear_sum_assignment(D)
        for i, j in zip(r, c):
            d = D[i, j]
            if d >= gate_m:
                continue
            row, col = np.delete(D[i], j), np.delete(D[:, j], i)
            if (len(row) and row.min() < d + margin_m) or (len(col) and col.min() < d + margin_m):
                continue
            out[ca][ia[i]] = ib[j]; out[cb][ib[j]] = ia[i]
    return out


def _owner(tracks: list[Track], data: Data) -> dict[str, np.ndarray]:
    """{cam: track index owning each box, -1 if none}."""
    own = {cam: np.full(len(c.k), -1) for cam, c in data.cams.items()}
    for q, t in enumerate(tracks):
        for p in t.values():
            for cam, i in p.boxes:
                own[cam][i] = q
    return own


def xref_swaps(tracks: list[Track], data: Data, ref: list[Track], partners: dict[str, np.ndarray] | None = None,
               min_run: int = 2, lookahead: int = 20, far_m: float = 1.0) -> dict:
    """Identity swaps of `tracks` witnessed by the other camera's reference tracklets `ref` (pure, e.g. conservative).

    For every track and camera, map each of its boxes to the reference tracklet of its unambiguous cross-view
    partner. A change of reference X -> Y along the track is benign when X ends (the reference fragmented); it is a
    swap when X carries on within `lookahead` frames somewhere > far_m away from the track: the reference saw the
    person continue elsewhere while the track followed someone else. Reference errors add the same floor to every
    source scored against the same reference, so compare sources, not absolute levels."""
    if partners is None:
        if data not in _PARTNERS:
            _PARTNERS[data] = xview_partners(data)
        partners = _PARTNERS[data]
    own = _owner(ref, data)
    cams = list(data.cams)
    other = {cams[0]: cams[1], cams[1]: cams[0]}
    swaps = changes = 0
    for t in tracks:
        ks = sorted(t)
        for cam in cams:
            seq = []
            for k in ks:
                for c, i in t[k].boxes:
                    if c == cam and partners[cam][i] >= 0:
                        r = own[other[cam]][partners[cam][i]]
                        if r >= 0:
                            seq.append((k, int(r)))
            runs: list[list] = []                                    # [ref id, first k, last k, count]
            for k, r in seq:
                if runs and runs[-1][0] == r:
                    runs[-1][2] = k; runs[-1][3] += 1
                else:
                    runs.append([r, k, k, 1])
            runs = [x for x in runs if x[3] >= min_run]
            merged: list[list] = []
            for x in runs:
                if merged and merged[-1][0] == x[0]:
                    merged[-1][2] = x[2]
                else:
                    merged.append(list(x))
            for x, y in zip(merged, merged[1:]):
                changes += 1
                X = ref[x[0]]
                after = [k for k in range(y[1], y[1] + lookahead + 1) if k in X]
                if not after:
                    continue
                here = [k for k in after if k in t]
                if here and min(float(np.linalg.norm(X[k].xy - t[k].xy)) for k in here) > far_m:
                    swaps += 1
    return {"xref_changes": changes, "xref_swaps": swaps}


def junction_audit(tracks: list[Track], pieces: list[Track], data: Data, ref: list[Track],
                   partners: dict[str, np.ndarray] | None = None, n_end: int = 10, lookahead: int = 20,
                   far_m: float = 1.0, cases: list | None = None) -> dict:
    """Link precision of a linker that joined single-view `pieces` into `tracks`, judged by the other camera.

    At every junction (consecutive samples of a track owned by different pieces) take the reference tracklet of the
    unambiguous cross-view partners over the n_end samples before (mode X) and after (mode Y). X == Y confirms the
    link; X != Y with X still running > far_m away from the track within `lookahead` frames contradicts it; anything
    else (no witness, the reference fragmented) is unknown. Only meaningful when `ref` comes from the other camera
    than the pieces, e.g. the conservative tracklets. `cases` (a list) collects (track index, junction frame) of
    every contradicted junction, for visual checks."""
    if partners is None:
        if data not in _PARTNERS:
            _PARTNERS[data] = xview_partners(data)
        partners = _PARTNERS[data]
    piece_of = _owner(pieces, data)
    own = _owner(ref, data)
    cams = list(data.cams)
    other = {cams[0]: cams[1], cams[1]: cams[0]}
    out = {"junctions": 0, "confirmed": 0, "contradicted": 0, "unknown": 0}

    def witness(t: Track, ks: list[int]) -> int:
        ids = []
        for k in ks:
            for cam, i in t[k].boxes:
                j = partners[cam][i]
                if j >= 0 and own[other[cam]][j] >= 0:
                    ids.append(int(own[other[cam]][j]))
        return max(set(ids), key=ids.count) if ids else -1

    for ti, t in enumerate(tracks):
        ks = sorted(t)
        pid = [piece_of[t[k].boxes[0][0]][t[k].boxes[0][1]] if t[k].boxes else -1 for k in ks]
        for q in range(1, len(ks)):
            if pid[q] == pid[q - 1] or pid[q] < 0 or pid[q - 1] < 0:
                continue
            out["junctions"] += 1
            x, y = witness(t, ks[max(0, q - n_end):q]), witness(t, ks[q:q + n_end])
            if x < 0 or y < 0:
                out["unknown"] += 1
            elif x == y:
                out["confirmed"] += 1
            else:
                X = ref[x]
                here = [k for k in range(ks[q], ks[q] + lookahead + 1) if k in X and k in t]
                if here and min(float(np.linalg.norm(X[k].xy - t[k].xy)) for k in here) > far_m:
                    out["contradicted"] += 1
                    if cases is not None:
                        cases.append((ti, ks[q]))
                else:
                    out["unknown"] += 1
    out["precision"] = round(out["confirmed"] / max(out["confirmed"] + out["contradicted"], 1), 3)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Cut, then re-link locally
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _End:
    """One end (head or tail) of a track, as seen by the local re-linker."""
    k: int                                            # frame of the end sample
    xy: np.ndarray                                    # its pitch position
    cov: np.ndarray                                   # its B2 covariance
    img: dict = field(default_factory=dict)           # cam -> (frame, box xyxy, centre velocity px/frame pointing outwards)
    feat: np.ndarray | None = None                    # mean clean ReID next to this end
    team: str = "U"


@dataclass
class RelinkStats:
    rounds: int = 0
    candidates: int = 0
    linked: int = 0
    ambiguous: int = 0                                # a best candidate existed but failed the margin test
    by_round: list = field(default_factory=list)
    links: list = field(default_factory=list)         # (rule 'margin' / 'proposal', tail frame, head frame, cost)


def _lsq_velocity(ks: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares velocity (units per frame) of samples X at frames ks; zero with fewer than 2 samples."""
    if len(ks) < 2:
        return np.zeros(X.shape[1])
    t = ks - ks.mean()
    return (t[:, None] * (X - X.mean(0))).sum(0) / max(float((t ** 2).sum()), 1e-9)


def _end(track: Track, data: Data, clean: dict[str, np.ndarray], tail: bool, n_vel: int = 5, n_feat: int = 15) -> _End:
    """Tail (tail=True) or head of a track: last / first sample, outward image velocity per camera from the n_vel
    nearest samples, and the mean of the n_feat nearest clean crops (largest box per frame)."""
    from blocks import normalise, point_cov
    ks = np.array(sorted(track), int)
    order = ks[::-1] if tail else ks
    k0 = int(order[0])
    e = _End(k0, track[k0].xy, point_cov(data, track[k0]), team=team_vote(track, data)[0])
    for cam, c in data.cams.items():
        rows = [(int(k), i) for k in order[:3 * n_vel] for cc, i in track[int(k)].boxes if cc == cam][:n_vel]
        if not rows:
            continue
        kk = np.array([r[0] for r in rows], float)
        ctr = np.array([[(c.xyxy[i, 0] + c.xyxy[i, 2]) / 2, (c.xyxy[i, 1] + c.xyxy[i, 3]) / 2] for _, i in rows])
        v = _lsq_velocity(kk, ctr)
        e.img[cam] = (rows[0][0], c.xyxy[rows[0][1]], v if tail else -v)
    feats = []
    for k in order:
        best = None
        for cam, i in track[int(k)].boxes:
            if clean[cam][i] and (best is None or data.cams[cam].h[i] > data.cams[best[0]].h[best[1]]):
                best = (cam, i)
        if best is not None:
            feats.append(data.cams[best[0]].reid[best[1]])
            if len(feats) >= n_feat:
                break
    if feats:
        e.feat = normalise(np.mean(feats, 0))
    return e


def _link_cost(a: _End, b: _End, data: Data, vmax: float, slack: float, w_dh: float, w_app: float, app_missing: float,
               w_gap: float, img_gate: float, app_gate: float) -> float:
    """Cost of continuing tail a with head b (b starts after a ends); np.inf when a hard gate fails: Y vs N bibs, an
    impossible run on the pitch (B3), an image-space miss above img_gate box heights, or clearly different
    appearance. Cost = constant-velocity image miss in box heights, averaged over both directions, in the camera that
    sees the person largest + w_dh x log height change + w_app x ReID distance + w_gap per missing frame."""
    if {a.team, b.team} == {"Y", "N"}:
        return np.inf
    g = b.k - a.k
    dt = g / data.rate
    lam = float(np.linalg.eigvalsh(a.cov + b.cov)[-1])
    if float(np.linalg.norm(b.xy - a.xy)) > vmax * dt + slack + 3.0 * np.sqrt(lam):
        return np.inf
    best = None
    for cam in a.img.keys() & b.img.keys():
        ka, ba, va = a.img[cam]
        kb, bb, vb = b.img[cam]
        gi = kb - ka
        if gi <= 0:
            continue
        ca = np.array([(ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2])
        cb = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        ha, hb = ba[3] - ba[1], bb[3] - bb[1]
        di = 0.5 * (np.linalg.norm(ca + va * gi - cb) + np.linalg.norm(cb + vb * gi - ca)) / (0.5 * (ha + hb))
        if best is None or min(ha, hb) > best[0]:
            best = (min(ha, hb), di + w_dh * abs(np.log(hb / ha)), di)
    if best is None:                                     # no camera sees both ends: pitch distance in sigma-ish units
        dm = float(np.linalg.norm(b.xy - a.xy)) / (np.sqrt(lam) + 0.5 + 0.5 * vmax * dt)
        best = (0.0, dm, dm)
    if best[2] > img_gate:
        return np.inf
    if a.feat is not None and b.feat is not None:
        da = 1.0 - float(a.feat @ b.feat)
        if da > app_gate:
            return np.inf
    else:
        da = app_missing
    return best[1] + w_app * da + w_gap * (g - 1)


def _chain(tracks: list[Track], links: list[tuple[int, int]]) -> list[Track]:
    """Join tracks along one-to-one links a -> b (b after a); unlinked tracks pass through."""
    nxt = dict(links)
    has_prev = set(nxt.values())
    out = []
    for q in range(len(tracks)):
        if q in has_prev:
            continue
        t, r = dict(tracks[q]), q
        while r in nxt:
            r = nxt[r]
            t.update(tracks[r])
        out.append(t)
    return out


def _proposal_links(chains: list[Track], proposals: list[Track], data: Data, max_gap: int) -> set[tuple[int, int]]:
    """Pairs (a, b) where a proposal track goes from the last box of chain a straight to the first box of chain b
    within max_gap frames."""
    chain_of = _owner(chains, data)
    prop_of = _owner(proposals, data)
    out = set()
    for a, t in enumerate(chains):
        ka = max(t)
        if not t[ka].boxes:
            continue
        cam, i = t[ka].boxes[0]
        P = prop_of[cam][i]
        if P < 0:
            continue
        nxt = next((k for k in range(ka + 1, ka + max_gap + 1) if k in proposals[P]), None)
        if nxt is None or not proposals[P][nxt].boxes:
            continue
        cb, j = proposals[P][nxt].boxes[0]
        b = chain_of[cb][j]
        if b >= 0 and b != a and min(chains[b]) == nxt:
            out.add((a, b))
    return out


def relink(tracks: list[Track], data: Data, max_gap: int = 10, min_len: int = 3, vmax: float = 9.0, slack: float = 1.0,
           w_dh: float = 0.5, w_app: float = 1.0, app_missing: float = 0.15, w_gap: float = 0.01,
           img_gate: float = 1.0, app_gate: float = 0.35, max_cost: float = 0.6, margin: float = 0.15,
           proposals: list[Track] | None = None, prop_margin: float = 0.0, prop_min_len: int = 1, max_rounds: int = 5,
           clean: dict[str, np.ndarray] | None = None, stats: RelinkStats | None = None) -> list[Track]:
    """Re-attach the continuation of a track that a conservative builder cut at an occlusion, when it is unambiguous.

    Candidates: a head 1..max_gap frames after a tail, passing the gates of _link_cost. Two rules make links:
      margin rule    among tracks with >= min_len samples (shorter ones are neither linked nor competitors: they are
                     mostly the overlapped boxes of the occlusion itself), a -> b is the cheapest option of both a and
                     b by at least `margin` and costs <= max_cost;
      proposal rule  (with `proposals`, e.g. a boxmot tracker's output on the same boxes) the proposal tracker goes
                     from a's last box straight to b's first box, a -> b costs <= max_cost and is the mutual best
                     over all tracks with >= prop_min_len samples (short ones included) by at least prop_margin.
                     Two different association rules rarely make the same mistake, so their agreement can bridge
                     occlusions the margin rule cannot. prop_min_len=2 keeps it from chaining through 1-sample
                     overlapped boxes, where most of its witnessed swaps were (prop_min_len <= min_len expected).
    Linked chains get fresh ends (motion and ReID of the joined track) and the matching repeats until nothing
    changes. Linked tracks never overlap in time, so every box stays in exactly one track."""
    st = stats if stats is not None else RelinkStats()
    clean = clean_masks(data, border=0.0) if clean is None else clean
    chains = [dict(t) for t in tracks if t]
    tails: dict[int, _End] = {}
    heads: dict[int, _End] = {}
    for _ in range(max_rounds):
        st.rounds += 1
        longq = {q for q, t in enumerate(chains) if len(t) >= min_len}
        idx = [q for q, t in enumerate(chains) if len(t) >= prop_min_len] if proposals is not None else sorted(longq)
        for q in idx:
            if q not in tails:
                tails[q], heads[q] = _end(chains[q], data, clean, True), _end(chains[q], data, clean, False)
        by_start: dict[int, list[int]] = {}
        for q in idx:
            by_start.setdefault(heads[q].k, []).append(q)
        cost: dict[tuple[int, int], float] = {}
        for a in idx:
            for g in range(1, max_gap + 1):
                for b in by_start.get(tails[a].k + g, []):
                    c = _link_cost(tails[a], heads[b], data, vmax, slack, w_dh, w_app, app_missing, w_gap,
                                   img_gate, app_gate)
                    if np.isfinite(c):
                        cost[(a, b)] = c
        st.candidates += len(cost)

        def mutual_best(pairs: dict[tuple[int, int], float], marg: float, only: set | None = None) -> list[tuple[int, int]]:
            row: dict[int, list[tuple[float, int]]] = {}
            col: dict[int, list[tuple[float, int]]] = {}
            for (a, b), c in pairs.items():
                row.setdefault(a, []).append((c, b))
                col.setdefault(b, []).append((c, a))
            out = []
            for a, opts in row.items():
                c, b = min(opts)
                if c > max_cost or min(col[b])[1] != a or (only is not None and (a, b) not in only):
                    continue
                others = [x for x, q in opts if q != b] + [x for x, q in col[b] if q != a]
                if min(others, default=np.inf) - c < marg:
                    st.ambiguous += 1
                    continue
                out.append((a, b))
            return out

        links = []
        if proposals is not None:
            links = mutual_best(cost, prop_margin, _proposal_links(chains, proposals, data, max_gap))
        rule = {p: "proposal" for p in links}
        has_next, has_prev = {a for a, _ in links}, {b for _, b in links}
        for a, b in mutual_best({p: c for p, c in cost.items() if p[0] in longq and p[1] in longq}, margin):
            if a not in has_next and b not in has_prev:
                links.append((a, b))
                rule[(a, b)] = "margin"
        st.links += [(rule[p], tails[p[0]].k, heads[p[1]].k, round(cost[p], 3)) for p in links]
        st.by_round.append(len(links))
        if not links:
            break
        st.linked += len(links)
        # _chain emits one chain per track that is not a link target, in order; only chains that took part in no
        # link keep their cached ends
        touched = {q for p in links for q in p}
        targets = {b for _, b in links}
        firsts = [q for q in range(len(chains)) if q not in targets]
        tails = {n: tails[q] for n, q in enumerate(firsts) if q not in touched and q in tails}
        heads = {n: heads[q] for n, q in enumerate(firsts) if q not in touched and q in heads}
        chains = _chain(chains, links)
    return sorted(chains, key=lambda t: min(t))


def absorb_short(tracks: list[Track], data: Data, max_len: int = 2, host_min_len: int = 10, gate_m: float = 1.0,
                 margin_m: float = 0.7, max_hole: int = 3) -> list[Track]:
    """Give the boxes of very short tracklets (<= max_len samples, mostly the overlapped boxes of an occlusion) to a
    long track (>= host_min_len) when that is unambiguous at every one of their samples: the host either has a point
    at that frame without a box of this camera (the box becomes its second view) or a hole of <= max_hole frames there
    (the box fills it, compared with the linear interpolation), the box lies within gate_m of the host, every other
    long track is at least margin_m farther, and the bib colours do not contradict. A short tracklet moves whole or not
    at all, so a box is never split from its tracklet's other samples by guesswork."""
    from blocks import fuse
    hosts = [q for q, t in enumerate(tracks) if len(t) >= host_min_len]
    team = {q: team_vote(tracks[q], data)[0] for q in hosts}
    out = [dict(t) for t in tracks]
    keys = {q: np.array(sorted(tracks[q]), int) for q in hosts}
    at: dict[int, list[int]] = {}
    for q in hosts:
        ks = keys[q]
        for k in range(int(ks[0]), int(ks[-1]) + 1):
            at.setdefault(k, []).append(q)

    def host_xy(q: int, k: int) -> tuple[np.ndarray | None, bool]:
        """(position of host q at frame k, has a point there); None when k is in a hole longer than max_hole."""
        t = tracks[q]
        if k in t:
            return t[k].xy, True
        ks = keys[q]
        j = int(np.searchsorted(ks, k))
        a, b = int(ks[j - 1]), int(ks[j])
        if b - a - 1 > max_hole:
            return None, False
        w = (k - a) / (b - a)
        return (1 - w) * t[a].xy + w * t[b].xy, False

    moved = set()
    for s, t in enumerate(tracks):
        if len(t) > max_len or s in moved:
            continue
        plan = []
        for k, p in sorted(t.items()):
            if len(p.boxes) != 1:
                plan = None
                break
            cam, i = p.boxes[0]
            tb = data.cams[cam].team[i]
            cands = []
            for q in at.get(k, []):
                xy, has = host_xy(q, k)
                if xy is None:
                    continue
                if k in out[q] and any(c == cam for c, _ in out[q][k].boxes):
                    d_ok = False                         # the host already has a box of this camera here
                else:
                    d_ok = not (tb and team[q] in ("Y", "N") and tb != team[q])
                cands.append((float(np.linalg.norm(xy - p.xy)), q, d_ok))
            cands.sort()
            if not cands or not cands[0][2] or cands[0][0] > gate_m or (len(cands) > 1 and cands[1][0] - cands[0][0] < margin_m):
                plan = None
                break
            plan.append((k, cands[0][1], (cam, i)))
        if not plan or len({q for _, q, _ in plan}) != 1:
            continue
        q = plan[0][1]
        for k, _, cb in plan:
            boxes = sorted((out[q][k].boxes if k in out[q] else []) + [cb])
            out[q][k] = TrackPoint(fuse(data, boxes)[0], boxes)
        out[s] = {}
        moved.add(s)
    return sorted((dict(sorted(t.items())) for t in out if t), key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# Pitch-plane source: Hungarian cross-view fusion per frame, then Hungarian frame-to-frame tracklets with ReID
# ---------------------------------------------------------------------------------------------------------------

Detection = tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]      # (xy, covariance, boxes)


def fused_detections(data: Data, min_conf: float = 0.3, gate_d2: float = 9.21, max_m: float = 1.5) -> list[list[Detection]]:
    """Per grid frame, [(xy, cov, boxes)]: cam1 / cam2 boxes paired by a Hungarian solve on the B2 Mahalanobis
    distance (d2 < gate_d2 and < max_m metres, bibs not contradicting) and fused by inverse covariance; unpaired
    boxes pass through as single views. Replaces fusion.fuse_frame's greedy isotropic pairing."""
    from scipy.optimize import linear_sum_assignment
    from blocks import box_covariances, fuse
    (ca, a), (cb, b) = data.cams.items()
    Ra, Rb = box_covariances(data, ca), box_covariances(data, cb)
    frames = []
    for k in range(data.n):
        ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k, min_conf)
        pairs = []
        if len(ia) and len(ib):
            d = a.xy[ia][:, None] - b.xy[ib][None]
            S = Ra[ia][:, None] + Rb[ib][None]
            D2 = np.einsum("ijk,ijk->ij", d, np.linalg.solve(S, d[..., None])[..., 0])
            ok = (D2 < gate_d2) & (np.linalg.norm(d, axis=2) < max_m)
            ta, tb = a.team[ia][:, None], b.team[ib][None]
            ok &= ~((ta != "") & (tb != "") & (ta != tb))
            r, c = linear_sum_assignment(np.where(ok, D2, 1e6))
            pairs = [(i, j) for i, j in zip(r, c) if ok[i, j]]
        out: list[Detection] = []
        for i, j in pairs:
            boxes = [(ca, int(ia[i])), (cb, int(ib[j]))]
            xy, cov = fuse(data, boxes)
            out.append((xy, cov, boxes))
        pa, pb = {i for i, _ in pairs}, {j for _, j in pairs}
        out += [(a.xy[ia[i]], Ra[ia[i]], [(ca, int(ia[i]))]) for i in range(len(ia)) if i not in pa]
        out += [(b.xy[ib[j]], Rb[ib[j]], [(cb, int(ib[j]))]) for j in range(len(ib)) if j not in pb]
        frames.append(out)
    return frames


def pitch_tracklets(data: Data, frames: list[list[Detection]] | None = None, accel: float = 4.0, gate_d2: float = 9.21,
                    w_app: float = 10.0, app_gate: float = 0.35, margin: float = 3.0, max_gap: int = 5,
                    vel_damp: float = 0.7, clean: dict[str, np.ndarray] | None = None) -> list[Track]:
    """Frame-to-frame tracklets on the pitch over fused_detections: Hungarian on cost = Mahalanobis d2 of the
    detection from the damped constant-velocity prediction (covariance grown by (accel dt^2 + 0.1 m)^2) + w_app x ReID
    distance (clean crop of the largest box against the track EMA; skipped when either is missing). Like the
    conservative builder it cuts instead of guessing: a match whose row or column runner-up is within `margin` ends
    the track and starts a new one; > max_gap missing frames end it too."""
    from scipy.optimize import linear_sum_assignment
    from blocks import normalise
    frames = fused_detections(data) if frames is None else frames
    clean = clean_masks(data, border=0.0) if clean is None else clean
    dt = 1.0 / data.rate
    live: list[dict] = []
    done: list[dict] = []

    def feat(boxes: list[tuple[str, int]]) -> np.ndarray | None:
        best = None
        for cam, i in boxes:
            if clean[cam][i] and (best is None or data.cams[cam].h[i] > data.cams[best[0]].h[best[1]]):
                best = (cam, i)
        return data.cams[best[0]].reid[best[1]].astype(np.float64) if best else None

    def start(k: int, det: Detection) -> dict:
        return {"pts": {k: det}, "last": k, "xy": det[0], "cov": det[1], "v": np.zeros(2), "emb": feat(det[2])}

    for k in range(data.n):
        dets = frames[k]
        keep = []
        for t in live:
            (done if k - t["last"] - 1 > max_gap else keep).append(t)
        live = keep
        if not live or not dets:
            live = live + [start(k, d) for d in dets]
            continue
        C = np.full((len(live), len(dets)), np.inf)
        fe = [feat(d[2]) for d in dets]
        for ti, t in enumerate(live):
            g = k - t["last"]
            pred = t["xy"] + vel_damp * t["v"] * g * dt
            P = t["cov"] + (accel * (g * dt) ** 2 + 0.1) ** 2 * np.eye(2)
            emb = normalise(t["emb"]) if t["emb"] is not None else None
            for di, d in enumerate(dets):
                r = d[0] - pred
                d2 = float(r @ np.linalg.solve(P + d[1], r))
                if d2 > gate_d2:
                    continue
                if emb is not None and fe[di] is not None:
                    da = 1.0 - float(emb @ fe[di])
                    if da > app_gate:
                        continue
                    d2 += w_app * da
                C[ti, di] = d2
        fin = np.isfinite(C)
        r, cidx = linear_sum_assignment(np.where(fin, C, 1e9))
        nxt, used_t, used_d = [], set(), set()
        for ti, di in zip(r, cidx):
            if not fin[ti, di]:
                continue
            used_t.add(ti); used_d.add(di)
            t, d = live[ti], dets[di]
            second = min(np.delete(C[ti], di).min(initial=np.inf), np.delete(C[:, di], ti).min(initial=np.inf))
            if second - C[ti, di] < margin:
                done.append(t)
                nxt.append(start(k, d))
                continue
            g = k - t["last"]
            t["v"] = 0.5 * t["v"] + 0.5 * (d[0] - t["xy"]) / (g * dt)
            t["xy"], t["cov"], t["last"] = d[0], d[1], k
            t["pts"][k] = d
            if fe[di] is not None:
                t["emb"] = fe[di] if t["emb"] is None else 0.9 * t["emb"] + 0.1 * fe[di]
            nxt.append(t)
        nxt += [t for ti, t in enumerate(live) if ti not in used_t]
        nxt += [start(k, d) for di, d in enumerate(dets) if di not in used_d]
        live = nxt
    done += live
    out = [{k: TrackPoint(np.asarray(d[0], float), list(d[2])) for k, d in sorted(t["pts"].items())} for t in done]
    return sorted(out, key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# Registered sources
# ---------------------------------------------------------------------------------------------------------------

_SOURCES: "weakref.WeakKeyDictionary[Data, dict]" = weakref.WeakKeyDictionary()

PROPOSER = "boosttrack"          # purest boxmot tracker here (team impurity 0.019 vs 0.04-0.11 for the others)


def percam_source(data: Data, proposals: bool = True) -> dict[str, list[Track]]:
    """{cam: single-view tracklets}: conservative builder, then the local re-linker (margin rule; with `proposals`
    also the BoostTrack-agreement rule, never through 1-sample pieces). Every point holds one box."""
    from percam import per_camera
    key = ("percam", proposals)
    cache = _SOURCES.setdefault(data, {})
    if key not in cache:
        base = per_camera(data, "conservative")
        prop = per_camera(data, PROPOSER) if proposals else {}
        cache[key] = {cam: relink(base[cam], data, proposals=prop.get(cam), prop_min_len=2) for cam in data.cams}
    return cache[key]


def best_source(data: Data, name: str = "relink") -> list[Track]:
    """Fused tracklet source for the linkers (memoised per Data and name):
      'relink'       percam_source(proposals=True) -> xview.pair_views (cut only where fusion would collide) ->
                     fused re-link (min_len 5) -> absorb_short. Longest of the pure sources.
      'relink_pure'  the same without BoostTrack proposals (margin rule only): ~40% fewer links, fewest witnessed
                     swaps.
      'relink_cut'   as 'relink' but pair_views cuts wherever a tracklet's cross-view partner changes (plan V3), so
                     every tracklet has at most one partner per camera; shorter.
      'pitch'        pitch_tracklets + re-link + absorb_short: fused-first alternative (Hungarian fusion, ReID in the
                     frame-to-frame cost); pure but much shorter than the per-camera route.
    Points carry their boxes; no box is used twice."""
    from xview import pair_views
    cache = _SOURCES.setdefault(data, {})
    if name not in cache:
        if name == "pitch":
            t = relink(pitch_tracklets(data, margin=1.0), data, min_len=3)
        elif name in ("relink", "relink_pure", "relink_cut"):
            per = percam_source(data, proposals=name != "relink_pure")
            c1, c2 = list(data.cams)
            t = pair_views(per[c1], per[c2], data, cut_on_partner_change=name == "relink_cut")
            t = relink(t, data, min_len=5) if name != "relink_cut" else t
        else:
            raise ValueError(f"unknown source {name!r}")
        cache[name] = absorb_short(t, data)
    return [dict(t) for t in cache[name]]


for _name in ("relink", "relink_pure", "relink_cut", "pitch"):
    register(f"src_{_name}")(lambda data, _n=_name: best_source(data, _n))
