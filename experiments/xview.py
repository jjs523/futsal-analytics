"""Cross-view pairing of per-camera tracklets (research_plan.md V3 step 2).

A cam1 tracklet and a cam2 tracklet are the same person when their pitch positions agree over many shared samples.
Deciding this per tracklet pair (evidence averaged over >= 1 s) instead of per frame means one bad frame cannot fuse
two players. The output fused tracklets carry both boxes where both views exist; unpaired parts stay single-view.

    python experiments/harness.py xv_conservative xview
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from harness import Data, Track, TrackPoint, register
from percam import per_camera


@dataclass
class XviewStats:
    candidates: int = 0             # pairs sharing >= min_shared samples
    accepted: int = 0               # ... passing the distance / inside / team gates
    rejected_team: int = 0
    windows: int = 0                # distinct sets of simultaneously active accepted pairs
    paired_samples: int = 0         # (cam1, cam2) sample pairs fused
    ambiguous_samples: int = 0      # left unpaired because a swapped assignment was within tie_m
    cuts: int = 0                   # tracklet splits at partner changes
    conflict_tracklets: int = 0     # (cut_on_partner_change=False) tracklets that had to be cut anyway
    medians: list = field(default_factory=list)


def _team(data: Data, boxes: list[tuple[str, int]], min_labels: int = 3, share: float = 0.7) -> str:
    """Majority bib colour of a tracklet: 'Y' / 'N' when >= share of its labelled boxes agree, else ''."""
    labs = [data.cams[cam].team[i] for cam, i in boxes if data.cams[cam].team[i]]
    if len(labs) < min_labels:
        return ""
    y = sum(1 for x in labs if x == "Y") / len(labs)
    return "Y" if y >= share else "N" if y <= 1 - share else ""


def _single(p: TrackPoint) -> tuple[str, int]:
    if len(p.boxes) != 1:
        raise ValueError("pair_views expects single-view tracklets (one box per TrackPoint)")
    return p.boxes[0]


def _solve(cost: dict[tuple, float], unpaired: float, forbid: tuple | None = None) -> tuple[dict[tuple, float], float]:
    """Min-cost partial matching: every tracklet may stay unpaired at cost `unpaired` (>= any valid pair cost, so a
    valid pair always beats leaving both sides alone); `forbid` removes one pair but keeps its tracklets in the
    problem, so totals stay comparable. Returns ({(a, b): cost}, total cost)."""
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


def pair_views(t1: list[Track], t2: list[Track], data: Data, min_shared: int = 10, gate_m: float = 1.0,
               min_inside: float = 0.7, cut_on_partner_change: bool = True, tie_m: float = 0.1,
               sigma_gate: float = 0.0, stats: XviewStats | None = None) -> list[Track]:
    """Fuse cam1 tracklets t1 with cam2 tracklets t2.

    Candidates share >= min_shared samples; cost = median pitch distance over the shared samples. A pair is accepted
    if the median is <= gate (gate_m, or sigma_gate x the median combined foot-point sigma if larger), >= min_inside
    of the shared samples lie within 2 x gate, and the majority bib colours do not contradict. Then, for every window
    of frames with the same set of active accepted pairs, a min-cost partial matching picks at most one partner per
    tracklet (mutual best); an assigned pair is dropped for that window if forbidding it costs < tie_m more in total
    (a near-equal swap, typical of two players on the camera diagonal). A tracklet whose partner changes is split there
    (cut_on_partner_change), so each output tracklet has at most one partner per camera. Fused points carry both boxes
    and the inverse-variance (sigma) mean position. Every input box ends up in exactly one output tracklet."""
    st = stats if stats is not None else XviewStats()
    tr = {("1", i): t for i, t in enumerate(t1)} | {("2", i): t for i, t in enumerate(t2)}
    box = {key: {k: _single(p) for k, p in t.items()} for key, t in tr.items()}
    team = {key: _team(data, list(b.values())) for key, b in box.items()}
    xy = lambda cb: data.cams[cb[0]].xy[cb[1]]
    sig = lambda cb: float(data.cams[cb[0]].sigma[cb[1]])

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
    parent = list(range(len(seg_keys)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in (key for key in tr if key[0] == "1"):
        for k, b in partner[a].items():
            parent[root(seg_of[a][k])] = root(seg_of[b][k])
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


def check_partition(inputs: list[Track], outputs: list[Track]) -> dict:
    """Every input box in exactly one output point at the same frame; at most one box per camera per point."""
    want = {(k, cb) for t in inputs for k, p in t.items() for cb in p.boxes}
    got = [(k, cb) for t in outputs for k, p in t.items() for cb in p.boxes]
    multi = sum(1 for t in outputs for p in t.values() if len({c for c, _ in p.boxes}) != len(p.boxes))
    return {"input_boxes": len(want), "output_boxes": len(got), "missing": len(want - set(got)),
            "extra": len(set(got) - want), "duplicated": len(got) - len(set(got)), "multi_box_same_cam": multi,
            "fused_points": sum(1 for t in outputs for p in t.values() if len(p.boxes) == 2)}


def xview(data: Data, source: str = "conservative", percam_params: dict | None = None, **params) -> list[Track]:
    per = per_camera(data, source, **(percam_params or {}))
    c1, c2 = list(data.cams)
    return pair_views(per[c1], per[c2], data, **params)


@register("xv_conservative")
def xv_conservative(data: Data) -> list[Track]:
    return xview(data, "conservative")


@register("xv_conservative_nocut")
def xv_conservative_nocut(data: Data) -> list[Track]:
    return xview(data, "conservative", cut_on_partner_change=False)


for _name in ("hybridsort", "deepocsort", "botsort"):
    register(f"xv_{_name}")(lambda data, _n=_name: xview(data, _n))
