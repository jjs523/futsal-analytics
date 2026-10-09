"""Closed-set identity assignment: tracklets -> one identity per player, using the known number of players.

Futsal is 5 v 5, so instead of growing identities until a linker runs out of evidence, every tracklet either gets
one of K identity slots per team (5 regular + 1 'spare' for substitutes and noise) or is left out. The solve is one
MILP (scipy.optimize.milp / HiGHS) over binaries x[t, s] (tracklet t -> slot s):

    sum_s x[t, s] <= 1                                    every tracklet at most once
    sum_{t in C} x[t, s] <= 1    for each maximal clique C of time-overlapping tracklets   (no overlap in a slot)
    sum_{t in C} sum_{s in team} x[t, s] <= on_court     at most 5 per team on the pitch at once (spare = sub)
    x[t, s] + x[u, s] <= 1       for every non-overlapping pair that fails identity.reachable, and for every
                                 hand-over within link_gap_s whose constant-velocity residual exceeds cv_gate sigma
    sum_s x[a, s] + x[b, s] <= 1 for duplicate pairs (same spot, same time, same team)

Overlap cliques of an interval graph are a consecutive-ones matrix, so without the reachability rows the LP is
already integral; the reachability rows are pairwise, but only pairs closer in time than the pitch diagonal at vmax
can fail, and a slot-mate between two unreachable tracklets would itself be unreachable from one of them, so the
pairwise rows are (up to that triangle argument) the exact successor-chain condition.

Objective: sum_t,s x[t, s] * len_t * (d_app(t, proto_s) - coverage)  [+ U-team and spare penalties]
           - sum of motion-continuity rewards z[t, u, s] for short, smooth hand-overs t -> u kept in one slot.
Appearance alone cannot hold a far-side player (no clean crops) in his slot; the link rewards do.

Before the solve, tracklets are cut where they fail reachability internally (id switches of the source tracker)
and where the bib flips, and duplicates seen by complementary cameras (missed cross-view pairs) are merged: in a
closed set a duplicate would take a second slot and push the real fifth player out. Prototypes come from the
longest window where 5 same-team tracklets with clean crops coexist (5 distinct people, no label symmetry), then
EM: re-estimate each slot's prototype from what it was given, re-solve. Tracklets shorter than `min_len_s` stay out
of the solve and are attached afterwards to the slot whose neighbouring points they reach best; finally the
multi-box points are re-fused with the anisotropic covariances, gaps <= fill_gap_s filled (never above 5 per team)
and positions Savitzky-Golay smoothed.

Ported from experiments/link_closed.py (candidate 'final_d' of experiments/combine.py: tracklets.build_tracklets
-> this, with the tuned coverage 0.7 / attach_gap_s 2.0 and the top-edge clean-crop exemption). Blind visual audit
on the 2026-10-08 test match: identity purity 0.981 (baseline 0.888), same-team crossing swaps 30 % (55 %).

    assign_identities           one solve over the whole input (fine up to ~10 minutes)
    assign_identities_windowed  a full match: overlapping windows solved independently, then chained

Needs scipy >= 1.9 (scipy.optimize.milp, a core dependency); ClosedParams.dbscan_eps (off by default) also needs
scikit-learn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..court import Court
from .boxes import CamBoxes, Track, TrackPoint
from .identity import (app_distance, clean_masks, dbscan_split, fill_gaps, fuse, normalise, point_cov, reachable,
                       smooth as savgol_smooth, split_team_flips, team_vote, track_embedding)

TEAMS = ("A", "B")


@dataclass(frozen=True)
class ClosedParams:
    slots: int = 5                    # regular slots per team
    spare: int = 1                    # extra high-cost slots per team (substitutes, noise)
    on_court: int = 5                 # at most this many slots of one team active in any frame
    min_len_s: float = 1.0            # shorter tracklets are attached after the solve
    coverage: float = 0.7             # per-sample reward for assigning a tracklet (0.6-0.8 gave identical results)
    d_none: float = 0.35              # appearance cost of a tracklet without clean crops
    app_w: float = 1.0                # weight of (d_app - d_none): 0 = motion / team / coverage only
    u_pen: float = 0.15               # per-sample penalty for a 'U' tracklet in either team
    spare_pen: float = 0.2            # per-sample penalty for a spare slot
    vmax: float = 8.0                 # reachability gate, m/s
    slack: float = 1.0                # ... plus this many metres
    link_w: float = 10.0              # max reward (in sample-cost units) for a smooth hand-over kept in one slot
    link_gap_s: float = 3.0           # hand-overs considered for the reward and the cv gate
    cv_gate: float | None = 4.0       # forbid hand-overs within link_gap_s whose CV residual exceeds this many sigma
    link_s0: float = 0.7              # motion residual scale: s0 m + s1 m/s * dt
    link_s1: float = 1.5
    proto_medoids: int = 30           # medoids kept per slot prototype
    rounds: int = 4                   # 1 initial solve + EM refinements
    pin_seeds: bool = True            # keep the initial 5 tracklets in their slots (anchors the labels)
    team_split: bool = True           # cut at lasting bib flips before assignment
    jump_split: bool = True           # cut tracklets at steps that fail reachability (source-tracker id switch)
    dbscan_eps: float | None = None   # purity split (identity.dbscan_split) before assignment, e.g. 0.25
    dup_m: float = 0.8                # duplicates: overlapping, team-compatible, median distance below this ...
    dup_inside: float = 0.8           # ... and this share of shared frames within 1.5 m
    dup_cover: float = 0.0            # ... over at least this share of the shorter tracklet (merging)
    excl_cover: float = 0.5           # same, for exclusion (drops a whole tracklet, so ask for more evidence)
    merge_dups: bool = True           # merge duplicates seen by complementary cameras (missed cross-view pairs)
    exclude_dups: bool = True         # other duplicates: at most one of the pair gets a slot
    use_team: bool = True             # False: one 12-slot pool without team constraint (ablation)
    attach_gap_s: float = 2.0         # short tracklets: at least one side within this gap of the slot's points
    fill_gap_s: float = 3.0
    refuse: bool = True               # re-fuse multi-box points with the anisotropic covariances
    smooth: bool = True               # Savitzky-Golay (7, 2) after filling
    time_limit: float = 60.0          # per MILP solve, s


@dataclass
class ClosedStats:
    tracklets: int = 0
    solved: int = 0                   # tracklets entering the MILP
    assigned: int = 0
    assigned_samples_share: float = 0.0
    attached_short: int = 0
    short_total: int = 0
    unassigned_long: int = 0
    unassigned_long_samples: int = 0
    merged_dups: int = 0
    unpinned: bool = False
    exclusive_pairs: int = 0
    seeds: dict = field(default_factory=dict)
    solver_s: list = field(default_factory=list)
    n_vars: int = 0
    n_rows: int = 0
    spare_samples: dict = field(default_factory=dict)
    spare_tracklets: dict = field(default_factory=dict)
    exact_5v5_slots: float = 0.0      # share of frames (input span) with exactly 5 A and 5 B slots active, unfilled
    exact_5v5_filled: float = 0.0     # ... after gap filling
    reach_violations: int = 0         # consecutive slot-mates failing reachability after attachment (should be 0)
    changes: list = field(default_factory=list)   # tracklets that changed slot per EM round
    unreachable_pairs: int = 0        # reachability exclusion rows (pairs)
    rough_pairs: int = 0              # CV-gate exclusion rows (pairs)
    links: int = 0                    # link-reward variables
    slot_team: list = field(default_factory=list)
    is_spare: list = field(default_factory=list)
    junctions: list = field(default_factory=list)  # (output index, last frame, first frame) of in-slot hand-overs
    output_team: list = field(default_factory=list)   # 'A' / 'B' per output track (its slot's team)


@dataclass
class _Tl:
    """A tracklet with the summary quantities the solve needs."""
    tr: Track
    ks: np.ndarray
    team: str
    emb: dict | None

    @property
    def start(self) -> int:
        return int(self.ks[0])

    @property
    def end(self) -> int:
        return int(self.ks[-1])

    def __len__(self) -> int:
        return len(self.ks)


@dataclass
class _Ctx:
    """What every step needs besides the tracklets: the boxes, the grid rate, the pitch and the frame span."""
    cams: dict[str, CamBoxes]
    rate: float
    court: Court
    lo: int                           # first and last grid frame of the input: per-frame counters cover lo..hi
    hi: int

    @property
    def n(self) -> int:
        return self.hi - self.lo + 1

    def reach(self, a: Track, b: Track, P: ClosedParams) -> bool:
        return reachable(a, b, self.cams, self.rate, P.vmax, P.slack)


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def _end_point(tl: _Tl, last: bool) -> Track:
    k = tl.end if last else tl.start
    return {k: tl.tr[k]}


def _velocity(tl: _Tl, rate: float, at_end: bool, span: int = 5, vmax: float = 8.0) -> np.ndarray:
    """Least-squares velocity over the last (first) `span` frames; zero for a single sample."""
    ks = tl.ks[tl.ks >= tl.end - span] if at_end else tl.ks[tl.ks <= tl.start + span]
    if len(ks) < 2:
        return np.zeros(2)
    t = (ks - ks.mean()) / rate
    P = np.stack([tl.tr[int(k)].xy for k in ks])
    v = (t[:, None] * (P - P.mean(0))).sum(0) / (t ** 2).sum()
    return v * min(1.0, vmax / max(float(np.linalg.norm(v)), 1e-9))


def _handover_q(A: _Tl, B: _Tl, ctx: _Ctx, P: ClosedParams) -> float:
    """Squared normalised constant-velocity residual of the hand-over A -> B (forward from A's end and backward
    from B's start, averaged), over s0^2 + (s1 dt)^2 + the endpoints' position variance."""
    rate = ctx.rate
    dt = (B.start - A.end) / rate
    pa, pb = A.tr[A.end], B.tr[B.start]
    fwd = pa.xy + _velocity(A, rate, True, vmax=P.vmax) * dt
    bwd = pb.xy - _velocity(B, rate, False, vmax=P.vmax) * dt
    e2 = 0.5 * (np.sum((pb.xy - fwd) ** 2) + np.sum((pa.xy - bwd) ** 2))
    noise = 0.5 * float(np.trace(point_cov(ctx.cams, pa) + point_cov(ctx.cams, pb)))
    return float(e2 / (P.link_s0 ** 2 + (P.link_s1 * dt) ** 2 + noise))


def split_jumps(T: list[Track], cams: dict[str, CamBoxes], rate: float, P: ClosedParams = ClosedParams()) -> list[Track]:
    """Cut every tracklet between consecutive samples that fail reachability: no one person makes that step, so the
    source tracker switched identities there (boxmot tracks do; the conservative builder should not)."""
    out = []
    for tr in T:
        ks = sorted(tr)
        cur: Track = {ks[0]: tr[ks[0]]}
        for a, b in zip(ks, ks[1:]):
            if not reachable({a: tr[a]}, {b: tr[b]}, cams, rate, P.vmax, P.slack):
                out.append(cur)
                cur = {}
            cur[b] = tr[b]
        out.append(cur)
    return out


def duplicate_pairs(T: list[Track], teams: list[str], P: ClosedParams, cover: float,
                    min_shared: int = 3) -> list[tuple[int, int, int]]:
    """(a, b, shared frames) for team-compatible tracklets that coexist at the same spot: median distance over the
    shared frames < dup_m, >= dup_inside of them within 1.5 m, sharing >= `cover` of the shorter one (a long impure
    tracklet that brushes past a person is not his duplicate). In a closed set such a pair would take two slots for
    one person and push the real fifth player out."""
    at: dict[int, list[int]] = {}
    for i, t in enumerate(T):
        for k in t:
            at.setdefault(k, []).append(i)
    dist: dict[tuple[int, int], list[float]] = {}
    for k, ids in at.items():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if {teams[a], teams[b]} != set(TEAMS):
                    dist.setdefault((a, b), []).append(float(np.linalg.norm(T[a][k].xy - T[b][k].xy)))
    out = []
    for (a, b), v in dist.items():
        v = np.array(v)
        if len(v) >= max(min_shared, cover * min(len(T[a]), len(T[b]))) and np.median(v) < P.dup_m \
                and (v < 1.5).mean() >= P.dup_inside:
            out.append((a, b, len(v)))
    return out


def merge_duplicates(T: list[Track], cams: dict[str, CamBoxes], P: ClosedParams = ClosedParams()) -> tuple[list[Track], int]:
    """Merge duplicate pairs whose boxes never come from the same camera in one frame (the cross-view pairing
    missed them): shared frames become fused two-camera points. Largest overlaps first; a merge that would put two
    boxes of one camera into a frame is skipped. Returns (tracklets, merges)."""
    teams = [team_vote(t, cams)[0] for t in T]
    parent = list(range(len(T)))
    groups: dict[int, Track] = {i: dict(t) for i, t in enumerate(T)}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    n = 0
    for a, b, _ in sorted(duplicate_pairs(T, teams, P, P.dup_cover), key=lambda r: -r[2]):
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        A, B = groups[ra], groups[rb]
        if any({c for c, _ in A[k].boxes} & {c for c, _ in B[k].boxes} for k in A.keys() & B.keys()):
            continue
        for k, p in B.items():
            if k in A:
                boxes = sorted(A[k].boxes + p.boxes)
                A[k] = TrackPoint(fuse(cams, boxes)[0], boxes)
            else:
                A[k] = p
        parent[rb] = ra
        del groups[rb]
        n += 1
    return [dict(sorted(g.items())) for g in groups.values()], n


def _cliques(intervals: list[tuple[int, int, int]]) -> list[list[int]]:
    """Maximal sets of pairwise time-overlapping items of an interval graph; intervals are (start, end, id), ends
    inclusive. Sweep: the active set right before the first removal after an addition is maximal."""
    ev = sorted([(s, 0, i) for s, _, i in intervals] + [(e + 1, -1, i) for _, e, i in intervals])
    act: set[int] = set()
    out, grew = [], False
    for _, kind, i in ev:
        if kind == 0:
            act.add(i)
            grew = True
        else:
            if grew and len(act) > 1:
                out.append(sorted(act))
            grew = False
            act.discard(i)
    return out


def _proto(members: list[_Tl], n_med: int) -> dict | None:
    """Slot prototype in track_embedding's format, from the slot's tracklets weighted by their clean-crop count."""
    embs = [m.emb for m in members if m.emb is not None]
    if not embs:
        return None
    w = np.array([e["n"] for e in embs], float)
    mean = normalise((np.stack([e["mean"] for e in embs]) * w[:, None]).sum(0))
    meds = np.concatenate([e["medoids"] for e in embs])
    mw = np.concatenate([np.full(len(e["medoids"]), e["n"] / len(e["medoids"])) for e in embs])
    if len(meds) > n_med:
        meds = meds[np.sort(np.argsort(-mw)[:n_med])]
    return {"mean": mean, "medoids": meds, "n": int(w.sum()), "span": (0, 0)}


def _seed(tls: list[_Tl], team: str, k_slots: int, rate: float, exclusive: set[frozenset] = frozenset(),
          min_s: float = 2.0) -> list[int]:
    """Indices of k_slots same-team tracklets (>= min_s, with clean crops) that coexist, taken at the frame where
    the shortest of the chosen is longest (a long window of 5 distinct people). Falls back to the most coexisting.
    Duplicate pairs (`exclusive`) are never seeded together: they are one person."""
    cand = [i for i, t in enumerate(tls) if t.team == team and t.emb is not None and len(t) >= min_s * rate]
    if not cand:
        return []
    best, best_score = [], (-1, -1)
    frames = sorted({t.start for t in (tls[i] for i in cand)})
    for k in frames:
        act = [i for i in cand if tls[i].start <= k <= tls[i].end]
        act.sort(key=lambda i: -len(tls[i]))
        pick: list[int] = []
        for i in act:
            if len(pick) < k_slots and not any(frozenset((i, j)) in exclusive for j in pick):
                pick.append(i)
        score = (len(pick), min(len(tls[i]) for i in pick) if pick else 0)
        if score > best_score:
            best, best_score = pick, score
    return best


def _cap(team: str, P: ClosedParams) -> int:
    return P.on_court * (len(TEAMS) if team == "*" else 1)


# ---------------------------------------------------------------------------------------------------------------
# The MILP
# ---------------------------------------------------------------------------------------------------------------

def _build_static(tls: list[_Tl], slot_team: list[str], P: ClosedParams, ctx: _Ctx,
                  exclusive: list[tuple[int, int]] = ()):
    """Everything that does not depend on the prototypes: variables, constraint rows, link rewards."""
    from scipy import sparse
    rate = ctx.rate
    S = len(slot_team)
    var: dict[tuple[int, int], int] = {}
    for t, tl in enumerate(tls):
        for s, st in enumerate(slot_team):
            if not P.use_team or tl.team == "U" or tl.team == st:
                var[(t, s)] = len(var)
    rows, cols, vals, ub = [], [], [], []

    def row(entries: list[int], bound: float, coef: list[float] | None = None) -> None:
        r = len(ub)
        rows.extend([r] * len(entries)); cols.extend(entries); ub.append(bound)
        vals.extend(coef if coef is not None else [1.0] * len(entries))

    for t in range(len(tls)):                                       # one slot per tracklet
        e = [var[(t, s)] for s in range(S) if (t, s) in var]
        if len(e) > 1:
            row(e, 1)
    for a, b in exclusive:                                          # duplicates: one of the two at most
        row([var[(t, s)] for t in (a, b) for s in range(S) if (t, s) in var], 1)
    cl = _cliques([(tl.start, tl.end, t) for t, tl in enumerate(tls)])
    for C in cl:                                                    # no overlap within a slot
        for s in range(S):
            e = [var[(t, s)] for t in C if (t, s) in var]
            if len(e) > 1:
                row(e, 1)
    for team in dict.fromkeys(slot_team):                           # on-court limit per team (fixed row order)
        team_slots = [s for s in range(S) if slot_team[s] == team]
        cap = _cap(team, P)
        if len(team_slots) <= cap:
            continue
        for C in cl:
            e = [var[(t, s)] for t in C for s in team_slots if (t, s) in var]
            if len(e) > cap:
                row(e, cap)
    # reachability: only pairs closer in time than the pitch diagonal at vmax can fail
    horizon = int(np.ceil((np.hypot(ctx.court.length, ctx.court.width) + P.slack) / P.vmax * rate)) + 1
    order = sorted(range(len(tls)), key=lambda t: tls[t].start)
    starts = np.array([tls[t].start for t in order])
    n_unreach = n_rough = 0
    links: list[tuple[int, int, float]] = []
    for a in range(len(tls)):
        A = tls[a]
        lo, hi = np.searchsorted(starts, A.end + 1), np.searchsorted(starts, A.end + horizon + 1)
        for b in (order[q] for q in range(lo, hi)):
            B = tls[b]
            if P.use_team and "U" not in (A.team, B.team) and A.team != B.team:
                continue
            if not ctx.reach(_end_point(A, True), _end_point(B, False), P):
                n_unreach += 1
                for s in range(S):
                    if (a, s) in var and (b, s) in var:
                        row([var[(a, s)], var[(b, s)]], 1)
                continue
            gap = B.start - A.end
            if gap > P.link_gap_s * rate:
                continue
            q = _handover_q(A, B, ctx, P)
            if P.cv_gate and q > P.cv_gate ** 2:                    # rough hand-over: never inside one slot
                n_rough += 1
                for s in range(S):
                    if (a, s) in var and (b, s) in var:
                        row([var[(a, s)], var[(b, s)]], 1)
                continue
            if P.link_w > 0:
                r = P.link_w * np.exp(-0.5 * q)
                if A.emb is not None and B.emb is not None:
                    r *= float(np.clip(1.5 - 2.5 * app_distance(A.emb, B.emb), 0.0, 1.0))
                if r > 0.05 * P.link_w:
                    links.append((a, b, float(r)))
    # link variables z[a, b, s] <= x[a, s], z <= x[b, s]; each tracklet keeps at most one in- and one out-link
    zvar: dict[tuple[int, int, int], int] = {}
    nx = len(var)
    out_of: dict[int, list[int]] = {}
    in_of: dict[int, list[int]] = {}
    for a, b, r in links:
        for s in range(S):
            if (a, s) in var and (b, s) in var:
                z = nx + len(zvar)
                zvar[(a, b, s)] = z
                row([z, var[(a, s)]], 0, [1.0, -1.0])
                row([z, var[(b, s)]], 0, [1.0, -1.0])
                out_of.setdefault(a, []).append(z); in_of.setdefault(b, []).append(z)
    for lst in list(out_of.values()) + list(in_of.values()):
        if len(lst) > 1:
            row(lst, 1)
    link_r = np.zeros(len(zvar))
    rmap = {(a, b): r for a, b, r in links}
    for (a, b, s), z in zvar.items():
        link_r[z - nx] = rmap[(a, b)]
    A_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ub), nx + len(zvar)))
    return var, zvar, link_r, A_mat, np.array(ub, float), (n_unreach, n_rough)


def _costs(tls: list[_Tl], var: dict, is_spare: list[bool], protos: list[dict | None], P: ClosedParams) -> np.ndarray:
    c = np.zeros(len(var))
    for (t, s), j in var.items():
        tl = tls[t]
        if tl.emb is None or protos[s] is None:
            d = P.d_none
        else:
            d = app_distance(tl.emb, protos[s])
        pen = (P.u_pen if tl.team == "U" and P.use_team else 0.0) + (P.spare_pen if is_spare[s] else 0.0)
        c[j] = len(tl) * (P.app_w * (d - P.d_none) + P.d_none - P.coverage + pen)
    return c


def _assign(tls: list[_Tl], ctx: _Ctx, P: ClosedParams, st: ClosedStats,
            exclusive: list[tuple[int, int]] = ()) -> dict[int, int]:
    """{tracklet index: slot} from the EM-refined closed-set MILP."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    if P.use_team:
        slot_team = [tm for tm in TEAMS for _ in range(P.slots + P.spare)]
        is_spare = [q % (P.slots + P.spare) >= P.slots for q in range(len(slot_team))]
    else:                                                           # ablation: one pool, seeds fill 0..9
        slot_team = ["*"] * (2 * (P.slots + P.spare))
        is_spare = [q >= 2 * P.slots for q in range(len(slot_team))]
    st.slot_team, st.is_spare = slot_team, is_spare
    if not tls:
        return {}
    var, zvar, link_r, A_mat, ub, (n_unreach, n_rough) = _build_static(tls, slot_team, P, ctx, exclusive)
    st.n_vars, st.n_rows = A_mat.shape[1], A_mat.shape[0]
    # seeds: 5 coexisting same-team tracklets per team
    protos: list[dict | None] = [None] * len(slot_team)
    pins: dict[int, int] = {}
    for tm in TEAMS:
        seeds = _seed(tls, tm, P.slots, ctx.rate, {frozenset(p) for p in exclusive})
        st.seeds[tm] = [(tls[i].start, tls[i].end) for i in seeds]
        base = slot_team.index(tm) if P.use_team else TEAMS.index(tm) * P.slots
        for q, i in enumerate(seeds):
            protos[base + q] = tls[i].emb
            pins[i] = base + q
    assignment: dict[int, int] = {}
    for _ in range(P.rounds):
        c = np.concatenate([_costs(tls, var, is_spare, protos, P), -link_r])
        lb = np.zeros(len(c))
        if P.pin_seeds:
            for i, s in pins.items():
                if (i, s) in var:
                    lb[var[(i, s)]] = 1.0
        t0 = time.time()
        opts = {"time_limit": P.time_limit, "mip_rel_gap": 1e-4, "disp": False}
        res = milp(c, integrality=np.ones(len(c)), bounds=Bounds(lb, np.ones(len(c))),
                   constraints=LinearConstraint(A_mat, -np.inf, ub), options=opts)
        if res.x is None and lb.any():                              # pinned seeds clash with a hard row: unpin
            st.unpinned = True
            res = milp(c, integrality=np.ones(len(c)), bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
                       constraints=LinearConstraint(A_mat, -np.inf, ub), options=opts)
        st.solver_s.append(round(time.time() - t0, 2))
        if res.x is None:
            raise RuntimeError(f"closed-set MILP failed: {res.message}")
        new = {t: s for (t, s), j in var.items() if res.x[j] > 0.5}
        st.changes.append(sum(1 for t in set(new) | set(assignment) if new.get(t) != assignment.get(t)))
        assignment = new
        members: dict[int, list[_Tl]] = {}
        for t, s in assignment.items():
            members.setdefault(s, []).append(tls[t])
        protos = [_proto(members.get(s, []), P.proto_medoids) if members.get(s) else protos[s]
                  for s in range(len(slot_team))]
    st.unreachable_pairs = n_unreach
    st.rough_pairs = n_rough
    st.links = len(link_r)
    return assignment


# ---------------------------------------------------------------------------------------------------------------
# Short tracklets, output
# ---------------------------------------------------------------------------------------------------------------

def _active(tr: Track, ctx: _Ctx, max_gap: int) -> np.ndarray:
    """0/1 per frame of the input span: the track has a point there or will get one from fill_gaps (gap <= max_gap)."""
    a = np.zeros(ctx.n, int)
    ks = np.array(sorted(k - ctx.lo for k in tr if ctx.lo <= k <= ctx.hi), int)
    if not len(ks):
        return a
    a[ks] = 1
    for lo, hi in zip(ks, ks[1:]):
        if 1 < hi - lo <= max_gap:
            a[lo:hi] = 1
    return a


def _attach_short(slots: list[Track], owner: list[dict[int, int]], slot_team: list[str], shorts: list[_Tl],
                  ctx: _Ctx, P: ClosedParams) -> int:
    """Give each short tracklet (longest first) to the team-compatible slot it fits best: no shared frame, reachable
    from the slot's previous point and to its next one, at least one side within attach_gap_s; the score is the
    larger junction distance in excess of a 1 m + 4 m/s * dt allowance. Unfit ones are left out."""
    rate = ctx.rate
    keys = [np.array(sorted(s), int) for s in slots]
    max_gap = int(P.fill_gap_s * rate)
    busy = {tm: np.zeros(ctx.n, int) for tm in set(slot_team)}
    act = [_active(tr, ctx, max_gap) for tr in slots]
    for a, tm in zip(act, slot_team):
        busy[tm] += a
    n = 0
    for q, tl in sorted(enumerate(shorts), key=lambda x: -len(x[1])):
        best, best_score = None, np.inf
        for s, tr in enumerate(slots):
            if P.use_team and tl.team != "U" and slot_team[s] != tl.team:
                continue
            ks = keys[s]
            if not len(ks):
                continue
            span = slice(tl.start - ctx.lo, tl.end - ctx.lo + 1)
            if (busy[slot_team[s]][span] - act[s][span]).max(initial=0) >= _cap(slot_team[s], P):
                continue                                            # the other slots already fill the team there
            lo, hi = np.searchsorted(ks, tl.start), np.searchsorted(ks, tl.end, side="right")
            if hi > lo:                                             # slot has points inside the span
                continue
            prev = int(ks[lo - 1]) if lo > 0 else None
            nxt = int(ks[lo]) if lo < len(ks) else None
            gaps = [tl.start - prev if prev is not None else np.inf, nxt - tl.end if nxt is not None else np.inf]
            if min(gaps) > P.attach_gap_s * rate:
                continue
            score, ok = 0.0, True
            for side, other in ((0, prev), (1, nxt)):
                if other is None:
                    continue
                a, b = ({other: tr[other]}, _end_point(tl, False)) if side == 0 else (_end_point(tl, True), {other: tr[other]})
                if not ctx.reach(a, b, P):
                    ok = False
                    break
                if gaps[side] <= P.attach_gap_s * rate:
                    pa, pb = next(iter(a.values())).xy, next(iter(b.values())).xy
                    dt = gaps[side] / rate
                    score = max(score, float(np.linalg.norm(pb - pa)) - (1.0 + 4.0 * dt))
            if ok and score < best_score:
                best, best_score = s, score
        if best is not None and best_score <= 0.0:
            slots[best].update(tl.tr)
            new = _active(slots[best], ctx, max_gap)
            busy[slot_team[best]] += new - act[best]
            act[best] = new
            owner[best].update({k: -1 - q for k in tl.tr})
            keys[best] = np.array(sorted(slots[best]), int)
            n += 1
    return n


def _fill_capped(slots: list[Track], slot_team: list[str], is_spare: list[bool], ctx: _Ctx,
                 P: ClosedParams) -> list[Track]:
    """fill_gaps per slot, then drop interpolated (box-less) points where a team would exceed the on-court cap: a
    slot's gap can coincide with another slot's tracklet, and filling both would show 6 players. Spare slots give
    way first."""
    cams = ctx.cams
    if P.refuse:
        slots = [{k: TrackPoint(fuse(cams, p.boxes)[0], p.boxes) if len(p.boxes) > 1 else p for k, p in s.items()}
                 for s in slots]
    filled = [fill_gaps(s, ctx.rate, P.fill_gap_s, P.vmax) if s else {} for s in slots]
    if P.smooth:
        filled = [savgol_smooth(f) for f in filled]
    for tm in set(slot_team):
        idx = [q for q in range(len(slots)) if slot_team[q] == tm]
        cnt = np.zeros(ctx.n, int)
        for q in idx:
            ks = np.array([k - ctx.lo for k in filled[q] if ctx.lo <= k <= ctx.hi], int)
            cnt[ks] += 1
        for i in np.flatnonzero(cnt > _cap(tm, P)):
            k = int(i) + ctx.lo
            for q in sorted(idx, key=lambda q: not is_spare[q]):
                if cnt[i] <= _cap(tm, P):
                    break
                if k in filled[q] and not filled[q][k].boxes:
                    del filled[q][k]
                    cnt[i] -= 1
    return filled


def _junctions(tr: Track, own: dict[int, int]) -> list[tuple[int, int]]:
    """Consecutive samples (a, b) of a slot track that come from different source tracklets."""
    ks = sorted(tr)
    return [(a, b) for a, b in zip(ks, ks[1:]) if own[a] != own[b]]


def _exact_5v5(slots: list[Track], slot_team: list[str], ctx: _Ctx) -> float:
    cnt = {tm: np.zeros(ctx.n, int) for tm in set(slot_team)}
    for tr, tm in zip(slots, slot_team):
        ks = np.array([k - ctx.lo for k in tr if ctx.lo <= k <= ctx.hi], int)
        cnt[tm][ks] += 1
    ok = np.ones(ctx.n, bool)
    for tm, c in cnt.items():
        ok &= c == (5 if len(cnt) == 2 else 10)
    return round(float(ok.mean()), 3)


def default_border_exempt(cams: dict[str, CamBoxes]) -> dict[str, tuple[str, ...]]:
    """Top image edge exempt from the clean-crop border rule on every camera. The phones stand at two diagonal
    corners looking across the pitch (docs/capture-guide.md), so the far touchline sits at the top of each picture:
    a box cut by the top edge is a far-side player whose head is out of frame, still one whole visible person and
    the right one, not a half-box of someone leaving the view. Requiring a top margin threw that evidence away
    (cam2 of the test match is aimed low and cut 86 % of the heads: 11 % clean crops instead of cam1's 76 %).
    Left, right and bottom keep the margin: a box cut there is usually a player entering or leaving the view."""
    return {cam: ("top",) for cam in cams}


def assign_identities(tracklets: list[Track], cams: dict[str, CamBoxes], rate: float,
                      params: ClosedParams = ClosedParams(), border_exempt: dict[str, tuple[str, ...]] | None = None,
                      stats: ClosedStats | None = None, court: Court | None = None) -> list[Track]:
    """Closed-set assignment of tracklets (e.g. tracklets.build_tracklets) to at most slots + spare identities per
    team (see the module docstring). `rate` is the grid rate in Hz; `border_exempt` goes to identity.clean_masks
    (None: default_border_exempt, the top edge of every camera; {} for no exemption). Returns one track per used
    slot, in slot order (team A first; stats.output_team gives each track's team), with re-fused, gap-filled
    (boxes=[]) and smoothed positions. Every box of the input appears at most once in the output."""
    P = params
    st = stats if stats is not None else ClosedStats()
    court = court or Court()
    T = [t for t in tracklets if t]
    if not T:
        return []
    lo = min(min(t) for t in T)
    hi = max(max(t) for t in T)
    ctx = _Ctx(cams, float(rate), court, int(lo), int(hi))
    if P.jump_split:
        T = split_jumps(T, cams, rate, P)
    if P.dbscan_eps:
        T = [piece for t in T for piece in dbscan_split(t, cams, rate, eps=P.dbscan_eps, min_samples=15)]
    if P.team_split:
        T = split_team_flips(T, cams)
    if P.merge_dups:
        T, st.merged_dups = merge_duplicates(T, cams, P)
    clean = clean_masks(cams, border_exempt=default_border_exempt(cams) if border_exempt is None else border_exempt)
    tls = []
    for tr in T:
        ks = np.array(sorted(tr), int)
        tls.append(_Tl(tr, ks, team_vote(tr, cams)[0], None))
    long_ = [tl for tl in tls if len(tl) >= P.min_len_s * rate]
    short = [tl for tl in tls if len(tl) < P.min_len_s * rate]
    for tl in long_:
        tl.emb = track_embedding(tl.tr, cams, clean)
    st.tracklets, st.solved, st.short_total = len(tls), len(long_), len(short)
    exclusive = []
    if P.exclude_dups:
        exclusive = [(a, b) for a, b, _ in
                     duplicate_pairs([t.tr for t in long_], [t.team for t in long_], P, P.excl_cover)]
    st.exclusive_pairs = len(exclusive)
    assignment = _assign(long_, ctx, P, st, exclusive)
    slot_team = st.slot_team
    slots: list[Track] = [{} for _ in slot_team]
    owner: list[dict[int, int]] = [{} for _ in slot_team]          # frame -> source tracklet (short ones < 0)
    for t, s in assignment.items():
        slots[s].update(long_[t].tr)
        owner[s].update({k: t for k in long_[t].tr})
    st.assigned = len(assignment)
    st.unassigned_long = len(long_) - len(assignment)
    st.unassigned_long_samples = sum(len(long_[t]) for t in range(len(long_)) if t not in assignment)
    for s, sp in enumerate(st.is_spare):
        if sp:
            st.spare_samples[s] = sum(len(long_[t]) for t, q in assignment.items() if q == s)
            st.spare_tracklets[s] = sum(1 for q in assignment.values() if q == s)
    st.attached_short = _attach_short(slots, owner, slot_team, short, ctx, P)
    total = sum(len(tl) for tl in tls)
    st.assigned_samples_share = round(sum(len(s) for s in slots) / max(total, 1), 4)
    st.exact_5v5_slots = _exact_5v5(slots, slot_team, ctx)
    filled = _fill_capped(slots, slot_team, st.is_spare, ctx, P)
    st.exact_5v5_filled = _exact_5v5(filled, slot_team, ctx)
    out = [f for f in filled if f]
    st.output_team = [tm for f, tm in zip(filled, slot_team) if f]
    st.reach_violations = sum(1 for tr, own in zip(slots, owner) for a, b in _junctions(tr, own)
                              if not ctx.reach({a: tr[a]}, {b: tr[b]}, P))
    st.junctions = [(q, a, b) for q, (tr, own) in enumerate((x for x in zip(slots, owner) if x[0]))
                    for a, b in _junctions(tr, own)]          # (index into the output, last frame, first frame)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Full matches: overlapping windows, chained by the boxes they share
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class WindowedStats:
    windows: list = field(default_factory=list)       # (first, last) grid frame of each window, inclusive
    cuts: list = field(default_factory=list)          # frame where ownership passes from window w to w + 1
    per_window: list = field(default_factory=list)    # ClosedStats of each window's solve
    matched: list = field(default_factory=list)       # identities carried over at each junction
    new_ids: list = field(default_factory=list)       # identities started at each junction (no match)
    ended: list = field(default_factory=list)         # identities of window w with no continuation
    agreement: list = field(default_factory=list)     # per junction: matched shared boxes / all overlap boxes
    output_team: list = field(default_factory=list)   # 'A' / 'B' per output identity (team of its first slot)


def _windows(lo: int, hi: int, win: int, step: int) -> list[tuple[int, int]]:
    """[first, last] frame of each window: fixed step, the last one ends at hi (and may be shorter). A remainder
    shorter than half the overlap is absorbed by the window before it instead of getting a window of its own: that
    window would own (from the cut in the middle of the overlap on) little more than its own edge, and a 6:00.2
    input would be solved as 6:00 + a 1-minute window instead of once."""
    out = []
    s = lo
    while True:
        e = min(s + win - 1, hi)
        if hi - e < (win - step) // 2:
            e = hi
        out.append((s, e))
        if e >= hi:
            return out
        s += step


def _box_frames(track: Track, lo: int, hi: int) -> set[tuple[int, str, int]]:
    return {(k, c, i) for k, p in track.items() if lo <= k <= hi for c, i in p.boxes}


def _chain_match(prev: list[Track], cur: list[Track], lo: int, hi: int, min_boxes: int,
                 min_agree: float) -> tuple[dict[int, int], int, int]:
    """{index in cur: index in prev} for identities of two windows that are the same person: Hungarian on the
    number of boxes they share in the overlap lo..hi, accepted when they share >= min_boxes and >= min_agree of the
    smaller one's overlap boxes. Returns (matches, matched shared boxes, all overlap boxes of both)."""
    from scipy.optimize import linear_sum_assignment
    A = [_box_frames(t, lo, hi) for t in prev]
    B = [_box_frames(t, lo, hi) for t in cur]
    if not A or not B:
        return {}, 0, sum(map(len, A)) + sum(map(len, B))
    M = np.array([[len(a & b) for b in B] for a in A], float)
    r, c = linear_sum_assignment(-M)
    out, shared = {}, 0
    for i, j in zip(r, c):
        n = int(M[i, j])
        if n >= min_boxes and n >= min_agree * max(1, min(len(A[i]), len(B[j]))):
            out[int(j)] = int(i)
            shared += n
    return out, shared, sum(map(len, A)) + sum(map(len, B))


def assign_identities_windowed(tracklets: list[Track], cams: dict[str, CamBoxes], rate: float,
                               params: ClosedParams = ClosedParams(), window_s: float = 360.0,
                               overlap_s: float = 60.0, border_exempt: dict[str, tuple[str, ...]] | None = None,
                               min_boxes: int = 10, min_agree: float = 0.5, stats: WindowedStats | None = None,
                               court: Court | None = None) -> list[Track]:
    """assign_identities for a whole match: overlapping windows, each solved on its own, chained into identities.

    Why windows: the MILP grows with the number of tracklets and the pairs within reach of each other (~600 long
    tracklets per 6 minutes); a 40-minute match in one solve is slow and its EM prototypes would have to cover
    lighting changes and substitutions. The closed set itself is local anyway: 5 + 1 slots per team only have to
    hold the people who play in the window, so a substitute who comes on later simply takes a slot of a later window.

    Design:
      1. Windows of window_s seconds start every window_s - overlap_s seconds (the last one ends with the input and
         may be shorter; a remainder under half the overlap is absorbed by the window before). Tracklets are clipped to each window (a tracklet crossing the edge is cut there) and the
         window is solved by assign_identities with the same params.
      2. Chaining: the identities of windows w and w + 1 were both computed on the overlap, from the same boxes.
         They are matched by a Hungarian assignment on the number of overlap boxes they share; a match needs
         >= min_boxes shared boxes and >= min_agree of the smaller side's overlap boxes. Matched identities continue
         the same output identity; an unmatched one of w + 1 starts a new identity (substitute, someone the slots
         now see differently), an unmatched one of w ends.
      3. Ownership: every frame belongs to exactly one window. The cut is the middle of the overlap, where both
         solves are farthest from their window edges (a solve is weakest near its edges: tracklets are clipped and
         the motion links stop there). An identity takes its points from window w before the cut and from w + 1
         from the cut on.
    Because each window's solve uses every box at most once and frames are partitioned between windows, the result
    never uses a box twice, and each identity's points come from consecutive windows in frame order, so they stay
    time-ordered. Gap filling and smoothing are per window; a gap that spans a cut is filled only if both windows
    filled it. With window_s >= the input span (or less than half the overlap
    short of it) this is exactly one assign_identities call.

    On the 6-minute test match (windows of 180 s every 120 s vs one solve): 13 vs 12 identities, count_5v5 0.845 vs
    0.841, box co-assignment F1 0.94 against the single solve; the windows agree with each other on 97-100 % of
    the overlap boxes, and the differences sit mostly in the last 30 s (an edge of both solves). Witnessed swaps
    4 vs 3, events 0.50 vs 0.33 per minute: about equal, not better or worse by the label-free metrics."""
    st = stats if stats is not None else WindowedStats()
    T = [t for t in tracklets if t]
    if not T:
        return []
    lo = min(min(t) for t in T)
    hi = max(max(t) for t in T)
    win = max(1, int(round(window_s * rate)))
    ov = int(round(overlap_s * rate))
    if not 0 <= ov < win:
        raise ValueError("overlap_s must be >= 0 and shorter than window_s")
    wins = _windows(lo, hi, win, win - ov)
    st.windows = wins
    sols: list[list[Track]] = []
    for a, b in wins:
        clip = [c for c in ({k: p for k, p in t.items() if a <= k <= b} for t in T if min(t) <= b and max(t) >= a) if c]
        wst = ClosedStats()
        sols.append(assign_identities(clip, cams, rate, params, border_exempt, wst, court))
        st.per_window.append(wst)
    # cut frames: window w owns [cut_{w-1}, cut_w)
    cuts = [(wins[w + 1][0] + wins[w][1] + 1) // 2 for w in range(len(wins) - 1)]
    st.cuts = cuts
    bounds = [lo] + cuts + [hi + 1]
    ids: list[Track] = []
    team: list[str] = []
    cur_id: list[int] = []                                          # output identity of each track of the last window
    for w, sol in enumerate(sols):
        a, b = bounds[w], bounds[w + 1]
        if w == 0:
            match = {}
        else:
            match, shared, total = _chain_match(sols[w - 1], sol, wins[w][0], wins[w - 1][1], min_boxes, min_agree)
            st.matched.append(len(match))
            st.new_ids.append(len(sol) - len(match))
            st.ended.append(len(sols[w - 1]) - len(match))
            st.agreement.append(round(2 * shared / max(total, 1), 4))
        nxt = []
        for j, tr in enumerate(sol):
            q = cur_id[match[j]] if j in match else len(ids)
            if q == len(ids):
                ids.append({})
                team.append(st.per_window[w].output_team[j])
            ids[q].update({k: p for k, p in tr.items() if a <= k < b})
            nxt.append(q)
        cur_id = nxt
    st.output_team = [tm for t, tm in zip(ids, team) if t]
    return [dict(sorted(t.items())) for t in ids if t]
