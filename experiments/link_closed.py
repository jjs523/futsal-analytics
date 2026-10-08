"""Closed-set identity assignment with the known number of players (research_plan.md V2).

Futsal is 5 v 5, so instead of growing identities until the linker runs out of evidence, every tracklet either
gets one of K identity slots per team (5 regular + 1 'spare' for substitutes and noise) or is left out. The solve
is one MILP (scipy.optimize.milp / HiGHS) over binaries x[t, s] (tracklet t -> slot s):

    sum_s x[t, s] <= 1                                    every tracklet at most once
    sum_{t in C} x[t, s] <= 1    for each maximal clique C of time-overlapping tracklets   (no overlap in a slot)
    sum_{t in C} sum_{s in team} x[t, s] <= on_court     at most 5 per team on the pitch at once (spare = sub)
    x[t, s] + x[u, s] <= 1       for every non-overlapping pair that fails blocks.reachable (B3), and for every
                                 hand-over within link_gap_s whose constant-velocity residual exceeds cv_gate sigma
    sum_s x[a, s] + x[b, s] <= 1 for duplicate pairs (same spot, same time, same team)

Overlap cliques of an interval graph are a consecutive-ones matrix, so without the reachability rows the LP is
already integral; the reachability rows are pairwise, but only pairs closer in time than the pitch diagonal at
vmax can fail, and a slot-mate between two unreachable tracklets would itself be unreachable from one of them, so
the pairwise rows are (up to that triangle argument) the exact successor-chain condition.

Objective: sum_t,s x[t, s] * len_t * (d_app(t, proto_s) - coverage)  [+ U-team and spare penalties]
           - sum of motion-continuity rewards z[t, u, s] for short, smooth hand-overs t -> u kept in one slot.
Appearance alone cannot hold a far-side player (no clean crops) in his slot; the link rewards do.

Before the solve, tracklets are cut where they fail B3 internally (boxmot id switches) and where the bib flips (B1),
and duplicates seen by complementary cameras (missed cross-view pairs) are merged: in a closed set a duplicate
would take a second slot and push the real fifth player out.
Prototypes come from the longest window where 5 same-team tracklets with clean crops coexist (5 distinct people,
no label symmetry), then EM: re-estimate each slot's prototype from what it was given, re-solve.
Tracklets shorter than `min_len_s` stay out of the solve and are attached afterwards to the slot whose neighbouring
points they reach best; finally B6: re-fuse multi-box points with the B2 covariances, fill_gaps (never above 5 per
team) and Savitzky-Golay smoothing.

    python experiments/harness.py lc_xvcons link_closed
    python experiments/metrics2.py lc_fused lc_xvcons lc_xvhyb --parts first,second
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from harness import Data, Track, TrackPoint, register
from blocks import app_distance, clean_masks, dbscan_split, fill_gaps, fuse, normalise, point_cov, reachable, \
    smooth as savgol_smooth, split_team_flips, team_vote, track_embedding

TEAMS = ("Y", "N")


@dataclass
class ClosedParams:
    slots: int = 5                    # regular slots per team
    spare: int = 1                    # extra high-cost slots per team (substitutes, noise)
    on_court: int = 5                 # at most this many slots of one team active in any frame
    min_len_s: float = 1.0            # shorter tracklets are attached after the solve
    coverage: float = 0.5             # per-sample reward for assigning a tracklet (lambda_cov)
    d_none: float = 0.35              # appearance cost of a tracklet without clean crops
    app_w: float = 1.0                # weight of (d_app - d_none): 0 = motion / team / coverage only
    u_pen: float = 0.15               # per-sample penalty for a 'U' tracklet in either team
    spare_pen: float = 0.2            # per-sample penalty for a spare slot
    vmax: float = 8.0                 # B3 gate
    slack: float = 1.0
    link_w: float = 10.0              # max reward (in sample-cost units) for a smooth hand-over kept in one slot
    link_gap_s: float = 3.0           # hand-overs considered for the reward and the cv gate
    cv_gate: float | None = 4.0       # forbid hand-overs within link_gap_s whose CV residual exceeds this many sigma
    link_s0: float = 0.7              # motion residual scale: s0 m + s1 m/s * dt
    link_s1: float = 1.5
    proto_medoids: int = 30           # medoids kept per slot prototype
    rounds: int = 4                   # 1 initial solve + EM refinements
    pin_seeds: bool = True            # keep the initial 5 tracklets in their slots (anchors the labels)
    team_split: bool = True           # B1 flip split before assignment
    jump_split: bool = True           # cut tracklets at steps that fail B3 (an id switch inside the source tracker)
    dbscan_eps: float | None = None   # purity split (blocks.dbscan_split) before assignment, e.g. 0.25
    dup_m: float = 0.8                # duplicates: overlapping, team-compatible, median distance below this ...
    dup_inside: float = 0.8           # ... and this share of shared frames within 1.5 m
    dup_cover: float = 0.0            # ... over at least this share of the shorter tracklet (merging)
    excl_cover: float = 0.5           # same, for exclusion (drops a whole tracklet, so ask for more evidence)
    merge_dups: bool = True           # merge duplicates seen by complementary cameras (missed cross-view pairs)
    exclude_dups: bool = True         # other duplicates: at most one of the pair gets a slot
    use_team: bool = True             # False: one 12-slot pool without team constraint (ablation)
    attach_gap_s: float = 1.0         # short tracklets: at least one side within this gap of the slot's points
    fill_gap_s: float = 3.0
    refuse: bool = True               # re-fuse multi-box points with the anisotropic B2 covariances
    smooth: bool = True               # B6 Savitzky-Golay (7, 2) after filling
    time_limit: float = 60.0


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
    exact_5v5_slots: float = 0.0      # frames where exactly 5 Y and 5 N slots are active (before gap filling)
    exact_5v5_filled: float = 0.0
    reach_violations: int = 0         # consecutive slot-mates failing B3 after attachment (should be 0)
    changes: list = field(default_factory=list)   # tracklets that changed slot per EM round
    unreachable_pairs: int = 0        # B3 exclusion rows (pairs)
    rough_pairs: int = 0              # CV-gate exclusion rows (pairs)
    links: int = 0                    # link-reward variables
    slot_team: list = field(default_factory=list)
    is_spare: list = field(default_factory=list)
    junctions: list = field(default_factory=list)  # (output index, last frame, first frame) of in-slot hand-overs


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


# ---------------------------------------------------------------------------------------------------------------
# Tracklet sources
# ---------------------------------------------------------------------------------------------------------------

def fused_tracklets(data: Data, ambiguity: float = 0.3, min_conf: float = 0.3) -> list[Track]:
    """pitch_tracker.tracklets on the per-frame fused detections, with box references (source (a))."""
    from futsal.pipeline import pitch_tracker
    from trackers import fused_frames
    frames = fused_frames(data, min_conf)
    out = []
    for tl in pitch_tracker.tracklets(frames, data.rate, ambiguity=ambiguity):
        tr: Track = {}
        for k, xy in sorted(tl.points.items()):
            boxes = []
            for f in frames[k]:
                if np.allclose(f.xy, xy, atol=1e-9):
                    boxes = [(o.camera, o.track_id) for o in f.sources]
                    break
            tr[int(k)] = TrackPoint(np.asarray(xy, float).copy(), boxes)
        out.append(tr)
    return out


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


def _handover_q(A: _Tl, B: _Tl, data: Data, P: ClosedParams) -> float:
    """Squared normalised constant-velocity residual of the hand-over A -> B (forward from A's end and backward
    from B's start, averaged), over s0^2 + (s1 dt)^2 + the endpoints' position variance (B2)."""
    rate = data.rate
    dt = (B.start - A.end) / rate
    pa, pb = A.tr[A.end], B.tr[B.start]
    fwd = pa.xy + _velocity(A, rate, True, vmax=P.vmax) * dt
    bwd = pb.xy - _velocity(B, rate, False, vmax=P.vmax) * dt
    e2 = 0.5 * (np.sum((pb.xy - fwd) ** 2) + np.sum((pa.xy - bwd) ** 2))
    noise = 0.5 * float(np.trace(point_cov(data, pa) + point_cov(data, pb)))
    return float(e2 / (P.link_s0 ** 2 + (P.link_s1 * dt) ** 2 + noise))


def split_jumps(T: list[Track], data: Data, P: ClosedParams) -> list[Track]:
    """Cut every tracklet between consecutive samples that fail B3: no one person makes that step, so the source
    tracker switched identities there (boxmot tracks do; the conservative builder never should)."""
    out = []
    for tr in T:
        ks = sorted(tr)
        cur: Track = {ks[0]: tr[ks[0]]}
        for a, b in zip(ks, ks[1:]):
            if not reachable({a: tr[a]}, {b: tr[b]}, data, P.vmax, P.slack):
                out.append(cur)
                cur = {}
            cur[b] = tr[b]
        out.append(cur)
    return out


def duplicate_pairs(T: list[Track], teams: list[str], P: ClosedParams, cover: float,
                    min_shared: int = 3) -> list[tuple[int, int, int]]:
    """(a, b, shared frames) for team-compatible tracklets that coexist at the same spot: median distance over the
    shared frames < dup_m, >= dup_inside of them within 1.5 m, sharing >= `cover` of the shorter one (a long
    impure tracklet that brushes past a person is not his duplicate). In a closed set such a pair would take two slots
    for one person and push the real fifth player out."""
    at: dict[int, list[int]] = {}
    for i, t in enumerate(T):
        for k in t:
            at.setdefault(k, []).append(i)
    dist: dict[tuple[int, int], list[float]] = {}
    for k, ids in at.items():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if {teams[a], teams[b]} != {"Y", "N"}:
                    dist.setdefault((a, b), []).append(float(np.linalg.norm(T[a][k].xy - T[b][k].xy)))
    out = []
    for (a, b), v in dist.items():
        v = np.array(v)
        if len(v) >= max(min_shared, cover * min(len(T[a]), len(T[b]))) and np.median(v) < P.dup_m \
                and (v < 1.5).mean() >= P.dup_inside:
            out.append((a, b, len(v)))
    return out


def merge_duplicates(T: list[Track], data: Data, P: ClosedParams) -> tuple[list[Track], int]:
    """Merge duplicate pairs whose boxes never come from the same camera in one frame (the cross-view pairing
    missed them): shared frames become fused two-camera points (B2 fusion). Largest overlaps first; a merge that
    would put two boxes of one camera into a frame is skipped. Returns (tracklets, merges)."""
    teams = [team_vote(t, data)[0] for t in T]
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
                A[k] = TrackPoint(fuse(data, boxes)[0], boxes)
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


# ---------------------------------------------------------------------------------------------------------------
# The MILP
# ---------------------------------------------------------------------------------------------------------------

def _build_static(tls: list[_Tl], slot_team: list[str], P: ClosedParams, data: Data,
                  exclusive: list[tuple[int, int]] = ()):
    """Everything that does not depend on the prototypes: variables, constraint rows, link rewards."""
    rate = data.rate
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
    for team in set(slot_team):                                     # on-court limit per team
        team_slots = [s for s in range(S) if slot_team[s] == team]
        cap = P.on_court * (len(TEAMS) if team == "*" else 1)
        if len(team_slots) <= cap:
            continue
        for C in cl:
            e = [var[(t, s)] for t in C for s in team_slots if (t, s) in var]
            if len(e) > cap:
                row(e, cap)
    # reachability: only pairs closer in time than the pitch diagonal at vmax can fail
    horizon = int(np.ceil((np.hypot(data.court.length, data.court.width) + P.slack) / P.vmax * rate)) + 1
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
            if not reachable(_end_point(A, True), _end_point(B, False), data, P.vmax, P.slack):
                n_unreach += 1
                for s in range(S):
                    if (a, s) in var and (b, s) in var:
                        row([var[(a, s)], var[(b, s)]], 1)
                continue
            gap = B.start - A.end
            if gap > P.link_gap_s * rate:
                continue
            q = _handover_q(A, B, data, P)
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


def _costs(tls: list[_Tl], var: dict, slot_team: list[str], is_spare: list[bool], protos: list[dict | None],
           P: ClosedParams) -> np.ndarray:
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


def assign(tls: list[_Tl], data: Data, P: ClosedParams, st: ClosedStats,
           exclusive: list[tuple[int, int]] = ()) -> dict[int, int]:
    """{tracklet index: slot} from the EM-refined closed-set MILP."""
    if P.use_team:
        slot_team = [tm for tm in TEAMS for _ in range(P.slots + P.spare)]
        is_spare = [q % (P.slots + P.spare) >= P.slots for q in range(len(slot_team))]
    else:                                                           # ablation: one pool, seeds fill 0..9
        slot_team = ["*"] * (2 * (P.slots + P.spare))
        is_spare = [q >= 2 * P.slots for q in range(len(slot_team))]
    var, zvar, link_r, A_mat, ub, (n_unreach, n_rough) = _build_static(tls, slot_team, P, data, exclusive)
    st.n_vars, st.n_rows = A_mat.shape[1], A_mat.shape[0]
    # seeds: 5 coexisting same-team tracklets per team
    protos: list[dict | None] = [None] * len(slot_team)
    pins: dict[int, int] = {}
    for tm in TEAMS:
        seeds = _seed(tls, tm, P.slots, data.rate, {frozenset(p) for p in exclusive})
        st.seeds[tm] = [(tls[i].start, tls[i].end) for i in seeds]
        base = slot_team.index(tm) if P.use_team else TEAMS.index(tm) * P.slots
        for q, i in enumerate(seeds):
            protos[base + q] = tls[i].emb
            pins[i] = base + q
    assignment: dict[int, int] = {}
    nx = len(var)
    for rnd in range(P.rounds):
        c = np.concatenate([_costs(tls, var, slot_team, is_spare, protos, P), -link_r])
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
    st.slot_team = slot_team
    st.is_spare = is_spare
    st.unreachable_pairs = n_unreach
    st.rough_pairs = n_rough
    st.links = len(link_r)
    return assignment


# ---------------------------------------------------------------------------------------------------------------
# Short tracklets, output
# ---------------------------------------------------------------------------------------------------------------

def _attach_short(slots: list[Track], owner: list[dict[int, int]], slot_team: list[str], shorts: list[_Tl],
                  data: Data, P: ClosedParams) -> int:
    """Give each short tracklet (longest first) to the team-compatible slot it fits best: no shared frame, B3
    reachable from the slot's previous point and to its next one, at least one side within attach_gap_s; the score
    is the larger junction distance in excess of a 1 m + 4 m/s * dt allowance. Unfit ones are left out."""
    keys = [np.array(sorted(s), int) for s in slots]
    max_gap = int(P.fill_gap_s * data.rate)
    busy = {tm: np.zeros(data.n + 1, int) for tm in set(slot_team)}
    act = [_active(tr, data.n + 1, max_gap) for tr in slots]
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
            span = slice(max(tl.start, 0), tl.end + 1)
            if (busy[slot_team[s]][span] - act[s][span]).max(initial=0) >= _cap(slot_team[s], P):
                continue                                            # the other slots already fill the team there
            lo, hi = np.searchsorted(ks, tl.start), np.searchsorted(ks, tl.end, side="right")
            if hi > lo:                                             # slot has points inside the span
                continue
            prev = int(ks[lo - 1]) if lo > 0 else None
            nxt = int(ks[lo]) if lo < len(ks) else None
            gaps = [tl.start - prev if prev is not None else np.inf, nxt - tl.end if nxt is not None else np.inf]
            if min(gaps) > P.attach_gap_s * data.rate:
                continue
            score, ok = 0.0, True
            for side, other in ((0, prev), (1, nxt)):
                if other is None:
                    continue
                a, b = ({other: tr[other]}, _end_point(tl, False)) if side == 0 else (_end_point(tl, True), {other: tr[other]})
                if not reachable(a, b, data, P.vmax, P.slack):
                    ok = False
                    break
                if gaps[side] <= P.attach_gap_s * data.rate:
                    pa, pb = next(iter(a.values())).xy, next(iter(b.values())).xy
                    dt = gaps[side] / data.rate
                    score = max(score, float(np.linalg.norm(pb - pa)) - (1.0 + 4.0 * dt))
            if ok and score < best_score:
                best, best_score = s, score
        if best is not None and best_score <= 0.0:
            slots[best].update(tl.tr)
            new = _active(slots[best], data.n + 1, max_gap)
            busy[slot_team[best]] += new - act[best]
            act[best] = new
            owner[best].update({k: -1 - q for k in tl.tr})
            keys[best] = np.array(sorted(slots[best]), int)
            n += 1
    return n


def _cap(team: str, P: ClosedParams) -> int:
    return P.on_court * (len(TEAMS) if team == "*" else 1)


def _active(tr: Track, n: int, max_gap: int) -> np.ndarray:
    """0/1 per frame: the track has a point there or will get one from fill_gaps (gap <= max_gap)."""
    a = np.zeros(n, int)
    ks = np.array(sorted(k for k in tr if 0 <= k < n), int)
    if not len(ks):
        return a
    a[ks] = 1
    for lo, hi in zip(ks, ks[1:]):
        if 1 < hi - lo <= max_gap:
            a[lo:hi] = 1
    return a


def _fill_capped(slots: list[Track], slot_team: list[str], is_spare: list[bool], data: Data,
                 P: ClosedParams) -> list[Track]:
    """B6 fill_gaps per slot, then drop interpolated (box-less) points where a team would exceed the on-court cap:
    a slot's gap can coincide with another slot's tracklet, and filling both would show 6 players. Spare slots
    give way first."""
    if P.refuse:
        slots = [{k: TrackPoint(fuse(data, p.boxes)[0], p.boxes) if len(p.boxes) > 1 else p for k, p in s.items()}
                 for s in slots]
    filled = [fill_gaps(s, data.rate, P.fill_gap_s, P.vmax) if s else {} for s in slots]
    if P.smooth:
        filled = [savgol_smooth(f) for f in filled]
    for tm in set(slot_team):
        idx = [q for q in range(len(slots)) if slot_team[q] == tm]
        cnt = np.zeros(data.n + 1, int)
        for q in idx:
            ks = np.array([k for k in filled[q] if 0 <= k <= data.n], int)
            cnt[ks] += 1
        for k in np.flatnonzero(cnt > _cap(tm, P)):
            for q in sorted(idx, key=lambda q: not is_spare[q]):
                if cnt[k] <= _cap(tm, P):
                    break
                if k in filled[q] and not filled[q][k].boxes:
                    del filled[q][k]
                    cnt[k] -= 1
    return filled


def _junctions(tr: Track, own: dict[int, int]) -> list[tuple[int, int]]:
    """Consecutive samples (a, b) of a slot track that come from different source tracklets."""
    ks = sorted(tr)
    return [(a, b) for a, b in zip(ks, ks[1:]) if own[a] != own[b]]


def _reach_violations(slots: list[Track], owner: list[dict[int, int]], data: Data, P: ClosedParams) -> int:
    """Hand-overs between two source tracklets inside one slot that fail B3 (the MILP should make this 0)."""
    return sum(1 for tr, own in zip(slots, owner) for a, b in _junctions(tr, own)
               if not reachable({a: tr[a]}, {b: tr[b]}, data, P.vmax, P.slack))


def _exact_5v5(slots: list[Track], slot_team: list[str], n: int) -> float:
    cnt = {tm: np.zeros(n, int) for tm in set(slot_team)}
    for tr, tm in zip(slots, slot_team):
        ks = np.array([k for k in tr if 0 <= k < n], int)
        cnt[tm][ks] += 1
    ok = np.ones(n, bool)
    for tm, c in cnt.items():
        ok &= c == (5 if len(cnt) == 2 else 10)
    return round(float(ok.mean()), 3)


def link_closed(tracklets: list[Track], data: Data, params: ClosedParams | None = None,
                stats: ClosedStats | None = None) -> list[Track]:
    """Closed-set assignment of any tracklet source (see module docstring)."""
    P = params or ClosedParams()
    st = stats if stats is not None else ClosedStats()
    T = [t for t in tracklets if t]
    if P.jump_split:
        T = split_jumps(T, data, P)
    if P.dbscan_eps:
        T = [piece for t in T for piece in dbscan_split(t, data, eps=P.dbscan_eps, min_samples=15)]
    if P.team_split:
        T = split_team_flips(T, data)
    if P.merge_dups:
        T, st.merged_dups = merge_duplicates(T, data, P)
    clean = clean_masks(data)
    tls = []
    for tr in T:
        ks = np.array(sorted(tr), int)
        tls.append(_Tl(tr, ks, team_vote(tr, data)[0], None))
    long_ = [tl for tl in tls if len(tl) >= P.min_len_s * data.rate]
    short = [tl for tl in tls if len(tl) < P.min_len_s * data.rate]
    for tl in long_:
        tl.emb = track_embedding(tl.tr, data, clean)
    st.tracklets, st.solved, st.short_total = len(tls), len(long_), len(short)
    exclusive = []
    if P.exclude_dups:
        exclusive = [(a, b) for a, b, _ in
                     duplicate_pairs([t.tr for t in long_], [t.team for t in long_], P, P.excl_cover)]
    st.exclusive_pairs = len(exclusive)
    assignment = assign(long_, data, P, st, exclusive)
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
    st.attached_short = _attach_short(slots, owner, slot_team, short, data, P)
    total = sum(len(tl) for tl in tls)
    st.assigned_samples_share = round(sum(len(s) for s in slots) / max(total, 1), 4)
    st.exact_5v5_slots = _exact_5v5(slots, slot_team, data.n)
    filled = _fill_capped(slots, slot_team, st.is_spare, data, P)
    st.exact_5v5_filled = _exact_5v5(filled, slot_team, data.n)
    out = [f for f in filled if f]
    st.reach_violations = _reach_violations(slots, owner, data, P)
    st.junctions = [(q, a, b) for q, (tr, own) in enumerate((x for x in zip(slots, owner) if x[0]))
                    for a, b in _junctions(tr, own)]          # (index into the output, last frame, first frame)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Registered variants
# ---------------------------------------------------------------------------------------------------------------

def source_tracklets(data: Data, source: str) -> list[Track]:
    """'fused' (pitch_tracker tracklets, ambiguity 0.3) or an xview source ('conservative', 'hybridsort', ...)."""
    if source == "fused":
        return fused_tracklets(data, 0.3)
    from xview import xview
    return xview(data, source)


LAST_STATS: dict[str, ClosedStats] = {}


def _run(data: Data, name: str, source: str, **kw) -> list[Track]:
    st = ClosedStats()
    out = link_closed(source_tracklets(data, source), data, ClosedParams(**kw), st)
    LAST_STATS[name] = st
    print(f"[{name}] tracklets {st.tracklets} solved {st.solved} assigned {st.assigned} unassigned_long "
          f"{st.unassigned_long} ({st.unassigned_long_samples} samples) short attached {st.attached_short}/"
          f"{st.short_total} sample share {st.assigned_samples_share} | vars {st.n_vars} rows {st.n_rows} "
          f"solver {st.solver_s} s changes {st.changes} | exact 5v5 {st.exact_5v5_slots} filled "
          f"{st.exact_5v5_filled} | spare samples {st.spare_samples} tracklets {st.spare_tracklets} | "
          f"reach violations {st.reach_violations} | merged dups {st.merged_dups} exclusive {st.exclusive_pairs} "
          f"unreachable {st.unreachable_pairs} rough {st.rough_pairs} links {st.links} unpinned {st.unpinned} | "
          f"seeds {st.seeds}", file=sys.stderr)
    return out


VARIANTS = {
    "lc_fused": ("fused", {}),                                      # (a) per-frame fused tracklets
    "lc_xvcons": ("conservative", {}),                              # (b) cross-view conservative tracklets
    "lc_xvhyb": ("hybridsort", {}),                                 # (c) cross-view HybridSort tracklets
    # positions as linked, no re-fusion / smoothing: comparable with the (unsmoothed) baseline
    "lc_fused_raw": ("fused", {"refuse": False, "smooth": False}),
    "lc_xvcons_raw": ("conservative", {"refuse": False, "smooth": False}),
    # ablations on (b)
    "lc_xvcons_noteam": ("conservative", {"use_team": False}),
    "lc_xvcons_nolink": ("conservative", {"link_w": 0.0, "cv_gate": None}),
    "lc_xvcons_court6": ("conservative", {"on_court": 6}),
}

for _name, (_src, _kw) in VARIANTS.items():
    register(_name)(lambda data, _n=_name, _s=_src, _k=_kw: _run(data, _n, _s, **_k))
