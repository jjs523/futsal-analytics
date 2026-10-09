"""Targeted linker-input fixes for the dominant swap cause that crossfix cannot reach (experiments/audit/forensics.json).

Forensics of the 10 audited SWAP crossings: 9/10 come from the linker, 6/10 from attaching short pieces that have no
clean crops (final_d: link_closed._attach_short chains of 1-7 frame pieces across 1-2.3 s gaps; final_b:
link_gta._attach_debris chains of 1-frame debris), 2/10 (final_d) from hand-over overlaps: the same player's
continuation in the other camera starts 1-5 frames before his piece ends, so the MILP's no-overlap rows push it into
another slot. crossfix only swaps tails of two long tracks with clean crops on both sides, so it cannot fix either.

Fixes (both only change what the linkers are given / allowed to attach; nothing in src/ or the linker modules is
edited, the functions are swapped in for the duration of one run like combine.closed_top_exempt):

  handover_merge   before link_closed: merge tracklet A into B when B starts 1-5 frames before A ends, the shared
                   frames come from different cameras and lie within 1 m, the teams agree, the pair is the only
                   candidate for A's end and for B's start, both pieces are MILP inputs (>= 1 s), and the SSL head on
                   >= 3 clean crops each side (A's last 3 s, B's first 3 s) says one person (cos >= 0.9).
                   27 of 95 candidate pairs merge. Looser gates (cos 0.7, any length, 1 crop: 48 merges) also merged
                   short pieces and perturbed the MILP: a new swap-like change on the audited OK sheet final_d ev19.
  strict_attach    link_closed short pieces: judged against the slot's LONG (MILP-assigned) points only, so attached
                   pieces never extend a chain; the nearer long point must be within 0.5 s (was 2 s against any
                   point) and the best slot must beat every other feasible slot by 1 m of junction excess.
                   Keeping chains and only adding the margin (sa_long_only=False) changed almost nothing; 1.0 s
                   instead of 0.5 s kept fewer of the audited fixes.
  strict_debris    link_gta debris: all debris is scored against the clusters as they were before any debris joined
                   (no chains through freshly attached debris), with debris_gap_s 1.5 -> 0.5 s and a 2-sigma margin.

Variants: final_d2 = both final_d fixes, final_d2h = handover_merge only, final_d2s = strict_attach only,
final_b2 = strict_debris on final_b; <name>_cf / _cfall add crossfix (first-half / whole-window head).
Crop-ownership proxy on the audited sheets (the crops a rater saw, followed into the new result; not a re-audit):
  final_d2   the wrong crops of d ev08 / ev14 / ev20 (and half of ev13) are no longer in the same track; ev01 and
             ev23 unchanged; 1 audited OK sheet changed (ev19: B's +1 s crop, a short piece, now in another slot).
             Cost: count_5v5 0.866 / 0.816 -> 0.806 / 0.727 (gaps the chains used to bridge stay open).
  final_d2h  best label-free numbers (count_5v5 0.903 / 0.847, top10 0.982 / 0.951, dup_pairs 3 / 4), 0 OK sheets
             changed, but only d ev14 of the 6 SWAP sheets changes.
  final_b2   the debris crops of b ev07 / ev22 / ev23 are dropped from the tracks (3 of 4 SWAP sheets), 0 OK sheets
             changed. b ev05 (a per-camera relink error) stays, and crossfix no longer finds it: in final_b the
             encounter existed only because of the debris at the start of track 12. Per-camera relink max_cost
             0.6 -> 0.45 did not fix ev05 and cost 2 ids / 0.08 top10.
  *_cf on final_d2 / d2s: crossfix makes one one-sided tail swap (10:49.5 to the window end) that raises
             reid_dispersion 0.136 -> 0.148 on the first half: do not use without a crop check.

    python experiments/harness.py final_d2 fix2
    python experiments/harness.py final_b2 fix2
    python experiments/metrics2.py final_d final_d2 final_d2h final_b final_b2 --parts first,second
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from harness import Data, Track, TrackPoint, register
from blocks import fuse, reachable, team_vote

RESULTS = os.path.join(harness.ROOT, "experiments", "results")


@dataclass(frozen=True)
class Fix2Params:
    handover: bool = True
    ho_max_overlap: int = 5           # frames B may start before A ends
    ho_gate_m: float = 1.0            # every shared frame within this distance
    ho_win_s: float = 3.0             # crop windows: A's last / B's first seconds
    ho_min_sim: float = 0.9           # SSL head cosine of the two windows (same ~0.97-0.99, other ~-0.2 .. 0.3)
    ho_min_len_s: float = 1.0         # both pieces at least this long (= the MILP inputs of link_closed)
    ho_min_crops: int = 3             # clean crops needed in each window
    ho_head: str = "first"            # 'first' (held-out second half, like crossfix *_cf) or 'all' (production)
    strict_attach: bool = True
    sa_near_s: float = 0.5            # a short piece needs a long point of the slot within this gap
    sa_margin_m: float = 1.0          # ... and must fit that slot this much better than any other feasible one
    sa_long_only: bool = True         # neighbours = the slot's MILP (long) points only; False: any point (chains allowed)
    strict_debris: bool = False       # link_gta only
    sd_gap_s: float = 0.5
    sd_margin: float = 2.0            # sigmas


STATS: dict = {}


# ---------------------------------------------------------------------------------------------------------------
# Hand-over overlap merge (tracklet input of link_closed)
# ---------------------------------------------------------------------------------------------------------------

def handover_pairs(T: list[Track], data: Data, feats, clean, p: Fix2Params) -> list[dict]:
    """Candidate (A, B) hand-overs with their evidence; 'ok' marks the ones to merge."""
    from crossfix import window_emb
    teams = [team_vote(t, data)[0] for t in T]
    ks = [np.array(sorted(t), int) for t in T]
    by_start: dict[int, list[int]] = {}
    for q, k in enumerate(ks):
        by_start.setdefault(int(k[0]), []).append(q)
    win = int(round(p.ho_win_s * data.rate))
    out = []
    for a, A in enumerate(T):
        sa, ea = int(ks[a][0]), int(ks[a][-1])
        for ov in range(1, p.ho_max_overlap + 1):
            for b in by_start.get(ea - ov + 1, []):
                if b == a or ks[b][-1] <= ea or ks[b][0] <= sa:
                    continue
                if {teams[a], teams[b]} == {"Y", "N"}:
                    continue
                B = T[b]
                sh = [k for k in range(ea - ov + 1, ea + 1) if k in A and k in B]
                if not sh or any({c for c, _ in A[k].boxes} & {c for c, _ in B[k].boxes} for k in sh):
                    continue
                d = max(float(np.linalg.norm(A[k].xy - B[k].xy)) for k in sh)
                if d > p.ho_gate_m:
                    continue
                ea_, na = window_emb(A, ea - win, ea, feats, clean)
                eb_, nb = window_emb(B, int(ks[b][0]), int(ks[b][0]) + win, feats, clean)
                sim = float(ea_ @ eb_) if na >= p.ho_min_crops and nb >= p.ho_min_crops else None
                out.append({"a": a, "b": b, "k": ea, "overlap": ov, "d": round(d, 2), "sim": sim, "na": na, "nb": nb,
                            "long": min(len(A), len(B)) >= p.ho_min_len_s * data.rate})
    # uniqueness: one candidate for A's end and one for B's start (counted before the appearance gate)
    na_ = {}
    nb_ = {}
    for r in out:
        na_[r["a"]] = na_.get(r["a"], 0) + 1
        nb_[r["b"]] = nb_.get(r["b"], 0) + 1
    for r in out:
        r["unique"] = na_[r["a"]] == 1 and nb_[r["b"]] == 1
        r["ok"] = bool(r["unique"] and r["long"] and r["sim"] is not None and r["sim"] >= p.ho_min_sim)
    return out


def handover_merge(T: list[Track], data: Data, p: Fix2Params) -> tuple[list[Track], list[dict]]:
    import crossfix
    feats = crossfix.head(data, p.ho_head)
    clean = crossfix.crop_mask(data, crossfix.TUNED)
    T = [t for t in T if t]
    pairs = handover_pairs(T, data, feats, clean, p)
    parent = list(range(len(T)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    groups = {i: dict(t) for i, t in enumerate(T)}
    for r in sorted((r for r in pairs if r["ok"]), key=lambda r: r["k"]):
        ra, rb = root(r["a"]), root(r["b"])
        if ra == rb:
            r["ok"] = False
            continue
        A, B = groups[ra], groups[rb]
        if any({c for c, _ in A[k].boxes} & {c for c, _ in B[k].boxes} for k in A.keys() & B.keys()):
            r["ok"] = False
            continue
        for k, pt in B.items():
            if k in A:
                boxes = sorted(A[k].boxes + pt.boxes)
                A[k] = TrackPoint(fuse(data, boxes)[0], boxes)
            else:
                A[k] = pt
        parent[rb] = ra
        del groups[rb]
    return [dict(sorted(g.items())) for g in groups.values()], pairs


# ---------------------------------------------------------------------------------------------------------------
# Strict short-piece attachment (link_closed)
# ---------------------------------------------------------------------------------------------------------------

def make_strict_attach(near_s: float, margin_m: float, log: list | None = None, long_only: bool = True):
    import link_closed as lc

    def _attach_short(slots, owner, slot_team, shorts, data, P):
        keys = [np.array(sorted(s), int) for s in slots]
        long_keys = [np.array(sorted(k for k, o in own.items() if o >= 0), int) for own in owner] if long_only else keys
        max_gap = int(P.fill_gap_s * data.rate)
        near = near_s * data.rate
        busy = {tm: np.zeros(data.n + 1, int) for tm in set(slot_team)}
        act = [lc._active(tr, data.n + 1, max_gap) for tr in slots]
        for a, tm in zip(act, slot_team):
            busy[tm] += a
        n = rejected_amb = rejected_far = 0
        for q, tl in sorted(enumerate(shorts), key=lambda x: -len(x[1])):
            cands = []
            any_old = False
            for s, tr in enumerate(slots):
                if P.use_team and tl.team != "U" and slot_team[s] != tl.team:
                    continue
                ks, lk = keys[s], long_keys[s]
                if not len(ks) or not len(lk):
                    continue
                span = slice(max(tl.start, 0), tl.end + 1)
                if (busy[slot_team[s]][span] - act[s][span]).max(initial=0) >= lc._cap(slot_team[s], P):
                    continue
                lo, hi = np.searchsorted(ks, tl.start), np.searchsorted(ks, tl.end, side="right")
                if hi > lo:
                    continue
                # reachability against the actual neighbours (no speed violation in the output)
                prev = int(ks[lo - 1]) if lo > 0 else None
                nxt = int(ks[lo]) if lo < len(ks) else None
                ok = True
                for side, other in ((0, prev), (1, nxt)):
                    if other is None:
                        continue
                    a, b = ({other: tr[other]}, lc._end_point(tl, False)) if side == 0 else \
                        (lc._end_point(tl, True), {other: tr[other]})
                    if not reachable(a, b, data, P.vmax, P.slack):
                        ok = False
                        break
                if not ok:
                    continue
                # old rule's gap test (for the log): any point within attach_gap_s
                g_any = min(tl.start - prev if prev is not None else np.inf, nxt - tl.end if nxt is not None else np.inf)
                any_old |= g_any <= P.attach_gap_s * data.rate
                # new rule: the slot's LONG points only (attached pieces never extend a chain)
                l0, l1 = np.searchsorted(lk, tl.start), np.searchsorted(lk, tl.end, side="right")
                lp = int(lk[l0 - 1]) if l0 > 0 else None
                ln = int(lk[l1]) if l1 < len(lk) else None
                gaps = [tl.start - lp if lp is not None else np.inf, ln - tl.end if ln is not None else np.inf]
                if min(gaps) > near:
                    continue
                score = 0.0
                for side, other in ((0, lp), (1, ln)):
                    if other is None or gaps[side] > near:
                        continue
                    pa = tr[other].xy if side == 0 else tl.tr[tl.end].xy
                    pb = tl.tr[tl.start].xy if side == 0 else tr[other].xy
                    dt = gaps[side] / data.rate
                    score = max(score, float(np.linalg.norm(pb - pa)) - (1.0 + 4.0 * dt))
                if score <= 0.0:
                    cands.append((score, s))
            cands.sort()
            if not cands:
                rejected_far += any_old
                continue
            if len(cands) > 1 and cands[1][0] < cands[0][0] + margin_m:
                rejected_amb += 1
                continue
            best = cands[0][1]
            slots[best].update(tl.tr)
            new = lc._active(slots[best], data.n + 1, max_gap)
            busy[slot_team[best]] += new - act[best]
            act[best] = new
            owner[best].update({k: -1 - q for k in tl.tr})
            keys[best] = np.array(sorted(slots[best]), int)
            n += 1
        if log is not None:
            log.append({"attached": n, "rejected_ambiguous": rejected_amb, "rejected_far_old_ok": rejected_far,
                        "shorts": len(shorts), "attached_points": None})
        return n
    return _attach_short


@contextlib.contextmanager
def strict_attach(enabled: bool, near_s: float, margin_m: float, log: list | None = None, long_only: bool = True):
    if not enabled:
        yield
        return
    import link_closed as lc
    orig = lc._attach_short
    lc._attach_short = make_strict_attach(near_s, margin_m, log, long_only)
    try:
        yield
    finally:
        lc._attach_short = orig


# ---------------------------------------------------------------------------------------------------------------
# Strict debris attachment (link_gta)
# ---------------------------------------------------------------------------------------------------------------

def _attach_debris_strict(self, debris: list[int]) -> None:
    """link_gta.Linker._attach_debris, but every debris piece is scored against the clusters as they were before
    any debris joined (so debris never chains through debris), gap <= sd_gap_s, rival sd_margin sigmas worse."""
    p = self.p
    fp: Fix2Params = self._fix2
    near = int(np.ceil(fp.sd_gap_s * self.rate))
    pending = set(debris)
    snap = {c: C for c, C in self.clusters.items() if c not in pending}
    decisions = []
    for d in sorted(debris, key=lambda i: -len(self.clusters[i].ks)):
        Dc = self.clusters[d]
        lo, hi = Dc.ks[0] - near, Dc.ks[-1] + near
        scored = []
        for c, C in snap.items():
            if C.ks[0] > hi or C.ks[-1] < lo:
                continue
            r = self.evaluate(Dc, C)
            if not r.hard and r.gap <= fp.sd_gap_s and r.z <= p.noapp_z:
                scored.append((r.z, c))
        scored.sort()
        if scored and (len(scored) == 1 or scored[1][0] >= scored[0][0] + fp.sd_margin):
            decisions.append((d, scored[0][1]))
    cur = {c: c for c in snap}
    for d, c in decisions:
        tgt = cur[c]
        r = self.evaluate(self.clusters[d], self.clusters[tgt])
        if r.hard:                                  # two debris pieces of one camera landed on the same frame
            continue
        cur[c] = self._absorb(d, tgt)
        self.st.debris_attached += 1


@contextlib.contextmanager
def strict_debris(enabled: bool, fp: Fix2Params):
    if not enabled:
        yield
        return
    import link_gta as lg
    orig = lg.Linker._attach_debris
    lg.Linker._attach_debris = _attach_debris_strict
    lg.Linker._fix2 = fp
    try:
        yield
    finally:
        lg.Linker._attach_debris = orig
        del lg.Linker._fix2


# ---------------------------------------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------------------------------------

def run_d(data: Data, fp: Fix2Params, name: str) -> list[Track]:
    """final_d's pipeline (xview conservative -> link_closed, tuned ClosedParams, cam2 top exemption, raw OSNet)
    with the fixes."""
    import combine
    from link_closed import ClosedStats, link_closed
    T = combine.source_tracklets(data, "xv_conservative")
    info: dict = {"name": name, "params": fp.__dict__ if hasattr(fp, "__dict__") else str(fp)}
    if fp.handover:
        n0 = len(T)
        T, pairs = handover_merge(T, data, fp)
        info["handover"] = {"candidates": len(pairs), "merged": sum(r["ok"] for r in pairs),
                            "no_crops": sum(r["sim"] is None for r in pairs),
                            "low_sim": sum(r["sim"] is not None and r["sim"] < fp.ho_min_sim for r in pairs),
                            "not_unique": sum(not r["unique"] for r in pairs), "tracklets": (n0, len(T)),
                            "merged_at_k": [r["k"] for r in pairs if r["ok"]]}
    st = ClosedStats()
    log: list = []
    with combine.closed_top_exempt(True), strict_attach(fp.strict_attach, fp.sa_near_s, fp.sa_margin_m, log,
                                                                         fp.sa_long_only):
        out = link_closed(T, data, combine._closed_params(), st)
    info["closed"] = {"tracklets": st.tracklets, "solved": st.solved, "assigned": st.assigned,
                      "short_attached": st.attached_short, "short_total": st.short_total,
                      "share": st.assigned_samples_share, "exact5v5": (st.exact_5v5_slots, st.exact_5v5_filled),
                      "reach_viol": st.reach_violations, "merged_dups": st.merged_dups, "unpinned": st.unpinned,
                      "attach_log": log}
    STATS[name] = info
    print(f"[fix2] {name} {info}", file=sys.stderr)
    return out


def run_b(data: Data, fp: Fix2Params, name: str) -> list[Track]:
    """final_b's pipeline (relink source -> link_gta, min_len 5 s) with strict debris attachment."""
    import combine
    from link_gta import GtaStats, link
    T = combine.source_tracklets(data, "relink")
    st = GtaStats()
    with strict_debris(fp.strict_debris, fp):
        out = link(T, data, combine._gta_params(), st)
    STATS[name] = {"debris_attached": st.debris_attached, "outputs": st.outputs, "dropped_points": st.dropped_points}
    print(f"[fix2] {name} debris attached {st.debris_attached} outputs {st.outputs} dropped_points "
          f"{st.dropped_points} merges app {st.merges_app} motion {st.merges_motion}", file=sys.stderr)
    return out


D2 = Fix2Params()
VARIANTS = {
    "final_d2": lambda data: run_d(data, D2, "final_d2"),                                       # both fixes
    "final_d2h": lambda data: run_d(data, replace(D2, strict_attach=False), "final_d2h"),       # hand-over merge only
    "final_d2s": lambda data: run_d(data, replace(D2, handover=False), "final_d2s"),            # strict attach only
    "final_b2": lambda data: run_b(data, Fix2Params(handover=False, strict_attach=False, strict_debris=True), "final_b2"),
}
for _n, _f in VARIANTS.items():
    register(_n)(_f)


def _cf(data: Data, base: str, window: str) -> list[Track]:
    """crossfix (tuned, crossfix.TUNED) on a saved fix2 result; logs to results/<base>_cf.crossfix_log.json."""
    import crossfix
    if not os.path.exists(os.path.join(RESULTS, f"{base}.json")):
        raise FileNotFoundError(f"run `python experiments/harness.py {base} fix2` first")
    return crossfix.run(data, base, window, out_name=f"{base}_{'cf' if window == 'first' else 'cfall'}")


for _b in ("final_d2", "final_d2h", "final_d2s", "final_b2"):
    register(f"{_b}_cf")(lambda data, _b=_b: _cf(data, _b, "first"))
    register(f"{_b}_cfall")(lambda data, _b=_b: _cf(data, _b, "all"))
