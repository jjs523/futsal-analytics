"""Crossing re-assignment post-processor for ANY final tracks (list[Track]).

Where two (or three) same-team tracks come close, the linker / tracker had to decide who is who after the
encounter with little evidence; the blind visual audit showed this is where the remaining identity swaps live
(final_d 6/20, final_b 4/22 decided crossings). The per-match SSL head (ssl_embed) separates teammates on clean crops
far better than raw OSNet (same-team AUC 0.999 vs 0.909 held out), so it is used here LOCALLY:

  1. encounters: every pair of same-team tracks (clear team vote, both 'Y' or both 'N') closer than `enc_m` metres,
     close frames less than `merge_s` apart form one run; runs longer than `max_len_s` (two tracks moving together:
     duplicates, not crossings) are skipped. Runs that overlap in time and share a track form one group, so a
     3-player cluster is decided jointly.
  2. evidence: clean crops (blocks.clean_masks: big, confident, IoU < 0.1 with every other box, off the image
     border; optionally cam2's top-cut crops too) of each track in [k0 - 3 s, k0 - 0.5 s] (pre) and
     [k1 + 0.5 s, k1 + 3 s] (post), where [k0, k1] is the group's close run; a window with < min_crops crops may
     widen its far end to `ext_s`. Both windows stop at the neighbouring group of either track, so every window only
     ever holds frames between two decisions. Window embedding = L2-normalised mean of the head vectors.
  3. decision: for a pair, same = cos(A_pre, A_post) + cos(B_pre, B_post), cross = cos(A_pre, B_post) +
     cos(B_pre, A_post); swap when cross - same > margin. A track born (or lost) inside the encounter has no pre (post)
     window: with min_terms=1 the decision may rest on the terms that exist (A_pre vs A_post and B_post), rescaled
     to a pair's two terms. A 3-player group tries every permutation of the post segments (gain scaled by the
     number of terms / 2), greedily over pairs in groups of more than `max_perm` tracks. Pairs whose pre (or post)
     windows look alike (> dup_sim) are one person tracked twice and never swap.
     A swap also needs motion continuity both ways: blocks.reachable from the last point before the run to the
     first point after it for every new junction, and the constant-velocity extrapolation error of the new pairing
     no more than `motion_slack_m` worse than that of the old one.
  4. swap: from the closest-approach frame c the post-encounter segments are exchanged until the next group of
     either track ("until the next encounter"; later groups are relabelled and decide again on the corrected
     identities) - unless the head sees the linker hand the right player back earlier (`segment_end`, 2 s chunks):
     then the swap ends there. Without that, a linker that took the wrong player only until its next junction (the
     common case: final_d 10:44 was wrong for ~5 s) would get the fix carried into minutes that were right.

Safety: points are moved, never copied or dropped, so no frame gets two points in one track and no box is used twice
more than before; only same-team tracks exchange tails and the moved tail must not vote for the other team;
every group decision (swap or not, and why) is logged. Idempotent: on its own output every applied swap scores
-(cross - same) < -margin, and windows never straddle a decision of the same track (checked by `check_idempotent`).

Registered (head trained on the first half only, so second-half numbers are held out; *_cfall: head trained on the
whole window, the production setting):
    python experiments/harness.py final_d_cf crossfix          # also final_d_cfall, final_b_cf, final_b_cfall
    python experiments/crossfix.py tune                          # margin / motion grid on the first half
    python experiments/crossfix.py report                        # metrics2 + audited events, writes crossfix_report.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from harness import ROOT, Data, Track, TrackPoint, register
from blocks import clean_masks, normalise, reachable, team_vote

RESULTS = os.path.join(ROOT, "experiments", "results")


@dataclass(frozen=True)
class CrossParams:
    enc_m: float = 1.5                # closest approach below this = encounter
    merge_s: float = 1.0              # close frames less than this apart are one encounter
    max_len_s: float = 5.0            # longer close runs are tracks moving together, not crossings
    pre_s: tuple = (3.0, 0.5)         # pre window [k0 - 3 s, k0 - 0.5 s]
    post_s: tuple = (0.5, 3.0)        # post window [k1 + 0.5 s, k1 + 3 s]
    min_crops: int = 2                # clean crops needed in every window that is compared
    margin: float = 0.2               # swap when cross - same > margin (tuned on the first half)
    motion_slack_m: float = 3.0       # new pairing's CV extrapolation error may exceed the old one by this much
    vel_s: float = 1.0                # velocity / reachability support on each side of the run
    vmax: float = 8.0                 # blocks.reachable
    slack: float = 1.0
    max_perm: int = 3                 # groups up to this size: all permutations; larger: greedy pairs
    top_exempt: bool = False          # cam2 crops cut at the top edge count as clean (link_gta.clean_crops)
    crop_min_h: float = 60.0          # blocks.clean_masks min_h / max_iou (the head was trained on 60 / 0.1)
    crop_max_iou: float = 0.1
    ext_s: float = 0.0                # if a window has < min_crops, widen its far end up to this many seconds
    dup_sim: float = 0.7              # two tracks whose pre (or post) windows look alike are one person twice: no swap
    break_sim: float = 0.5            # log a track whose own pre/post similarity is below this (identity break)
    min_terms: int = 2                # evidence terms S[t, new tail] - S[t, t] needed (1 = one-sided decisions allowed)
    segment: bool = True              # end the swap where the head sees the linker hand back (segment_end)
    seg_chunk_s: float = 2.0
    seg_unknown: float = 0.4          # similarity assumed for a chunk side without crops (same ~1, other ~-0.2)
    seg_missing_m: float = 3.0        # hand-back cost of a junction side without a point


@dataclass
class Group:
    k0: int
    k1: int
    tracks: list[int]                         # current track indices
    pairs: list[tuple[int, int, int, float]]  # (i, j, kc, dmin) of the encounters in the group


# ---------------------------------------------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------------------------------------------

def _positions(tracks: list[Track], n: int) -> np.ndarray:
    P = np.full((len(tracks), n, 2), np.nan)
    for q, t in enumerate(tracks):
        for k, p in t.items():
            if 0 <= k < n:
                P[q, k] = p.xy
    return P


def encounters(tracks: list[Track], teams: list[str], data: Data, p: CrossParams) -> list[tuple[int, int, int, int, int, float]]:
    """(k0, k1, kc, i, j, dmin) for every close run of two same-team tracks (kc = closest approach)."""
    P = _positions(tracks, data.n)
    out = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if teams[i] != teams[j] or teams[i] not in ("Y", "N"):
                continue
            d = np.linalg.norm(P[i] - P[j], axis=1)
            close = np.flatnonzero(d < p.enc_m)                      # NaN compares False
            if not len(close):
                continue
            for run in np.split(close, np.flatnonzero(np.diff(close) > p.merge_s * data.rate) + 1):
                if run[-1] - run[0] + 1 > p.max_len_s * data.rate:
                    continue
                kc = int(run[np.argmin(d[run])])
                out.append((int(run[0]), int(run[-1]), kc, i, j, float(d[kc])))
    return sorted(out)


def groups_of(enc: list[tuple], pad: int = 0) -> list[Group]:
    """Union of encounters whose runs overlap in time (within `pad`) and share a track."""
    groups: list[Group] = []
    for k0, k1, kc, i, j, dm in enc:
        hit = [g for g in groups if k0 <= g.k1 + pad and k1 >= g.k0 - pad and ({i, j} & set(g.tracks))]
        if not hit:
            groups.append(Group(k0, k1, [i, j], [(i, j, kc, dm)]))
            continue
        g = hit[0]
        for h in hit[1:]:
            g.k0, g.k1 = min(g.k0, h.k0), max(g.k1, h.k1)
            g.tracks = sorted(set(g.tracks) | set(h.tracks))
            g.pairs += h.pairs
            groups.remove(h)
        g.k0, g.k1 = min(g.k0, k0), max(g.k1, k1)
        g.tracks = sorted(set(g.tracks) | {i, j})
        g.pairs.append((i, j, kc, dm))
    return sorted(groups, key=lambda g: (g.k0, g.k1))


# ---------------------------------------------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------------------------------------------

def crop_mask(data: Data, p: CrossParams) -> dict[str, np.ndarray]:
    """Which boxes give a crop of one whole person: blocks.clean_masks (IoU < crop_max_iou with every other box of the
    frame, big and confident enough, off the image border), optionally without cam2's top-border rule."""
    if not p.top_exempt:
        return clean_masks(data, min_h=p.crop_min_h, max_iou=p.crop_max_iou)
    free = clean_masks(data, min_h=p.crop_min_h, max_iou=p.crop_max_iou, border=0.0)
    m = 0.02 * 1920
    out = {}
    for cam, c in data.cams.items():
        x = c.xyxy
        edge = (x[:, 0] >= m) & (x[:, 2] <= 1920 - m) & (x[:, 3] <= 1080 - m)
        if cam != "cam2":
            edge &= x[:, 1] >= m
        out[cam] = free[cam] & edge
    return out


def window_emb(track: Track, lo: int, hi: int, feats: dict[str, np.ndarray], clean: dict[str, np.ndarray]) -> tuple[np.ndarray | None, int]:
    """Normalised mean head vector of the track's clean crops in frames [lo, hi], and the crop count."""
    rows = [feats[c][i] for k in range(lo, hi + 1) if k in track for c, i in track[k].boxes if clean[c][i]]
    if not rows:
        return None, 0
    return normalise(np.mean(rows, 0)), len(rows)


def _side(track: Track, lo: int, hi: int, rate: float, last: bool):
    """(k, xy, velocity, piece) at the end (last=True) / start of the track's points in [lo, hi]; None if empty."""
    ks = [k for k in range(lo, hi + 1) if k in track]
    if not ks:
        return None
    ks = np.array(ks)
    X = np.stack([track[int(k)].xy for k in ks])
    v = np.zeros(2)
    if len(ks) >= 2:
        t = (ks - ks.mean()) / rate
        v = (t[:, None] * (X - X.mean(0))).sum(0) / max((t ** 2).sum(), 1e-9)
    k = int(ks[-1] if last else ks[0])
    return k, track[k].xy, v, {k: track[k]}


def _cv_err(a, b, rate: float, vmax: float) -> float:
    """Mean of forward (a's end extrapolated with its velocity) and backward (b's start with its velocity) errors."""
    ka, xa, va, _ = a
    kb, xb, vb, _ = b
    dt = (kb - ka) / rate
    cl = lambda v: v * min(1.0, vmax / max(float(np.linalg.norm(v)), 1e-9))
    f = float(np.linalg.norm(xa + cl(va) * dt - xb))
    g = float(np.linalg.norm(xb - cl(vb) * dt - xa))
    return 0.5 * (f + g)


# ---------------------------------------------------------------------------------------------------------------
# The post-processor
# ---------------------------------------------------------------------------------------------------------------

def _swap_tails(tracks: list[Track], perm: dict[int, int], c: int, e: int | None = None) -> None:
    """In place: new track t = old t before frame c + old perm[t] in [c, e) + old t from e on (e=None: to the end)."""
    old = {q: tracks[q] for q in set(perm) | set(perm.values())}
    for t, s in perm.items():
        mid = {k: p for k, p in old[s].items() if k >= c and (e is None or k < e)}
        keep = {k: p for k, p in old[t].items() if k < c or (e is not None and k >= e)}
        tracks[t] = dict(sorted((keep | mid).items()))


def _crop_frames(track: Track, lo: int, hi: int, clean) -> list[int]:
    return [k for k in range(lo, hi + 1) if k in track and any(clean[c][i] for c, i in track[k].boxes)]


def _near(track: Track, k: int, side: int, reach: int = 10):
    """Position of the track's point at k, or the nearest one within `reach` frames before (side=-1) / after (+1)."""
    for d in range(reach + 1):
        q = track.get(k + side * d)
        if q is not None:
            return q.xy
    return None


def segment_end(tracks: list[Track], m: dict[int, int], c: int, hi: int, pre: dict, feats, clean, p: "CrossParams",
                rate: float) -> tuple[int | None, dict]:
    """Where the swapped stretch ends. Linkers often take the wrong player only until their next junction and then
    return to the right one, so swapping the whole tail would carry the fix into frames that were right. From c up
    to `hi` (the next group of any moved track), every `seg_chunk_s` chunk asks whether new track t's content (old
    m[t]) still matches t's pre-encounter identity better than t's own content: chunk score = sum over moved t with a
    pre window of cos(pre_t, chunk of m[t]) - cos(pre_t, chunk of t), scaled like the gain. The first chunk scoring
    < -margin is a return; the swap then ends at the frame between the last supporting crop and the first returning
    crop where the hand-back is smoothest (sum of |old m[t] at e-1 - old t at e|). None (whole tail; the next group
    decides again) when no chunk returns before `hi`."""
    L = max(1, int(round(p.seg_chunk_s * rate)))
    last_support, info = c - 1, {"chunks": []}
    for lo in range(c, hi + 1, L):
        top = min(lo + L - 1, hi)
        terms, frames = [], []
        for t, s in m.items():
            if pre.get(t) is None:
                continue
            es, ns = window_emb(tracks[s], lo, top, feats, clean)
            et, nt = window_emb(tracks[t], lo, top, feats, clean)
            if es is None and et is None:
                continue
            # a side without crops (or without points: the source track ended) counts as 'unknown' = p.seg_unknown
            terms.append((float(pre[t] @ es) if es is not None else p.seg_unknown)
                         - (float(pre[t] @ et) if et is not None else p.seg_unknown))
            frames += _crop_frames(tracks[s], lo, top, clean) + _crop_frames(tracks[t], lo, top, clean)
        if not terms:
            continue
        sc = sum(terms) * 2.0 / len(terms)
        info["chunks"].append((lo, round(sc, 2)))
        if sc > p.margin:
            last_support = max(frames)
        elif sc < -p.margin:
            first_return = min(frames)
            best, e = None, None
            for cand in range(max(last_support + 1, c + 1), first_return + 1):
                cost = 0.0
                for t, s in m.items():
                    a, b = _near(tracks[s], cand - 1, -1), _near(tracks[t], cand, +1)
                    cost += float(np.linalg.norm(a - b)) if a is not None and b is not None else p.seg_missing_m
                if best is None or cost < best:
                    best, e = cost, cand
            info["end_cost_m"] = round(best, 2) if best is not None else None
            return e, info
    return None, info


def crossfix(tracks: list[Track], data: Data, feats: dict[str, np.ndarray], p: CrossParams = CrossParams(),
             log: list | None = None, part: tuple[int, int] | None = None) -> list[Track]:
    """Re-assign post-encounter segments of same-team tracks where the head says they were swapped. Returns new
    tracks (the TrackPoint objects are shared, never copied); `log` gets one dict per group. `part` = (lo, hi):
    only decide groups whose run starts in [lo, hi) (for tuning on one half)."""
    rate = data.rate
    tracks = [dict(t) for t in tracks if t]
    clean = crop_mask(data, p)
    teams = [team_vote(t, data)[0] for t in tracks]
    groups = groups_of(encounters(tracks, teams, data, p), pad=int(round(0.5 * rate)))
    r_pre = (int(round(p.pre_s[0] * rate)), int(round(p.pre_s[1] * rate)))
    r_post = (int(round(p.post_s[0] * rate)), int(round(p.post_s[1] * rate)))
    r_vel = int(round(p.vel_s * rate))
    r_ext = int(round(p.ext_s * rate))
    log = log if log is not None else []
    for gi, g in enumerate(groups):
        rec = {"group": gi, "k0": g.k0, "k1": g.k1, "t": _clock(data, g.k0), "tracks": list(g.tracks),
               "team": teams[g.tracks[0]], "pairs": [(i, j, kc, round(dm, 2)) for i, j, kc, dm in g.pairs]}
        log.append(rec)
        if part is not None and not part[0] <= g.k0 < part[1]:
            rec["decision"] = "outside_part"
            continue
        # window bounds: never cross the previous / next group of the same track
        lo_b, hi_b = {}, {}
        for t in g.tracks:
            prev = [h.k1 for h in groups[:gi] if t in h.tracks]
            nxt = [h.k0 for h in groups[gi + 1:] if t in h.tracks]
            lo_b[t] = max(prev) + 1 if prev else -10 ** 9
            hi_b[t] = min(nxt) - 1 if nxt else 10 ** 9
        pre, post, npre, npost = {}, {}, {}, {}
        for t in g.tracks:
            for far in sorted({r_pre[0], max(r_pre[0], r_ext)}):
                pre[t], npre[t] = window_emb(tracks[t], max(g.k0 - far, lo_b[t]), g.k0 - r_pre[1], feats, clean)
                if npre[t] >= p.min_crops:
                    break
            for far in sorted({r_post[1], max(r_post[1], r_ext)}):
                post[t], npost[t] = window_emb(tracks[t], g.k1 + r_post[0], min(g.k1 + far, hi_b[t]), feats, clean)
                if npost[t] >= p.min_crops:
                    break
        rec["crops"] = {str(t): [npre[t], npost[t]] for t in g.tracks}
        has_pre = {t for t in g.tracks if npre[t] >= p.min_crops}
        has_post = {t for t in g.tracks if npost[t] >= p.min_crops}
        full = has_pre & has_post                             # tracks whose own pre/post can be compared
        ok = sorted(has_pre | has_post)
        S = {(a, b): float(pre[a] @ post[b]) for a in has_pre for b in has_post}

        def scored(m: dict[int, int]) -> tuple[float, int] | None:
            """Gain of re-pairing m from the terms that have evidence: S[t, m[t]] - S[t, t] for every moved t with
            a pre window, its own post window and a post window of its new tail; scaled to a pair's two terms."""
            terms = [(t, s) for t, s in m.items() if t in full and s in has_post]
            if len(terms) < p.min_terms:
                return None
            return sum(S[t, s] - S[t, t] for t, s in terms) * 2.0 / len(terms), len(terms)

        cands = []                                            # (gain, terms, perm as {t: source of the new tail})
        if len(ok) <= p.max_perm:
            perms = ({a: b for a, b in zip(ok, perm) if a != b} for perm in itertools.permutations(ok))
        else:
            perms = ({a: b, b: a} for a, b in itertools.combinations(ok, 2))
        for m in perms:
            sc = scored(m) if m else None
            if sc is not None:
                cands.append((sc[0], sc[1], m))
        if not cands:
            rec["decision"] = "no_evidence"
            continue
        cands.sort(key=lambda x: -x[0])
        ends = {t: _side(tracks[t], max(g.k0 - r_vel, lo_b[t]), g.k0 - 1, rate, True) for t in ok}
        starts = {t: _side(tracks[t], g.k1 + 1, min(g.k1 + r_vel, hi_b[t]), rate, False) for t in ok}
        rec["sim"] = {f"{a}->{b}": round(v, 3) for (a, b), v in S.items()}
        # control: two different people at the same place and time (should be low if the head is not location-driven)
        rec["sim_pre_pair"] = {f"{a}|{b}": round(float(pre[a] @ pre[b]), 3)
                               for a, b in itertools.combinations(sorted(has_pre), 2)}
        dup = {frozenset((a, b)) for a, b in itertools.combinations(ok, 2)
               if (a in has_pre and b in has_pre and float(pre[a] @ pre[b]) > p.dup_sim)
               or (a in has_post and b in has_post and float(post[a] @ post[b]) > p.dup_sim)}
        if dup:
            rec["duplicates"] = [sorted(d) for d in dup]
        brk = [t for t in sorted(full) if S[t, t] < p.break_sim]
        if brk:
            rec["breaks"] = brk
        rec["best_gain"] = round(cands[0][0], 3)
        cands = [(gain, m) for gain, _, m in cands]
        applied = None
        used: set[int] = set()
        for gain, m in cands:
            if gain <= p.margin:
                break
            if used & set(m):
                continue
            # cut at the closest approach of the moved tracks
            kcs = [(dm, kc) for i, j, kc, dm in g.pairs if i in m and j in m]
            c = min(kcs)[1] if kcs else (g.k0 + g.k1) // 2
            if any(frozenset((t, s)) in dup for t, s in m.items()):
                why = "duplicate"
            else:
                why = _veto(m, tracks, teams, ends, starts, data, p)
            if not why:                                       # the moved tail (up to its next group) keeps the team
                for t, s in m.items():
                    tail = {k: q for k, q in tracks[s].items() if c <= k <= hi_b[s]}
                    if team_vote(tail, data)[0] not in (teams[t], "U"):
                        why = "tail_team"
            if why:
                rec.setdefault("vetoed", []).append({"perm": {str(a): b for a, b in m.items()}, "gain": round(gain, 3), "why": why})
                continue
            e, seg = (None, {}) if not p.segment else segment_end(
                tracks, m, c, min(min(hi_b[t] for t in m), data.n - 1), {t: pre[t] for t in has_pre}, feats, clean, p, rate)
            _swap_tails(tracks, m, c, e)
            if e is None:                    # relabel later groups: frames >= c of old track m[t] now live in track t
                inv = {s: t for t, s in m.items()}
                for h in groups[gi + 1:]:
                    h.tracks = sorted(inv.get(x, x) for x in h.tracks)
                    h.pairs = [(inv.get(i, i), inv.get(j, j), kc, dm) for i, j, kc, dm in h.pairs]
            applied = applied or []
            applied.append({"perm": {str(a): b for a, b in m.items()}, "gain": round(gain, 3), "cut": c,
                            "cut_t": _clock(data, c), "end": e, "end_t": _clock(data, e) if e is not None else "tail",
                            "segment": seg})
            used |= set(m)
            if len(ok) <= p.max_perm:
                break
        rec["decision"] = "swap" if applied else ("vetoed" if rec.get("vetoed") else "keep")
        if applied:
            rec["applied"] = applied
    return tracks


def _veto(m: dict[int, int], tracks, teams, ends, starts, data: Data, p: CrossParams) -> str:
    """'' if the re-pairing m (new track t gets the tail of old track m[t]) is safe, else the reason. A track that
    starts (or ends) inside the encounter has no motion to continue on that side; every new junction that has
    points on both sides must pass blocks.reachable, and the constant-velocity extrapolation error over the
    encounter may grow by at most motion_slack_m per pair of compared junctions."""
    links = []
    for t, s in m.items():
        if teams[t] != teams[s]:
            return "team"
        if ends[t] is not None and starts[s] is not None:
            if not reachable(ends[t][3], starts[s][3], data, p.vmax, p.slack):
                return "unreachable"
            links.append((t, s))
    if not links:
        return "no_motion_support"
    cmp = [(t, s) for t, s in links if starts[t] is not None]
    old = sum(_cv_err(ends[t], starts[t], data.rate, p.vmax) for t, _ in cmp)
    new = sum(_cv_err(ends[t], starts[s], data.rate, p.vmax) for t, s in cmp)
    if new - old > p.motion_slack_m * max(len(cmp), 1) / 2.0:
        return f"motion {new:.1f} vs {old:.1f} m"
    return ""


def _clock(data: Data, k: int) -> str:
    s = data.start + k / data.rate
    return f"{int(s // 60)}:{s % 60:04.1f}"


def check_safe(before: list[Track], after: list[Track], data: Data) -> dict:
    """Same points (by object), no frame claimed twice by one track (dict keys), team votes unchanged in kind."""
    a = sorted(id(p) for t in before if t for p in t.values())
    b = sorted(id(p) for t in after for p in t.values())
    used = {}
    for t in after:
        for p in t.values():
            for cb in p.boxes:
                used[cb] = used.get(cb, 0) + 1
    used0 = {}
    for t in before:
        for p in t.values():
            for cb in p.boxes:
                used0[cb] = used0.get(cb, 0) + 1
    return {"same_points": a == b, "double_boxes_before": sum(v > 1 for v in used0.values()),
            "double_boxes_after": sum(v > 1 for v in used.values())}


# ---------------------------------------------------------------------------------------------------------------
# Heads, bases, registered variants
# ---------------------------------------------------------------------------------------------------------------

# `python experiments/crossfix.py tune` (results/crossfix_tune.json, 240 configs, first half only): the only
# first-half audited fixes (2: final_d 10:44 and final_b 10:04, both crossings with a clear crop on each side)
# need cam2's top-cut crops (top_exempt), 40 px crops and one-sided evidence (min_terms=1: in final_b 10:04 one
# track is born inside the encounter); no config broke an audited non-SWAP sheet and none changed the
# witnessed (xref) swaps. ext_s=6 / min_crops=1 tie on that score and give the most groups with evidence. Every
# margin 0 .. 1.5 gives the same first-half decisions (the head's gains sit near -2 or +2): the median, 0.5.
TUNED = CrossParams(top_exempt=True, crop_min_h=40.0, ext_s=6.0, min_crops=1, min_terms=1, margin=0.5)


def head(data: Data, window: str = "first") -> dict[str, np.ndarray]:
    from ssl_embed import load_or_train
    return load_or_train(data, **({"train_lo": 0, "train_hi": data.n} if window == "all" else {}))


def base_tracks(data: Data, name: str) -> list[Track]:
    """The saved result (what the audit judged) if present, else re-run the candidate."""
    path = os.path.join(RESULTS, f"{name}.json")
    if os.path.exists(path):
        return harness.load_tracks(path)
    import combine
    return combine.run_candidate(data, combine.FINALS[name])


def run(data: Data, base: str, window: str, p: CrossParams | None = None, out_name: str | None = None) -> list[Track]:
    p = p or TUNED
    tracks = base_tracks(data, base)
    log: list = []
    out = crossfix(tracks, data, head(data, window), p, log)
    half = data.n // 2
    sw = [r for r in log if r["decision"] == "swap"]
    summ = {"params": asdict(p), "base": base, "head": window, "groups": len(log),
            "decisions": {d: sum(r["decision"] == d for r in log) for d in sorted({r["decision"] for r in log})},
            "swaps_first": sum(a["cut"] < half for r in sw for a in r["applied"]),
            "swaps_second": sum(a["cut"] >= half for r in sw for a in r["applied"]),
            "safety": check_safe(tracks, out, data), "log": log}
    print(f"[crossfix] {base} head={window} groups {len(log)} {summ['decisions']} swaps first/second "
          f"{summ['swaps_first']}/{summ['swaps_second']} safety {summ['safety']}", file=sys.stderr)
    if out_name:
        json.dump(summ, open(os.path.join(RESULTS, f"{out_name}.crossfix_log.json"), "w"), indent=1, default=_js)
    return out


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


for _base in ("final_d", "final_b"):
    for _suf, _win in (("cf", "first"), ("cfall", "all")):
        register(f"{_base}_{_suf}")(lambda data, _b=_base, _w=_win, _s=_suf: run(data, _b, _w, out_name=f"{_b}_{_s}"))


# ---------------------------------------------------------------------------------------------------------------
# Audited events (experiments/make_event_audit.py + the blind verdicts in audit/_blind)
# ---------------------------------------------------------------------------------------------------------------

def audit_events(tracks: list[Track], data: Data, variant: str, dist: float = 1.2) -> list[dict]:
    """Re-run make_event_audit's deterministic encounter selection on `tracks` (the audited result) and return one
    dict per sheet: event number, k, ORIGINAL track indices (A, B) and the blind verdict ('SWAP' or 'OK/UNSURE':
    the summary only lists the SWAP sheets). The sampling step is recovered from events.json (the k of every sheet
    must match)."""
    from make_event_audit import majority_team
    keep = [q for q, t in enumerate(tracks) if len(t) >= 5 * data.rate]
    tr = [tracks[q] for q in keep]
    teams = [majority_team(t, data) for t in tr]
    n = data.n
    P = _positions(tr, n)
    enc = []
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            if teams[i] != teams[j] or teams[i] == "?":
                continue
            d = np.linalg.norm(P[i] - P[j], axis=1)
            close = np.where(d < dist)[0]
            if not len(close):
                continue
            for r in np.split(close, np.where(np.diff(close) > int(2 * data.rate))[0] + 1):
                k = int(r[np.argmin(d[r])])
                if k - 2 * data.rate < 0 or k + 2 * data.rate >= n:
                    continue
                enc.append((k, i, j))
    enc.sort()
    ev = json.load(open(os.path.join(ROOT, "experiments", "audit", variant, "events", "events.json")))
    want = [e["k"] for e in ev["sampled"]]
    if ev["encounters_total"] != len(enc):
        raise RuntimeError(f"{variant}: {len(enc)} encounters, audit had {ev['encounters_total']}")
    chosen = None
    for mx in range(1, 200):
        step = max(1, len(enc) // mx)
        c = enc[::step][:mx]
        if [k for k, _, _ in c] == want:
            chosen = c
            break
    if chosen is None:
        raise RuntimeError(f"{variant}: cannot reproduce the audited sheet selection")
    summ = json.load(open(os.path.join(ROOT, "experiments", "audit", "_blind", "summary.json")))
    swaps = set(summ[variant]["swap_sheets"])
    return [{"event": e, "sheet": f"ev_{e:02d}.jpg", "k": k, "t": _clock(data, k), "A": keep[i], "B": keep[j],
             "verdict": "SWAP" if f"ev_{e:02d}.jpg" in swaps else "OK/UNSURE"} for e, (k, i, j) in enumerate(chosen, 1)]


def _point_near(track: Track, k: int, search: int = 6):
    """The point make_event_audit would crop at frame k (nearest frame with boxes within `search`)."""
    for dk in [0] + [s * d for s in range(1, search + 1) for d in (-1, 1)]:
        q = track.get(k + dk)
        if q is not None and q.boxes:
            return q
    return None


def judge_events(events: list[dict], before: list[Track], after: list[Track], data: Data) -> list[dict]:
    """For every audited sheet, follow the crops the sheet showed (A and B at -2 s and +2 s) through the output.
    SWAP sheet: 'fixed' when A(-2 s) now continues into B(+2 s) (and B(-2 s), if it exists, into A(+2 s));
    'changed_3way' when A(-2 s) and A(+2 s) were separated without that. Non-SWAP sheet: 'broken' when any of the
    sheet's pairings A(-2)->A(+2), B(-2)->B(+2) was separated. 'kept' otherwise; 'n/a' without A(-2 s)/A(+2 s)."""
    owner = {id(p): q for q, t in enumerate(after) for p in t.values()}
    out = []
    r2 = int(2 * data.rate)
    for e in events:
        A, B, k = before[e["A"]], before[e["B"]], e["k"]
        a0, b0, a1, b1 = (owner.get(id(x)) if x is not None else None for x in
                          (_point_near(A, k - r2), _point_near(B, k - r2), _point_near(A, k + r2), _point_near(B, k + r2)))
        if a0 is None or a1 is None:
            out.append(e | {"status": "n/a"})
            continue
        a_kept = a0 == a1
        b_kept = b0 is None or b1 is None or b0 == b1
        if e["verdict"] == "SWAP":
            crossed = b1 is not None and a0 == b1 and (b0 is None or b0 == a1)
            st = "fixed" if crossed else ("kept" if a_kept and b_kept else "changed_3way")
        else:
            st = "kept" if a_kept and b_kept else "broken"
        out.append(e | {"status": st})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Tuning (first half only), idempotence, report
# ---------------------------------------------------------------------------------------------------------------

def _audit_summary(judged: list[dict], lo: int, hi: int) -> dict:
    j = [e for e in judged if lo <= e["k"] < hi]
    sw = [e for e in j if e["verdict"] == "SWAP"]
    ok = [e for e in j if e["verdict"] != "SWAP"]
    return {"swap_sheets": len(sw), "fixed": sum(e["status"] == "fixed" for e in sw),
            "swap_sheets_changed_3way": sum(e["status"] == "changed_3way" for e in sw),
            "ok_sheets": len(ok), "broken": sum(e["status"] == "broken" for e in ok)}


def check_idempotent(tracks: list[Track], data: Data, feats, p: CrossParams) -> int:
    """Swaps a second pass makes on crossfix's own output (0 = idempotent)."""
    once = crossfix(tracks, data, feats, p)
    log: list = []
    crossfix(once, data, feats, p, log)
    return sum(len(r.get("applied", [])) for r in log)


def tune(data: Data, bases=("final_d", "final_b"), verbose: bool = True) -> tuple[CrossParams, list[dict]]:
    """Grid on the FIRST half only (head trained on the first half, decisions restricted to groups starting in the
    first half, every score measured on the first half): first-half audited sheets (fixed - 2 x broken), then fewer
    first-half swaps witnessed by the other camera (combine.witnessed_swaps), then more groups with evidence. Among
    the margins tied on that score the median one is taken."""
    import combine
    half = data.n // 2
    feats = head(data, "first")
    ref = combine.source_tracklets(data, "xv_conservative")
    T = {b: base_tracks(data, b) for b in bases}
    E = {b: audit_events(T[b], data, b) for b in bases}
    x0 = {b: combine.witnessed_swaps(T[b], data, "first", ref)["xref_swaps"] for b in bases}
    rows, cache = [], {}
    grid = itertools.product((False, True), (40.0, 60.0), (0.0, 6.0), (1, 2, 3), (1, 2), (0.0, 0.25, 0.5, 1.0, 1.5))
    for top, mh, ext, mc, mt, mg in grid:
        p = replace(CrossParams(), top_exempt=top, crop_min_h=mh, ext_s=ext, min_crops=mc, min_terms=mt, margin=mg)
        row = {"top_exempt": top, "crop_min_h": mh, "ext_s": ext, "min_crops": mc, "min_terms": mt, "margin": mg,
               "fixed": 0, "broken": 0, "dxref": 0, "swaps": 0, "evidence": 0}
        for b in bases:
            log: list = []
            out = crossfix(T[b], data, feats, p, log, part=(0, half))
            sig = (b, tuple(sorted((a["cut"], tuple(sorted(a["perm"].items()))) for r in log for a in r.get("applied", []))))
            if sig not in cache:
                a = _audit_summary(judge_events(E[b], T[b], out, data), 0, half)
                cache[sig] = (a, combine.witnessed_swaps(out, data, "first", ref)["xref_swaps"] - x0[b])
            a, dx = cache[sig]
            row["fixed"] += a["fixed"]; row["broken"] += a["broken"]; row["dxref"] += dx
            row["swaps"] += len(sig[1])
            row["evidence"] += sum("best_gain" in r for r in log if r["decision"] != "outside_part")
        row["score"] = row["fixed"] - 2 * row["broken"]
        rows.append(row)
        if verbose:
            print(row, file=sys.stderr, flush=True)
    key = lambda r: (r["score"], -r["dxref"], r["evidence"])
    best = max(rows, key=key)
    fields = ("top_exempt", "crop_min_h", "ext_s", "min_crops", "min_terms")
    tied = sorted(r["margin"] for r in rows if key(r) == key(best) and all(r[f] == best[f] for f in fields))
    p = replace(CrossParams(), **{f: best[f] for f in fields}, margin=tied[len(tied) // 2])
    return p, rows


REPORT_KEYS = ["ids", "ids_over_20s", "frames_in_top10", "events_per_min", "teleports_per_min",
               "appearance_breaks_per_min", "team_impurity", "dup_pairs", "double_used_boxes", "box_coverage",
               "count_5v5", "count_over5", "speed_over9_per_min", "speed_jumps_per_min", "crossings",
               "crossings_scored", "swap_flagged_share", "reid_dispersion", "multi_cluster_share"]


def report(data: Data, names: dict[str, tuple[str, str | None]] | None = None) -> dict:
    """metrics2 (first / second), witnessed swaps, crossfix swaps per half, audited sheets fixed / broken."""
    import combine
    import metrics2
    names = names or {"final_d": ("final_d", None), "final_d_cf": ("final_d", "first"), "final_d_cfall": ("final_d", "all"),
                      "final_b": ("final_b", None), "final_b_cf": ("final_b", "first"), "final_b_cfall": ("final_b", "all")}
    half = data.n // 2
    ref = combine.source_tracklets(data, "xv_conservative")
    out = {"params": asdict(TUNED)}
    for name, (base, win) in names.items():
        T0 = base_tracks(data, base)
        path = os.path.join(RESULTS, f"{name}.json")
        T = harness.load_tracks(path) if win and os.path.exists(path) else (run(data, base, win) if win else T0)
        r = {}
        for part in ("first", "second"):
            m = metrics2.evaluate(T, data, part)
            r[part] = {k: m[k] for k in REPORT_KEYS} | combine.witnessed_swaps(T, data, part, ref)
        if win:
            lg = os.path.join(RESULTS, f"{name}.crossfix_log.json")
            if os.path.exists(lg):
                L = json.load(open(lg))
                r["swaps_first"], r["swaps_second"] = L["swaps_first"], L["swaps_second"]
                r["decisions"] = L["decisions"]
                r["applied"] = [a | {"tracks": x["tracks"], "team": x["team"]} for x in L["log"] for a in x.get("applied", [])]
            # recompute the swap on the in-memory base so judge_events can follow the TrackPoint objects
            T_mem = crossfix(T0, data, head(data, win), TUNED)
            judged = judge_events(audit_events(T0, data, base), T0, T_mem, data)
            r["audit_first"] = _audit_summary(judged, 0, half)
            r["audit_second"] = _audit_summary(judged, half, data.n)
            r["audit_changed"] = [{k: e[k] for k in ("sheet", "t", "verdict", "status", "A", "B")}
                                  for e in judged if e["status"] not in ("kept", "n/a")]
            r["idempotent_second_pass_swaps"] = check_idempotent(T0, data, head(data, win), TUNED)
        out[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ("first", "second")}, default=_js), file=sys.stderr, flush=True)
    json.dump(out, open(os.path.join(RESULTS, "crossfix_report.json"), "w"), indent=1, default=_js)
    return out


if __name__ == "__main__":
    _data = Data()
    if sys.argv[1:2] == ["tune"]:
        _p, _rows = tune(_data)
        json.dump({"chosen": asdict(_p), "grid": _rows}, open(os.path.join(RESULTS, "crossfix_tune.json"), "w"), indent=1)
        print("chosen", asdict(_p))
    elif sys.argv[1:2] == ["report"]:
        _r = report(_data)
        import metrics2
        print(metrics2.table({f"{n}/{part}": _r[n][part] for n in _r if n != "params" for part in ("first", "second")}))
