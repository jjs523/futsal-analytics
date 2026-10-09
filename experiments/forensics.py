"""Forensics of the blind-audit crossing events: which pipeline stage made each audited SWAP (and the OK controls)?

    python experiments/forensics.py [--verdicts <workflow output with the blind-audit details>] [--rebuild]

For final_d (xview('conservative') -> link_closed) and final_b (best_source('relink') -> link_gta) this
  1. re-selects the audited encounters exactly as make_event_audit.py does (same (k, i, j) per ev_XX) and checks
     them against audit/<variant>/events/events.json;
  2. rebuilds the tracklet source of each candidate step by step, exactly as combine.run_candidate / best_source do,
     and labels every box (cam, idx) with the unit it belongs to at every stage:
        L0  per-camera conservative tracklet (percam builder)
        L1  per-camera re-linked chain (final_b: sources.relink with BoostTrack proposals)
        L2  pair_views component (final_d: cut on partner change; final_b: no cut)
        L3  fused re-link chain (final_b: sources.relink min_len 5)
        SRC the linker's input tracklet (final_d: L2; final_b: absorb_short(L3))
        PC  the linker's internal piece (after split_jumps / team split / duplicate merge, resp. team split)
     and re-runs the linker with light instrumentation (link_closed.assign / _attach_short, link_gta.Linker._absorb)
     so every junction inside an output track is attributed to the decision that made it (closed: MILP slot vs
     short-tracklet attachment vs duplicate merge; gta: appearance merge vs motion-only merge vs debris attach);
     the re-run output is checked box-for-box against the saved result;
  3. for every event and both tracks: the custody chain between the crops the rater saw (best_box at -2/-1/+1/+2 s),
     every stage boundary within +-3 s of the closest approach, fused-point sanity (distance between the two views'
     foot points, agreement with sources.xview_partners), and clean ReID (raw OSNet and the SSL head) 1-3 s before
     vs after k: same-track and cross-track similarities and the 'swap if cross > same' rule;
  4. assigns a cause category: linker junction / source-internal (builder, per-camera relink, fused relink,
     absorb) / cross-view pairing / slot constraint / duplicate track; writes experiments/audit/forensics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, Data, Track, load_tracks

AUDIT = os.path.join(ROOT, "experiments", "audit")
OUT_JSON = os.path.join(AUDIT, "forensics.json")
VARIANTS = ("final_d", "final_b")
WIN_S = 3.0                       # stage boundaries within +-WIN_S of k count as "at the crossing"
DEFAULT_VERDICTS = os.environ.get("FORENSICS_VERDICTS", "")


# ---------------------------------------------------------------------------------------------------------------
# 1. Encounter selection (identical to make_event_audit.py)
# ---------------------------------------------------------------------------------------------------------------

def majority_team(t: Track, data: Data) -> str:
    labs = [data.cams[c].team[i] for p in t.values() for c, i in p.boxes if data.cams[c].team[i]]
    if not labs:
        return "?"
    y = sum(1 for x in labs if x == "Y")
    return "Y" if y >= len(labs) / 2 else "N"


def best_box(t: Track, k: int, data: Data, search: int = 6):
    for dk in [0] + [s * d for s in range(1, search + 1) for d in (-1, 1)]:
        p = t.get(k + dk)
        if p and p.boxes:
            return k + dk, max(p.boxes, key=lambda cb: data.cams[cb[0]].h[cb[1]])
    return None, None


def select_events(tracks: list[Track], data: Data, max_n: int = 30, dist: float = 1.2):
    """(filtered tracks, [(k, i, j)] in sheet order) exactly as make_event_audit.py."""
    tracks = [t for t in tracks if len(t) >= 5 * data.rate]
    teams = [majority_team(t, data) for t in tracks]
    n = data.n
    pos = []
    for t in tracks:
        p = np.full((n, 2), np.nan)
        for k, q in t.items():
            if 0 <= k < n:
                p[k] = q.xy
        pos.append(p)
    enc = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if teams[i] != teams[j] or teams[i] == "?":
                continue
            d = np.linalg.norm(pos[i] - pos[j], axis=1)
            close = np.where(d < dist)[0]
            if not len(close):
                continue
            runs = np.split(close, np.where(np.diff(close) > int(2 * data.rate))[0] + 1)
            for r in runs:
                k = int(r[np.argmin(d[r])])
                if k - 2 * data.rate < 0 or k + 2 * data.rate >= n:
                    continue
                enc.append((k, i, j))
    enc.sort()
    step = max(1, len(enc) // max_n) if enc else 1
    return tracks, teams, enc[::step][:max_n]


# ---------------------------------------------------------------------------------------------------------------
# 2. Source rebuild with per-box stage labels, instrumented linkers
# ---------------------------------------------------------------------------------------------------------------

def box_labels(tracklets: list[Track]) -> dict[tuple[str, int], int]:
    out = {}
    for q, t in enumerate(tracklets):
        for p in t.values():
            for cb in p.boxes:
                out[(cb[0], int(cb[1]))] = q
    return out


def box_sets(tracks: list[Track]) -> list[frozenset]:
    return [frozenset((k, cb[0], int(cb[1])) for k, p in t.items() for cb in p.boxes) for t in tracks]


def build_final_d(data: Data) -> dict:
    import link_closed as lc
    from combine import Candidate, run_candidate
    from percam import per_camera
    from xview import xview
    per = per_camera(data, "conservative")
    L0 = {}
    for cam, tl in per.items():
        L0.update({cb: f"{cam[-1]}:{q}" for cb, q in box_labels(tl).items()})
    src = xview(data, "conservative")
    cap: dict = {}
    orig_assign, orig_attach = lc.assign, lc._attach_short

    def assign_w(tls, data_, P, st, exclusive=()):
        a = orig_assign(tls, data_, P, st, exclusive)
        cap["long"], cap["assignment"], cap["exclusive"] = tls, dict(a), list(exclusive)
        return a

    def attach_w(slots, owner, slot_team, shorts, data_, P):
        n = orig_attach(slots, owner, slot_team, shorts, data_, P)
        cap["shorts"], cap["slots"], cap["owner"] = shorts, [dict(s) for s in slots], [dict(o) for o in owner]
        cap["slot_team"] = list(slot_team)
        return n
    lc.assign, lc._attach_short = assign_w, attach_w
    try:
        out = run_candidate(data, Candidate("xv_conservative", "closed", "raw"))
    finally:
        lc.assign, lc._attach_short = orig_assign, orig_attach
    pieces = [tl.tr for tl in cap["long"]] + [tl.tr for tl in cap["shorts"]]
    nl = len(cap["long"])
    piece_kind = ["milp"] * nl + ["short"] * len(cap["shorts"])
    piece_slot = {t: s for t, s in cap["assignment"].items()}
    # piece -> how many source tracklets it fuses (duplicate merges)
    lab_src = box_labels(src)
    piece_srcs = [sorted({lab_src.get((c, int(i)), -1) for p in pc.values() for c, i in p.boxes}) for pc in pieces]
    # output track -> slot (by box set)
    slot_sets = {frozenset((k, c, int(i)) for k, p in s.items() for c, i in p.boxes): q for q, s in enumerate(cap["slots"])}
    out_slot = [slot_sets.get(bs) for bs in box_sets(out)]
    owner_piece = []
    for s in out_slot:
        own = cap["owner"][s] if s is not None else {}
        owner_piece.append({k: (v if v >= 0 else nl + (-1 - v)) for k, v in own.items()})
    return {"labels": {"L0": L0, "SRC": {cb: f"x{q}" for cb, q in lab_src.items()}},
            "out": out, "out_slot": out_slot, "slot_team": cap["slot_team"], "owner_piece": owner_piece,
            "piece_kind": piece_kind, "piece_slot": piece_slot, "piece_srcs": piece_srcs,
            "piece_len": [len(p) for p in pieces], "exclusive": cap["exclusive"],
            "pieces_boxes": [{k: list(p.boxes) for k, p in pc.items()} for pc in pieces]}


def build_final_b(data: Data) -> dict:
    import link_gta as lg
    from combine import Candidate, run_candidate
    from percam import per_camera
    from sources import PROPOSER, RelinkStats, absorb_short, best_source, relink
    from xview import pair_views
    per = per_camera(data, "conservative")
    prop = per_camera(data, PROPOSER)
    L0, L1, rl1 = {}, {}, {}
    per1 = {}
    for cam in data.cams:
        st = RelinkStats()
        per1[cam] = relink(per[cam], data, proposals=prop.get(cam), prop_min_len=2, stats=st)
        rl1[cam] = st.links
        L0.update({cb: f"{cam[-1]}:{q}" for cb, q in box_labels(per[cam]).items()})
        L1.update({cb: f"{cam[-1]}:{q}" for cb, q in box_labels(per1[cam]).items()})
    c1, c2 = list(data.cams)
    t2 = pair_views(per1[c1], per1[c2], data, cut_on_partner_change=False)
    st3 = RelinkStats()
    t3 = relink(t2, data, min_len=5, stats=st3)
    t4 = absorb_short(t3, data)
    ref = best_source(data, "relink")
    same = sorted(box_sets(ref)) == sorted(box_sets(t4))
    cap: dict = {"merges": []}
    orig_init, orig_absorb = lg.Linker.__init__, lg.Linker._absorb

    def init_w(self, data_, tracklets, params, stats=None):
        orig_init(self, data_, tracklets, params, stats)
        cap["pieces"] = list(tracklets)
        cap["linker"] = self

    def absorb_w(self, a, b, partners=None):
        A, B = self.clusters[a], self.clusters[b]
        r = self.evaluate(A, B)
        cap["merges"].append({"a": list(A.members), "b": list(B.members), "phase": "debris" if partners is None else "agglo",
                              "motion_only": bool(r.motion_only), "cost": round(float(r.cost), 3),
                              "z": round(float(r.z), 2) if np.isfinite(r.z) else None,
                              "gap": round(float(r.gap), 2) if np.isfinite(r.gap) else None})
        return orig_absorb(self, a, b, partners)
    lg.Linker.__init__, lg.Linker._absorb = init_w, absorb_w
    try:
        out = run_candidate(data, Candidate("relink", "gta", "raw"))
    finally:
        lg.Linker.__init__, lg.Linker._absorb = orig_init, orig_absorb
    pieces = cap["pieces"]
    lk = cap["linker"]
    piece_emb = [lk.embs[q] is not None for q in range(len(pieces))]
    piece_debris = [(max(p) - min(p) + 1 < lk.p.debris_len_s * data.rate) and not piece_emb[q] for q, p in enumerate(pieces)]
    # output track -> pieces (by box)
    lab_piece = box_labels(pieces)
    owner_piece = []
    for t in out:
        own = {}
        for k, p in t.items():
            ps = {lab_piece.get((c, int(i))) for c, i in p.boxes} - {None}
            if ps:
                own[k] = min(ps) if len(ps) == 1 else tuple(sorted(ps))
        owner_piece.append(own)
    lab_src = box_labels(t4)
    piece_srcs = [sorted({lab_src.get((c, int(i)), -1) for p in pc.values() for c, i in p.boxes}) for pc in pieces]
    return {"labels": {"L0": L0, "L1": L1, "L2": {cb: f"p{q}" for cb, q in box_labels(t2).items()},
                       "L3": {cb: f"r{q}" for cb, q in box_labels(t3).items()},
                       "SRC": {cb: f"s{q}" for cb, q in lab_src.items()}},
            "src_matches_best_source": same, "relink1_links": rl1, "relink3_links": st3.links,
            "out": out, "owner_piece": owner_piece, "merges": cap["merges"], "piece_emb": piece_emb,
            "piece_debris": piece_debris, "piece_srcs": piece_srcs, "piece_len": [len(p) for p in pieces],
            "pieces_boxes": [{k: list(p.boxes) for k, p in pc.items()} for pc in pieces]}


def build(data: Data, cache_path: str | None = None, rebuild: bool = False) -> dict:
    """Everything above for both candidates (~1 min); optionally pickled to `cache_path` for re-analysis."""
    if cache_path and os.path.exists(cache_path) and not rebuild:
        return pickle.load(open(cache_path, "rb"))
    from sources import xview_partners
    B = {"final_d": build_final_d(data), "final_b": build_final_b(data)}
    B["partners"] = xview_partners(data)
    for v in VARIANTS:
        saved = load_tracks(os.path.join(ROOT, "experiments", "results", f"{v}.json"))
        B[v]["matches_saved"] = box_sets(saved) == box_sets(B[v]["out"])
        print(f"[{v}] re-run matches saved result box-for-box: {B[v]['matches_saved']}", file=sys.stderr)
    if cache_path:
        pickle.dump(B, open(cache_path, "wb"))
    return B


# ---------------------------------------------------------------------------------------------------------------
# 3. Per-event analysis
# ---------------------------------------------------------------------------------------------------------------

STAGES = {   # upstream -> downstream; the first stage that puts two boxes into one unit made the join between them
    "final_d": [("L0", "builder"), ("SRC", "xview_pairing"), ("PC", "linker_dup_merge"), ("OUT", "linker_junction")],
    "final_b": [("L0", "builder"), ("L1", "percam_relink"), ("L2", "xview_pairing"), ("L3", "fused_relink"),
                ("SRC", "absorb_short"), ("PC", "linker_piece"), ("OUT", "linker_junction")],
}

# Blind-audit verdicts per sheet: SWAP sheets are in audit/_blind/summary.json; the UNSURE ones come from the blind
# workflow's per-sheet details (unblinded with audit/_blind/key.json); every other audited sheet (ev 1-24) is OK.
UNSURE = {"final_d": {3, 15, 21, 22}, "final_b": {4, 9}}
# Which row the rater saw change person, between which shown crops (seconds), from the rater's written reasons.
# 'dup': on one side of the change the row shows the OTHER row's player (two tracks on one person).
RATER_SWITCH = {
    ("final_d", 1): [("A", -1, 2), ("B", 1, 2)],      # A: red-striped pants -> other player; A's player in B +1
    ("final_d", 8): [("A", -2, 1), ("B", -1, 1)],      # clean exchange
    ("final_d", 13): [("A", -1, 1), ("B", -1, 2)],     # A takes B's armband player at +1; B +2 another player
    ("final_d", 14): [("A", -2, 1)],                   # A: grey shorts -> long trousers; B constant
    ("final_d", 20): [("B", 1, 2), ("A", 1, 2)],       # both jump at +2 s (B near -> far player, A jumps location)
    ("final_d", 23): [("B", -1, 1, "dup")],            # B before = A's striped-jacket player, B +1 another player
    ("final_b", 5): [("A", -1, 1)],                    # A red jersey -> blue/purple top; A's player in B +2
    ("final_b", 7): [("B", -1, 1)],                    # B black top -> red shirt
    ("final_b", 22): [("B", -2, 2)],                   # B blue shirt -> red shirt (crops hold several people)
    ("final_b", 23): [("B", -1, 1, "dup")],            # B no-bib black T-shirt -> yellow bib -> A's player at +2
}


def verdict_of(variant: str, ev: int, summary: dict) -> str:
    if f"ev_{ev:02d}.jpg" in summary.get(variant, {}).get("swap_sheets", []):
        return "SWAP"
    return "UNSURE" if ev in UNSURE.get(variant, set()) else "OK"


def load_feats(data: Data) -> dict[str, dict[str, np.ndarray]]:
    from blocks import normalise
    from ssl_embed import load_or_train
    f = {"raw": {cam: normalise(c.reid.astype(np.float32)) for cam, c in data.cams.items()},
         "ssl": load_or_train(data)}                                   # SSL_DEFAULTS: head trained on minutes 0-3
    f["ssl2"] = load_or_train(data, train_lo=1800, train_hi=3600)      # same head trained on minutes 3-6 (cached)
    return f


def cos(a, b):
    return None if a is None or b is None else round(float(a @ b), 3)


class Ctx:
    def __init__(self, data: Data, B: dict, feats: dict):
        from link_gta import clean_crops
        self.data, self.B, self.feats = data, B, feats
        self.clean = clean_crops(data, ("cam2",))           # the linkers' clean-crop rule (cam2 top edge exempt)
        self.rate = int(data.rate)

    def label(self, v: str, ti: int, k: int, cb, stage: str):
        if stage == "OUT":
            return "out"
        if stage == "PC":
            return self.B[v]["owner_piece"][ti].get(k)
        return self.B[v]["labels"][stage].get((cb[0], int(cb[1])))

    def link_type(self, v: str, ti: int, a, b) -> str:
        """Stage that first joins boxes a = (k, cb) and b = (k, cb) of output track ti (see STAGES)."""
        if a[1] is None or b[1] is None:
            return "no_crop"
        for stage, cause in STAGES[v]:
            la, lb = self.label(v, ti, a[0], a[1], stage), self.label(v, ti, b[0], b[1], stage)
            if la is not None and la == lb:
                return cause
        return "?"

    def emb_frames(self, frames: dict, kind: str):
        """Mean clean embedding over {k: boxes} (largest clean box per frame), and the crop count."""
        rows = []
        for k, boxes in frames.items():
            best = None
            for cam, i in boxes:
                if self.clean[cam][i] and (best is None or self.data.cams[cam].h[i] > self.data.cams[best[0]].h[best[1]]):
                    best = (cam, i)
            if best is not None:
                rows.append(self.feats[kind][best[0]][best[1]])
        if len(rows) < 2:
            return None, len(rows)
        m = np.mean(rows, 0)
        return m / max(np.linalg.norm(m), 1e-9), len(rows)

    def emb(self, t: Track, lo: int, hi: int, kind: str):
        return self.emb_frames({k: t[k].boxes for k in range(lo, hi + 1) if k in t}, kind)

    def piece_emb(self, v: str, q, kind: str, lo: int | None = None, hi: int | None = None):
        if not isinstance(q, int):
            return None
        pc = self.B[v]["pieces_boxes"][q]
        return self.emb_frames({k: b for k, b in pc.items() if (lo is None or k >= lo) and (hi is None or k <= hi)},
                               kind)[0]


def timeline(ctx: Ctx, v: str, ti: int, t: Track, lo: int, hi: int) -> list[str]:
    """Run-length segments of the track's stage labels in [lo, hi] (box-bearing frames), as compact strings."""
    stages = [s for s, _ in STAGES[v] if s != "OUT"]
    segs: list[list] = []
    for k in range(lo, hi + 1):
        p = t.get(k)
        if not p or not p.boxes:
            continue
        lab = tuple("+".join(sorted(str(ctx.label(v, ti, k, cb, s)) for cb in p.boxes)) if s != "PC"
                    else str(ctx.label(v, ti, k, None, "PC")) for s in stages)
        if segs and segs[-1][2] == lab and k - segs[-1][1] <= 3:
            segs[-1][1] = k
        else:
            segs.append([k, k, lab])
    return [f"{a - 0}-{b}: " + " ".join(f"{s}={x}" for s, x in zip(stages, lab)) for a, b, lab in segs]


def boundaries(ctx: Ctx, v: str, ti: int, t: Track, lo: int, hi: int) -> dict[str, list]:
    """Stage boundaries between consecutive box-bearing points in [lo, hi] (L0 / L1 per camera)."""
    out: dict[str, list] = {}
    ks = [k for k in range(lo, hi + 1) if k in t and t[k].boxes]
    for stage, _ in STAGES[v]:
        if stage == "OUT":
            continue
        ch = []
        if stage in ("L0", "L1"):
            for cam in ctx.data.cams:
                seq = [(k, cb) for k in ks for cb in t[k].boxes if cb[0] == cam]
                for (ka, a), (kb, b) in zip(seq, seq[1:]):
                    la, lb = ctx.label(v, ti, ka, a, stage), ctx.label(v, ti, kb, b, stage)
                    if la != lb:
                        ch.append([ka, kb, la, lb])
        else:
            prev = None
            for k in ks:
                lab = ctx.label(v, ti, k, None, "PC") if stage == "PC" else \
                    "+".join(sorted({str(ctx.label(v, ti, k, cb, stage)) for cb in t[k].boxes}))
                lab = str(lab)
                if prev is not None and lab != prev[1]:
                    ch.append([prev[0], k, prev[1], lab])
                prev = (k, lab)
        out[stage] = ch
    return out


def fused_checks(ctx: Ctx, t: Track, lo: int, hi: int) -> dict:
    """Two-view points in [lo, hi]: distance between the two views' foot points, and pairs that contradict the
    unambiguous per-frame cross-view partner (sources.xview_partners)."""
    P = ctx.B["partners"]
    n = far = disagree = 0
    dmax = 0.0
    for k in range(lo, hi + 1):
        p = t.get(k)
        if not p or len(p.boxes) != 2:
            continue
        (ca, ia), (cb, ib) = sorted(p.boxes)
        d = float(np.linalg.norm(ctx.data.cams[ca].xy[ia] - ctx.data.cams[cb].xy[ib]))
        n += 1
        dmax = max(dmax, d)
        far += d > 1.0
        pa, pb = P[ca][ia], P[cb][ib]
        disagree += bool((pa >= 0 and pa != ib) or (pb >= 0 and pb != ia))
    return {"fused_pts": n, "fused_far_1m": far, "fused_dmax_m": round(dmax, 2), "partner_disagree": disagree}


def merge_joining(Bv: dict, pa, pb) -> dict | None:
    """The link_gta merge that first put pieces pa and pb (a piece, or a tuple of pieces sharing a fused point) into
    one cluster."""
    A = set(pa) if isinstance(pa, tuple) else {pa}
    Bs = set(pb) if isinstance(pb, tuple) else {pb}
    if A & Bs:
        A, Bs = A - Bs, Bs - A
        if not A or not Bs:
            return None
    for m in Bv["merges"]:
        if (A & set(m["a"]) and Bs & set(m["b"])) or (Bs & set(m["a"]) and A & set(m["b"])):
            return m
    return None


def linker_junctions(ctx: Ctx, v: str, ti: int, lo: int, hi: int) -> list[dict]:
    """Junctions between linker input pieces inside [lo, hi] and the decision that made them, with the clean
    appearance similarity (raw / ssl) of the two pieces on either side."""
    Bv = ctx.B[v]
    own = Bv["owner_piece"][ti]
    ks = [k for k in sorted(own) if lo <= k <= hi]
    out = []
    for a, b in zip(ks, ks[1:]):
        pa, pb = own[a], own[b]
        if pa == pb:
            continue
        j = {"k_from": a, "k_to": b, "gap_s": round((b - a) / ctx.rate, 1), "piece_from": str(pa), "piece_to": str(pb)}
        ints = [q for q in (pa, pb) if isinstance(q, int)]
        j["src_from"] = Bv["piece_srcs"][pa] if isinstance(pa, int) else None
        j["src_to"] = Bv["piece_srcs"][pb] if isinstance(pb, int) else None
        j["len_from"] = Bv["piece_len"][pa] if isinstance(pa, int) else None
        j["len_to"] = Bv["piece_len"][pb] if isinstance(pb, int) else None
        if v == "final_d":
            j["made_by"] = "+".join(sorted({Bv["piece_kind"][q] for q in ints}))
        else:
            m = merge_joining(Bv, pa, pb)
            j["made_by"] = ("shared_frame" if m is None else         # a fused point of two clusters' boxes
                            "debris_attach" if m["phase"] == "debris" else
                            "motion_merge" if m["motion_only"] else "app_merge")
            j["merge"] = m
        for kind in ("raw", "ssl"):
            j[f"sim_{kind}"] = cos(ctx.piece_emb(v, pa, kind, hi=a), ctx.piece_emb(v, pb, kind, lo=b))
        out.append(j)
    return out


def appearance(ctx: Ctx, A: Track, Bt: Track, k: int) -> dict:
    """Clean ReID 1-3 s before vs after k for both tracks; same / cross similarities and the swap rules."""
    r = ctx.rate
    out = {}
    for kind in ctx.feats:
        a0, na0 = ctx.emb(A, k - 3 * r, k - r, kind)
        a1, na1 = ctx.emb(A, k + r, k + 3 * r, kind)
        b0, nb0 = ctx.emb(Bt, k - 3 * r, k - r, kind)
        b1, nb1 = ctx.emb(Bt, k + r, k + 3 * r, kind)
        d = {"n_crops": [na0, na1, nb0, nb1], "same_A": cos(a0, a1), "same_B": cos(b0, b1),
             "cross_AB": cos(a0, b1), "cross_BA": cos(b0, a1)}
        # per track: does the after-part look more like the other track's before-part than like its own?
        d["flag_A"] = None if None in (d["same_A"], d["cross_BA"]) else d["cross_BA"] > d["same_A"]
        d["flag_B"] = None if None in (d["same_B"], d["cross_AB"]) else d["cross_AB"] > d["same_B"]
        # pair rule: swap if cross_AB + cross_BA > same_A + same_B (needs all four)
        vals = [d["same_A"], d["same_B"], d["cross_AB"], d["cross_BA"]]
        d["rule_pair"] = None if None in vals else (vals[2] + vals[3] > vals[0] + vals[1])
        out[kind] = d
    return out


def crops_shown(ctx: Ctx, v: str, ti: int, t: Track, k: int) -> list[dict]:
    rows = []
    for ds in (-2, -1, 1, 2):
        kk, cb = best_box(t, int(k + ds * ctx.rate), ctx.data)
        lab = {}
        if cb is not None:
            for stage, _ in STAGES[v]:
                if stage != "OUT":
                    lab[stage] = str(ctx.label(v, ti, kk, cb, stage))
        rows.append({"ds": ds, "k": kk, "box": None if cb is None else [cb[0], int(cb[1])], "labels": lab})
    return rows


def analyse_event(ctx: Ctx, v: str, e: int, k: int, ti: int, tj: int, verdict: str) -> dict:
    data, r = ctx.data, ctx.rate
    tracks = ctx.B[v]["out"]
    A, Bt = tracks[ti], tracks[tj]
    lo, hi = int(k - WIN_S * r), int(k + WIN_S * r)
    t_s = data.start + k / data.rate
    row = {"variant": v, "ev": e, "t": f"{int(t_s // 60)}:{t_s % 60:04.1f}", "k": k, "verdict": verdict,
           "tracks": {"A": ti, "B": tj}}
    per = {}
    for lab, q, t in (("A", ti, A), ("B", tj, Bt)):
        shown = crops_shown(ctx, v, q, t, k)
        steps = []
        for x, y in zip(shown, shown[1:]):
            steps.append({"from": x["ds"], "to": y["ds"],
                          "join": ctx.link_type(v, q, (x["k"], tuple(x["box"]) if x["box"] else None),
                                                (y["k"], tuple(y["box"]) if y["box"] else None))})
        per[lab] = {"shown": shown, "steps": steps, "boundaries": boundaries(ctx, v, q, t, lo, hi),
                    "linker_junctions": linker_junctions(ctx, v, q, lo, hi), "fused": fused_checks(ctx, t, lo, hi),
                    "timeline": timeline(ctx, v, q, t, lo, hi)}
        if v == "final_d":
            per[lab]["slot"] = ctx.B[v]["out_slot"][q]
    row["tracks_detail"] = per
    row["appearance"] = appearance(ctx, A, Bt, k)
    # the joins between the crops the rater saw change (SWAP), else between -1 s and +1 s (controls)
    sw = RATER_SWITCH.get((v, e), [])
    joins = []
    for s in sw:
        lab, d0, d1 = s[:3]
        q, t = (ti, A) if lab == "A" else (tj, Bt)
        x = next(c for c in per[lab]["shown"] if c["ds"] == d0)
        y = next(c for c in per[lab]["shown"] if c["ds"] == d1)
        jt = ctx.link_type(v, q, (x["k"], tuple(x["box"]) if x["box"] else None), (y["k"], tuple(y["box"]) if y["box"] else None))
        joins.append({"row": lab, "from": d0, "to": d1, "join": jt, "dup": len(s) > 3,
                      "shown_sim_raw": None if not (x["box"] and y["box"]) else
                      round(float(ctx.feats["raw"][x["box"][0]][x["box"][1]] @ ctx.feats["raw"][y["box"][0]][y["box"][1]]), 3),
                      "shown_sim_ssl": None if not (x["box"] and y["box"]) else
                      round(float(ctx.feats["ssl"][x["box"][0]][x["box"][1]] @ ctx.feats["ssl"][y["box"][0]][y["box"][1]]), 3)})
    row["rater_switch_joins"] = joins
    row["control_joins"] = {lab: next(s["join"] for s in per[lab]["steps"] if s["from"] == -1) for lab in ("A", "B")}
    row["junction_within_3s"] = {lab: [[j["k_from"], j["k_to"], j["made_by"]] for j in per[lab]["linker_junctions"]]
                                 for lab in ("A", "B")}
    row["junction_kinds_3s"] = sorted({str(j["made_by"]) for lab in ("A", "B") for j in per[lab]["linker_junctions"]})
    row["appearance_ext"] = appearance_ext(ctx, A, Bt, k)
    row["units"] = units_table(ctx, v, ti, tj, k)
    if v == "final_b":
        row["percam_relinks_3s"] = percam_relinks(ctx, [("A", A), ("B", Bt)], lo, hi)
    row["overlap_blocks"] = overlap_blocks(ctx, v, row["units"]) if v == "final_d" else []
    causes = []
    for j in joins:
        q = ti if j["row"] == "A" else tj
        x = next(c for c in per[j["row"]]["shown"] if c["ds"] == j["from"])
        y = next(c for c in per[j["row"]]["shown"] if c["ds"] == j["to"])
        j["cause"] = switch_cause(ctx, v, q, x["k"], y["k"], j["join"], per[j["row"]], row["overlap_blocks"],
                                  [r for r in row.get("percam_relinks_3s", []) if r["track"] == j["row"]])
        causes.append(j["cause"])
    row["cause_auto"] = "+".join(sorted(set(causes))) if causes else None
    adj = ADJUDICATED.get((v, e))
    row["cause"] = adj[0] if adj else row["cause_auto"]
    row["cause_note"] = adj[1] if adj else None
    return row


def appearance_ext(ctx: Ctx, A: Track, Bt: Track, k: int, horizon_s: float = 10.0, n_max: int = 20) -> dict:
    """As `appearance`, but each side takes the n_max clean crops nearest to k (beyond +-1 s, within horizon_s):
    the 1-3 s windows are often empty at crossings (overlapped boxes are never clean)."""
    r = ctx.rate

    def side(t: Track, before: bool, kind: str):
        ks = range(k - r, k - int(horizon_s * r) - 1, -1) if before else range(k + r, k + int(horizon_s * r) + 1)
        frames = {}
        for kk in ks:
            if kk in t and any(ctx.clean[c][i] for c, i in t[kk].boxes):
                frames[kk] = t[kk].boxes
                if len(frames) >= n_max:
                    break
        return ctx.emb_frames(frames, kind)
    out = {}
    for kind in ctx.feats:
        (a0, na0), (a1, na1) = side(A, True, kind), side(A, False, kind)
        (b0, nb0), (b1, nb1) = side(Bt, True, kind), side(Bt, False, kind)
        d = {"n_crops": [na0, na1, nb0, nb1], "same_A": cos(a0, a1), "same_B": cos(b0, b1),
             "cross_AB": cos(a0, b1), "cross_BA": cos(b0, a1)}
        d["flag_A"] = None if None in (d["same_A"], d["cross_BA"]) else d["cross_BA"] > d["same_A"]
        d["flag_B"] = None if None in (d["same_B"], d["cross_AB"]) else d["cross_AB"] > d["same_B"]
        vals = [d["same_A"], d["same_B"], d["cross_AB"], d["cross_BA"]]
        d["rule_pair"] = None if None in vals else (vals[2] + vals[3] > vals[0] + vals[1])
        out[kind] = d
    return out


def units_table(ctx: Ctx, v: str, ti: int, tj: int, k: int, span_s: float = 5.0) -> list[dict]:
    """Linker input pieces on either track within +-span_s of k: span, clean crops, source tracklets, how the linker
    used them, and their clean-appearance similarity (ssl / raw) to every other embedded piece in the table."""
    Bv, r = ctx.B[v], ctx.rate
    rows, embs = [], []
    for lab, q in (("A", ti), ("B", tj)):
        own = Bv["owner_piece"][q]
        seen = []
        for kk in range(int(k - span_s * r), int(k + span_s * r) + 1):
            p = own.get(kk)
            if isinstance(p, int) and p not in seen:
                seen.append(p)
        for p in seen:
            pc = Bv["pieces_boxes"][p]
            e_raw, n = ctx.emb_frames(pc, "raw")
            e_ssl, _ = ctx.emb_frames(pc, "ssl")
            d = {"track": lab, "piece": p, "span": [min(pc), max(pc)], "len": len(pc), "n_clean": n,
                 "src": Bv["piece_srcs"][p]}
            if v == "final_d":
                d["kind"], d["slot"] = Bv["piece_kind"][p], Bv["piece_slot"].get(p)
            else:
                d["kind"] = "debris" if Bv["piece_debris"][p] else ("embedded" if Bv["piece_emb"][p] else "no_app")
            rows.append(d)
            embs.append((e_raw, e_ssl))
    for a, d in enumerate(rows):
        d["sims_ssl_raw"] = {f"{rows[b]['track']}{rows[b]['piece']}": [cos(embs[a][1], embs[b][1]), cos(embs[a][0], embs[b][0])]
                             for b in range(len(rows)) if b != a and embs[a][0] is not None and embs[b][0] is not None}
    return rows


def overlap_blocks(ctx: Ctx, v: str, units: list[dict], max_overlap: int = 5, min_ssl: float = 0.8,
                   min_raw: float = 0.85) -> list[dict]:
    """Closed set only: pairs of pieces on the two tracks that are one person by appearance (ssl and raw) and
    overlap in time by 1..max_overlap frames. link_closed's 'no overlap within a slot' clique rows forbid keeping
    such a pair in one slot, so the person's continuation is pushed into another slot."""
    out = []
    for a in units:
        for b in units:
            if a["track"] != "A" or b["track"] != "B":
                continue
            ov = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0]) + 1
            s = a["sims_ssl_raw"].get(f"B{b['piece']}")
            if 1 <= ov <= max_overlap and s and s[0] is not None and s[0] >= min_ssl and s[1] >= min_raw:
                out.append({"A_piece": a["piece"], "B_piece": b["piece"], "overlap_frames": ov, "sim_ssl": s[0],
                            "sim_raw": s[1], "A_span": a["span"], "B_span": b["span"]})
    return out


def percam_relinks(ctx: Ctx, tracks: list[tuple[str, Track]], lo: int, hi: int) -> list[dict]:
    """final_b: per-camera re-link joins crossed by either track in [lo, hi]: consecutive L0 tracklets of one camera
    on the track that belong to one L1 chain, with the re-linker's link (rule, tail frame, head frame, cost) when the
    two were linked directly."""
    Bb = ctx.B["final_b"]
    L0 = Bb["labels"]["L0"]
    if not hasattr(ctx, "l0_span"):
        ctx.l0_span, ctx.l1_of = {}, {}
        for (cam, i), l0 in L0.items():
            kk = int(ctx.data.cams[cam].k[i])
            a, b = ctx.l0_span.get(l0, (kk, kk))
            ctx.l0_span[l0] = (min(a, kk), max(b, kk))
            ctx.l1_of[l0] = Bb["labels"]["L1"][(cam, i)]
    out = []
    for lab, t in tracks:
        for cam in ctx.data.cams:
            seq = []
            for kk in range(lo, hi + 1):
                for c, i in (t[kk].boxes if kk in t else []):
                    l0 = L0.get((c, int(i)))
                    if c == cam and l0 and (not seq or seq[-1] != l0):
                        seq.append(l0)
            pairs = []                                   # X -> the next L0 of the same L1 chain on the track
            for a, X in enumerate(seq):
                Y = next((Z for Z in seq[a + 1:] if ctx.l1_of[Z] == ctx.l1_of[X] and Z != X), None)
                if Y is not None and (X, Y) not in pairs:
                    pairs.append((X, Y))
            for X, Y in pairs:
                tail, head = ctx.l0_span[X][1], ctx.l0_span[Y][0]
                out.append({"track": lab, "cam": cam, "L0_from": X, "L0_to": Y, "L1": ctx.l1_of[X], "tail": tail,
                            "head": head, "link": [[l[0], int(l[1]), int(l[2]), round(float(l[3]), 3)]
                                                   for l in Bb["relink1_links"][cam] if l[1] == tail and l[2] == head]})
    return out


def switch_cause(ctx: Ctx, v: str, q: int, k0, k1, jt: str, det: dict, blocks: list[dict], relinks: list[dict]) -> str:
    """Cause category of one rater-seen identity change between shown crops at frames k0 -> k1 of output track q."""
    if k0 is None or k1 is None:
        return "no_crop"
    lo, hi = min(k0, k1), max(k0, k1)
    if jt == "builder":
        return "source_builder"
    if jt in ("percam_relink", "fused_relink", "absorb_short"):
        return f"source_{jt}"
    if jt == "xview_pairing":
        # custody goes from one camera's tracklet to the other's inside one pair_views component: if a per-camera
        # re-link join with a recorded link lies between the crops, the re-linker made the decisive join
        if any(r["link"] and lo <= r["tail"] <= hi for r in relinks):
            return "source_percam_relink"
        f = det["fused"]
        return "source_xview_pairing" if (f["fused_far_1m"] or f["partner_disagree"]) else "source_xview_chain"
    if jt == "linker_dup_merge":
        return "linker_dup_merge"
    if jt == "linker_junction":
        js = linker_junctions(ctx, v, q, lo, hi)
        kinds = {str(j["made_by"]) for j in js}
        if v == "final_d":
            span_pieces = {j["piece_from"] for j in js} | {j["piece_to"] for j in js}
            if any(str(b["A_piece"]) in span_pieces or str(b["B_piece"]) in span_pieces for b in blocks):
                return "linker_slot_overlap_constraint"
            if any("short" in x for x in kinds):
                return "linker_short_attach"
            return "linker_milp"
        if "debris_attach" in kinds:
            return "linker_debris_attach"
        return "linker_motion_merge" if "motion_merge" in kinds else "linker_app_merge"
    return jt


# Adjudicated causes of the audited SWAPs (evidence: rows of this script + the sheets + the raters' reasons).
ADJUDICATED = {
    ("final_d", 1): ("linker_slot_overlap_constraint",
                     "A's red-striped player continues in piece 20 (cam2 2:56 + duplicate-merged cam1 1:46; ssl 0.99 / "
                     "raw 0.94 to A's piece 4), which starts on the last frame of piece 4 (1-frame overlap, other "
                     "camera, missed by xview because the cam2 tracklet restarted at the crossing). The MILP 'no "
                     "overlap in a slot' rows push piece 20 into the Y spare slot (B), which then runs on to the "
                     "goalkeeper (piece 23, ssl -0.22)."),
    ("final_d", 8): ("linker_slot_overlap_constraint",
                     "B's player continues as piece 164 (cam2 2:518 from 1044, ssl 0.99 to B's piece 154) which overlaps "
                     "piece 154 (cam1 1:397, to 1045) by 2 frames: forbidden in B's slot, so the MILP gives it to A's "
                     "slot (A's own piece ended at 1020 and was bridged by 1-frame short attachments); B's slot gets "
                     "A's player in short piece 1389 (ssl 0.93 to A's piece 138). The pair rule (ssl and raw) flags it."),
    ("final_d", 13): ("linker_short_attach",
                      "Transient: both tracks' long clean pieces are self-consistent (A 218~230 ssl 0.99, B 220~226 "
                      "0.97, A vs B -0.1). Between 1389 and 1431 A is a chain of 1-5 frame short attachments and a "
                      "MILP piece without clean crops (224), B a 2.3 s jump to short piece 1724: the crops the rater "
                      "saw come from those pieces."),
    ("final_d", 14): ("linker_short_attach",
                      "Transient: A's long pieces 245~255 (ssl 1.00), B's 244~253~258 (1.00). A's -2/-1/+1 s crops are "
                      "three 1-frame cam1 short attachments 1-1.5 s apart (1603, 1618, 1628) that belong to other "
                      "players. Contributing: B's piece 244 holds two duplicate-merged cam1 tracklets with a 3.3 m "
                      "view gap (merge_duplicates), which may have taken A's own cam1 boxes."),
    ("final_d", 20): ("linker_short_attach",
                      "Far-side, single-view: A's MILP piece 323 has no clean crops; both tracks jump at +2 s into "
                      "short attachments (A 2232 after a 1.2 s gap, B 1-frame 2235 after 1 s)."),
    ("final_d", 23): ("linker_milp",
                      "B (Y spare slot) carries the striped-jacket player (pieces 328, 333); his continuation 335 "
                      "(ssl 0.99 / raw 0.93 to 333, no overlap) goes to A's regular slot 0 (whose later piece 347 is the "
                      "same player), B's slot gets piece 334 without clean crops. One player held by two slots at "
                      "different times: a closed-set assignment / spare-slot duplicate, not a forced constraint."),
    ("final_b", 5): ("source_percam_relink",
                     "Inside source tracklet s7 (647 frames): cam1 1:189 + cam2 2:335 are fused and consistent to 637; "
                     "then cam2's re-linker (margin rule, cost 0.52 of max 0.6) joins 2:335 (ends 640) to 2:367 "
                     "(starts 643), another player, and cam1 has stopped. A's red-jersey player reappears in B's "
                     "piece 331 (ssl 0.99 / raw 0.97 to s7)."),
    ("final_b", 7): ("linker_debris_attach",
                     "B is a chain of 1-2 frame debris pieces (636-646, attached on motion alone, gaps up to 0.5 s) in "
                     "front of piece 331 (the red-jersey player, ssl -0.27 to A); the -1 s crop is debris of another "
                     "player."),
    ("final_b", 22): ("linker_debris_attach",
                      "B = piece 826 (no clean crops) + 1-frame debris attached every 0.3-1.5 s (1745 ... 1793): a "
                      "motion-only chain that hops between players near the goal."),
    ("final_b", 23): ("linker_debris_attach",
                      "Same debris chain as ev 22 (track 15): 1-frame pieces at 1760, 1769, 1790, 1793; at +2 s it "
                      "sits on A's player (duplicate)."),
}



# ---------------------------------------------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------------------------------------------

def run(cache_path: str | None = None, rebuild: bool = False) -> dict:
    data = Data()
    B = build(data, cache_path, rebuild)
    ctx = Ctx(data, B, load_feats(data))
    summary = json.load(open(os.path.join(AUDIT, "_blind", "summary.json"), encoding="utf-8"))
    rows = []
    for v in VARIANTS:
        # the saved result (positions rounded to mm) is what make_event_audit selected from; the re-run has the
        # same boxes but unrounded positions, which can move an encounter across the 1.2 m threshold
        full = load_tracks(os.path.join(ROOT, "experiments", "results", f"{v}.json"))
        B[v]["out"] = full
        keep = [q for q, t in enumerate(full) if len(t) >= 5 * data.rate]
        tracks, teams, chosen = select_events(full, data, max_n=24)     # the audit ran make_event_audit --max 24
        ref = json.load(open(os.path.join(AUDIT, v, "events", "events.json"), encoding="utf-8"))["sampled"]
        for e, (k, i, j) in enumerate(chosen, 1):
            if e > len(ref) or e > 24:
                break
            assert ref[e - 1]["k"] == k and ref[e - 1]["team"] == teams[i], (v, e, k, ref[e - 1])
            rows.append(analyse_event(ctx, v, e, k, keep[i], keep[j], verdict_of(v, e, summary)))
    return {"rebuild_checks": {v: {"rerun_matches_saved": B[v]["matches_saved"]} for v in VARIANTS} |
            {"final_b_source_matches_best_source": B["final_b"]["src_matches_best_source"]},
            "summary": summarise(rows), "global": global_stats(ctx), "rows": rows}


def _rule_counts(rows: list[dict], key: str, kind: str) -> dict:
    """Pair rule and per-track flags over events: caught / missed / undecidable (missing clean crops)."""
    c = Counter()
    for r in rows:
        d = r[key][kind]
        c["pair_" + {True: "swap", False: "keep", None: "n/a"}[d["rule_pair"]]] += 1
        f = [d["flag_A"], d["flag_B"]]
        c["any_track_" + ("swap" if True in f else "keep" if False in f else "n/a")] += 1
    return dict(sorted(c.items()))


def summarise(rows: list[dict]) -> dict:
    out = {}
    for v in VARIANTS:
        R = [r for r in rows if r["variant"] == v]
        sw = [r for r in R if r["verdict"] == "SWAP"]
        ok = [r for r in R if r["verdict"] == "OK"]
        o = {"verdicts": dict(Counter(r["verdict"] for r in R)),
             "swap_causes": dict(Counter(r["cause"] for r in sw)),
             "swap_causes_auto": dict(Counter(r["cause_auto"] for r in sw)),
             "auto_agrees": sum(r["cause"] == r["cause_auto"] for r in sw),
             "linker_junction_within_3s": {vd: f"{sum(1 for r in G if any(r['junction_within_3s'].values()))}/{len(G)}"
                                           for vd, G in (("SWAP", sw), ("OK", ok))},
             "junction_kinds_within_3s": {vd: dict(Counter(x for r in G for x in r["junction_kinds_3s"]))
                                          for vd, G in (("SWAP", sw), ("OK", ok))},
             "control_join_at_crossing": {vd: dict(Counter(x for r in G for x in r["control_joins"].values()))
                                          for vd, G in (("SWAP", sw), ("OK", ok))},
             "overlap_blocks": {vd: sum(1 for r in G if r["overlap_blocks"]) for vd, G in (("SWAP", sw), ("OK", ok))}}
        o["swap_stage"] = dict(Counter(r["cause"].split("_")[0] for r in sw))
        o["appearance_rule"] = {f"{key}:{kind}": {vd: _rule_counts(G, key, kind) for vd, G in (("SWAP", sw), ("OK", ok))}
                                for key in ("appearance", "appearance_ext") for kind in ("raw", "ssl", "ssl2")}
        o["swap_rule_hits"] = {f"ev{r['ev']:02d}": {f"{key}:{kind}": [r[key][kind]["rule_pair"], r[key][kind]["flag_A"],
                                                                   r[key][kind]["flag_B"]]
                                                    for key in ("appearance", "appearance_ext") for kind in ("raw", "ssl")}
                               for r in sw}
        out[v] = o
    return out


def global_stats(ctx: Ctx) -> dict:
    """Window-wide frequency of the two failure patterns found at the audited swaps.
    final_d: (a) output points that come from short attached tracklets (< 1 s, no appearance), and how many of those
    are 1-frame pieces; (b) 'overlap splits': an embedded piece whose same-person continuation (ssl >= 0.8 and raw >=
    0.85, both embedded) starts 0-5 frames before it ends, i.e. overlaps it, and therefore sits in another slot.
    final_b: points from debris attachments, and debris chains (>= 3 consecutive debris pieces in one output track)."""
    out = {}
    Bd = ctx.B["final_d"]
    own = Bd["owner_piece"]
    pts = sum(len(o) for o in own)
    short_pts = sum(1 for o in own for q in o.values() if Bd["piece_kind"][q] == "short")
    pieces_used = {q for o in own for q in o.values()}
    short_used = [q for q in pieces_used if Bd["piece_kind"][q] == "short"]
    slot_of = {}
    for s, o in enumerate(own):
        for q in o.values():
            slot_of[q] = s
    emb = {}
    for q in pieces_used:
        if Bd["piece_kind"][q] == "milp" and Bd["piece_len"][q] >= 5:
            e_r, n = ctx.emb_frames(Bd["pieces_boxes"][q], "raw")
            e_s, _ = ctx.emb_frames(Bd["pieces_boxes"][q], "ssl")
            if e_r is not None and n >= 3:
                emb[q] = (e_r, e_s, min(Bd["pieces_boxes"][q]), max(Bd["pieces_boxes"][q]))
    splits = []
    for p, (pr, ps, p0, p1) in emb.items():
        for q, (qr, qs, q0, q1) in emb.items():
            if q != p and p1 - 4 <= q0 <= p1 and q1 > p1 and float(ps @ qs) >= 0.8 and float(pr @ qr) >= 0.85:
                splits.append({"piece": p, "next": q, "overlap": p1 - q0 + 1, "same_slot": slot_of[p] == slot_of[q],
                               "k": int(p1)})
    out["final_d"] = {"box_points": pts, "short_attached_points": short_pts,
                      "short_pieces_attached": len(short_used),
                      "short_pieces_1frame": sum(1 for q in short_used if Bd["piece_len"][q] == 1),
                      "overlap_splits": len(splits), "overlap_splits_other_slot": sum(1 for x in splits if not x["same_slot"]),
                      "overlap_split_list": splits}
    Bb = ctx.B["final_b"]
    pts = deb_pts = chains = 0
    for o in Bb["owner_piece"]:
        seq = []
        for k in sorted(o):
            q = o[k]
            pts += 1
            d = isinstance(q, int) and Bb["piece_debris"][q]
            deb_pts += bool(d)
            if not seq or seq[-1] != q:
                seq.append(q)
        run = 0
        for q in seq:
            if isinstance(q, int) and Bb["piece_debris"][q]:
                run += 1
            else:
                chains += run >= 3
                run = 0
        chains += run >= 3
    out["final_b"] = {"box_points": pts, "debris_points": deb_pts, "debris_chains_ge3": chains}
    return out


def print_rows(res: dict) -> None:
    for r in res["rows"]:
        if r["verdict"] == "UNSURE":
            continue
        a, e = r["appearance"]["ssl"], r["appearance_ext"]["ssl"]
        ar, er = r["appearance"]["raw"], r["appearance_ext"]["raw"]
        f = lambda d: f"sA {d['same_A']} sB {d['same_B']} xAB {d['cross_AB']} xBA {d['cross_BA']} -> {d['rule_pair']}"
        print(f"{r['variant']} ev{r['ev']:02d} {r['t']} k{r['k']} {r['verdict']:5s} cause={r['cause']} "
              f"junctions={r['junction_within_3s']}")
        if r["verdict"] == "SWAP":
            print(f"    raw13 {f(ar)} | ssl13 {f(a)}")
            print(f"    rawExt {f(er)} | sslExt {f(e)}")
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps({v: {k: x for k, x in g.items() if k != "overlap_split_list"} for v, g in res["global"].items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="pickle the rebuilt stages here (and reuse them)")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    res = run(a.cache, a.rebuild)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=1, default=str)
    print_rows(res)
    print(f"wrote {OUT_JSON}")
