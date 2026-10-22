"""Open-set identity linking: agglomerative, constrained (GTA-style) merging of tracklets into identities.

The alternative to link_closed (ids='v2gta' in run.build_tracks). No number of players is assumed; tracklets are
merged cheapest pair first until no feasible merge is cheap enough:

  1. purity first: the input tracklets are cut where the team flips for good (identity.split_team_flips) and,
     optionally, where their clean-crop ReID forms several clusters (identity.dbscan_split);
  2. agglomerative merging under hard cannot-links (two boxes of one camera in one frame, A vs B teams, an
     impossible run at vmax with the anisotropic covariances at any hand-over of the merged timeline).
     cost = appearance distance (crop-weighted average linkage of identity.app_distance between members, or the
     distance of the pooled embeddings) + a motion term for short gaps + a penalty for an undecided team; gaps
     > long_gap_s need a stricter appearance distance; tracklets without clean crops (far side, small, overlapped)
     merge on motion + team only when the gap is short and the choice is unambiguous (best vs second best in motion
     z-score, counting only rivals that exclude each other). Two clusters may share frames when their boxes there
     come from different cameras (one player's two views that the cross-view pairing left apart): the shared frames
     are fused (identity.fuse). Merging stops at merge_thr.
  3. debris (< 1 s and no clean crops: mostly the overlapped boxes the per-camera builder emits as single samples at
     crossings) stays out of the agglomeration and is attached afterwards to the one identity that clearly explains
     it (strict rule, the default: every piece is scored against the identities as they were before any debris
     joined, so debris never chains through debris; gap <= 0.5 s; the runner-up 2 sigma worse). Unattached debris
     and identities shorter than min_len_s are dropped.
  4. Savitzky-Golay smoothing and Hermite gap filling (<= 3 s) of the final identities.

    link_tracklets            one agglomeration over the whole input (fine up to ~10 minutes)
    link_tracklets_windowed   a full match: overlapping windows linked independently, then chained
                              (link_closed.chain_windows)

Ported from experiments/link_gta.py with the settings of candidate 'final_b2' (experiments/combine.py
_gta_params: min_len_s 5; experiments/fix2.py 'strict_debris'), on the tracklets of sources.relink_source. Blind
visual audit of all same-team crossings on the 2026-10-08 test match: 3/31 swaps (9.7 %) against 7/36 for the
closed-set final_d2h and 6/20 for final_d, identity purity 0.97. Being open-set it shows more people than play
(goalkeepers, substitutes and off-court people get identities of their own) and fragments where a crossing has no
clean crops; link_closed fills 5 + 1 slots per team instead.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .boxes import CamBoxes, Track, TrackPoint
from .identity import (NO_BOX_VAR, app_distance, box_covariances, clean_masks, dbscan_split, fill_gaps, fuse,
                       normalise, smooth as savgol_smooth, split_team_flips, track_embedding)
from .link_closed import WindowedStats, chain_windows, default_border_exempt

INF = float("inf")


@dataclass(frozen=True)
class GtaParams:
    # purity first
    split_team: bool = True              # cut the input tracklets at lasting team flips
    dbscan_eps: float | None = None      # DBSCAN purity split (None = off); OSNet x1.0 needs eps 0.2-0.3, not 0.55
    dbscan_min_samples: int = 15
    # appearance
    linkage: str = "average"             # 'average': crop-weighted mean of member-pair distances;
                                         # 'medoid': distance of the pooled cluster embeddings
    min_crops: int = 3                   # clean crops for a tracklet embedding (identity.track_embedding)
    merge_thr: float = 0.35              # stop when the cheapest feasible merge costs more
    long_gap_s: float = 8.0              # gaps longer than this need ...
    long_thr: float = 0.25               # ... appearance distance <= long_thr
    max_gap_s: float | None = None       # never link across longer gaps (None = no limit)
    u_penalty: float = 0.05              # a team-'U' side pays this
    team_camera_conflict: bool = True    # cluster team 'U' when the two cameras read the shirt oppositely
                                         # (identity.team_vote); False = the research behaviour
    # motion
    vmax: float = 9.0                    # reachability: vmax * dt + slack + 3 sigma
    slack: float = 1.5
    motion_gap_s: float = 2.0            # the motion term applies to hand-overs up to this gap
    w_motion: float = 0.05               # cost per unit of (z / 3)^2, z = motion prediction error in sigmas (a tie-break:
                                         # appearance orders the merges better than noisy far-side velocities)
    sig_v: float = 1.5                   # m/s: velocity uncertainty of the constant-velocity prediction
    sig_v_none: float = 4.0              # m/s: same when neither side has a velocity (singletons)
    vel_win: int = 5                     # frames used for the end velocities
    overlap_gate_m: float = 1.0          # shared (cross-camera) frames: median distance gate
    # tracklets without clean crops
    noapp_gap_s: float = 1.5
    noapp_z: float = 2.5                 # motion gate (sigmas)
    noapp_margin: float = 1.0            # best rival must be this many sigmas worse
    noapp_base: float = 0.15             # base cost (orders motion-only merges among appearance merges)
    retry_rounds: int = 2                # re-try ambiguous motion-only merges after the other merges settled
    # debris: tracklets shorter than debris_len_s without clean crops, attached after the agglomeration
    debris_len_s: float = 1.0
    strict_debris: bool = True           # score every piece against the identities before any debris joined ...
    strict_gap_s: float = 0.5            # ... across hand-overs up to this gap ...
    strict_margin: float = 2.0           # ... with the runner-up this many sigmas worse (same z gate noapp_z)
    debris_gap_s: float = 1.5            # strict_debris=False (research final_b): longest piece first against the
                                         # current clusters (chains allowed), this gap, noapp_margin
    # output
    min_len_s: float = 5.0               # drop final identities shorter than this (unattached debris, short
                                         # off-court people; 1 s left 21 ids instead of 16 on the test match)
    fill_gap_s: float = 3.0              # identity.fill_gaps
    smooth: str = "both"                 # Savitzky-Golay: 'pre' (before the fill, so the Hermite end velocities are not
                                         # jitter), 'post', 'both' or 'none' (when refine.refine_positions follows)


@dataclass
class GtaStats:
    inputs: int = 0
    after_split: int = 0
    embedded: int = 0
    merges_app: int = 0
    merges_motion: int = 0
    debris: int = 0
    debris_attached: int = 0
    ambiguous: int = 0
    outputs: int = 0
    dropped_points: int = 0
    costs: list = field(default_factory=list)
    output_team: list = field(default_factory=list)   # 'A' / 'B' / 'U' per output identity (its cluster's vote)


# ---------------------------------------------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _Cluster:
    members: list[int]
    track: Track
    ks: np.ndarray            # sorted frames
    xy: np.ndarray            # (n, 2)
    cov: np.ndarray           # (n, 2, 2)
    cams: np.ndarray          # (n,) bitmask of the cameras with a box at that frame
    vb: np.ndarray            # (n, 2) velocity from the samples before (NaN if none within vel_win)
    vf: np.ndarray            # (n, 2) velocity from the samples after
    asum: float               # team_vote evidence: sum of h over labelled 'A' boxes ...
    wsum: float               # ... and over all labelled boxes
    cam_lab: dict             # cam -> [labelled boxes, of them 'A'] (team_vote's camera conflict)
    rows: np.ndarray          # embedded members' rows in the tracklet appearance matrix
    wts: np.ndarray           # their clean-crop counts
    emb: dict | None          # pooled embedding (medoid linkage)
    team: str = "U"


def _team(asum: float, wsum: float, cam_lab: dict, camera_conflict: bool, min_cam_labels: int = 10) -> str:
    """identity.team_vote from additive sums (same thresholds, same camera-conflict rule)."""
    if wsum <= 0:
        return "U"
    if camera_conflict:
        shares = [a / n for n, a in cam_lab.values() if n >= min_cam_labels]
        if len(shares) >= 2 and max(shares) >= 0.7 and min(shares) <= 0.3:
            return "U"
    s = asum / wsum
    return "A" if s >= 0.7 else "B" if s <= 0.3 else "U"


def _lam_max(R: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of symmetric 2x2 matrices (..., 2, 2), closed form."""
    a, b, c = R[..., 0, 0], R[..., 0, 1], R[..., 1, 1]
    return 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)


def _velocities(ks: np.ndarray, xy: np.ndarray, rate: float, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-point velocities over up to `win` frames before / after each sample (NaN where no other sample)."""
    idx = np.arange(len(ks))
    j = np.searchsorted(ks, ks - win)
    vb = np.full_like(xy, np.nan)
    ok = j < idx
    vb[ok] = (xy[ok] - xy[j[ok]]) / ((ks[ok] - ks[j[ok]]) / rate)[:, None]
    j = np.searchsorted(ks, ks + win, side="right") - 1
    vf = np.full_like(xy, np.nan)
    ok = j > idx
    vf[ok] = (xy[j[ok]] - xy[ok]) / ((ks[j[ok]] - ks[ok]) / rate)[:, None]
    return vb, vf


class _Pair(NamedTuple):
    cost: float
    z: float
    gap: float
    motion_only: bool
    hard: bool               # refused by a cannot-link
    permanent: bool          # ... that no later merge can lift


def _pool(e1: dict | None, e2: dict | None, max_medoids: int = 30) -> dict | None:
    """Pooled embedding of two clusters: crop-weighted mean, union of medoids (evenly thinned)."""
    if e1 is None or e2 is None:
        return e1 if e2 is None else e2
    meds = np.concatenate([e1["medoids"], e2["medoids"]])
    if len(meds) > max_medoids:
        meds = meds[np.linspace(0, len(meds) - 1, max_medoids).astype(int)]
    return {"mean": normalise(e1["n"] * e1["mean"] + e2["n"] * e2["mean"]), "medoids": meds, "n": e1["n"] + e2["n"],
            "span": (min(e1["span"][0], e2["span"][0]), max(e1["span"][1], e2["span"][1]))}


def _app_matrix(embs: list[dict]) -> np.ndarray:
    """All-pairs identity.app_distance (min of mean distance and 20th percentile of medoid distances), vectorised."""
    n = len(embs)
    if not n:
        return np.zeros((0, 0))
    M = np.stack([e["mean"] for e in embs])
    Dm = 1.0 - M @ M.T
    m = max(len(e["medoids"]) for e in embs)
    med = np.zeros((n, m, M.shape[1]), np.float32)
    valid = np.zeros((n, m), bool)
    for i, e in enumerate(embs):
        med[i, :len(e["medoids"])] = e["medoids"]
        valid[i, :len(e["medoids"])] = True
    P = np.empty((n, n))
    for i, e in enumerate(embs):
        S = 1.0 - np.einsum("jad,bd->jab", med, e["medoids"].astype(np.float32))      # (n, m, m_i)
        S[~valid] = np.nan
        P[i] = np.nanpercentile(S.reshape(n, -1), 20, axis=1)
    return np.clip(np.minimum(Dm, P), 0.0, 2.0)


class Linker:
    """Agglomerative constrained linker over a fixed list of input tracklets (module docstring, steps 2-3)."""

    def __init__(self, tracklets: list[Track], cams: dict[str, CamBoxes], rate: float, params: GtaParams,
                 clean: dict[str, np.ndarray], stats: GtaStats | None = None):
        self.cams_, self.p = cams, params
        self.st = stats if stats is not None else GtaStats()
        self.rate = float(rate)
        self.cam_bit = {cam: 1 << q for q, cam in enumerate(cams)}
        self.box_cov = {cam: box_covariances(cams, cam) for cam in cams}
        embs = [track_embedding(t, cams, clean, min_crops=params.min_crops) for t in tracklets]
        self.row_of = {}
        rows = []
        for i, e in enumerate(embs):
            if e is not None:
                self.row_of[i] = len(rows)
                rows.append(e)
        self.embs = embs
        self.D = _app_matrix(rows)
        self.st.embedded = len(rows)
        self.clusters: dict[int, _Cluster] = {}
        for i, t in enumerate(tracklets):
            self.clusters[i] = self._from_track(t, [i])
        self.next_id = len(tracklets)

    # --- cluster construction -------------------------------------------------------------------------------
    def _point_cov(self, p: TrackPoint) -> np.ndarray:
        if not p.boxes:
            return NO_BOX_VAR * np.eye(2)
        if len(p.boxes) == 1:
            cam, i = p.boxes[0]
            return self.box_cov[cam][i]
        return fuse(self.cams_, p.boxes)[1]

    def _team_evidence(self, t: Track) -> tuple[float, float, dict]:
        """team_vote's evidence (h >= 40, conf >= 0.4, clear team) as additive sums."""
        a_s = ws = 0.0
        per: dict[str, list[int]] = {}
        for p in t.values():
            for cam, i in p.boxes:
                c = self.cams_[cam]
                if c.h[i] >= 40 and c.conf[i] >= 0.4 and c.team[i]:
                    ws += c.h[i]
                    a_s += c.h[i] * (c.team[i] == "A")
                    n = per.setdefault(cam, [0, 0])
                    n[0] += 1
                    n[1] += c.team[i] == "A"
        return a_s, ws, per

    def _finish(self, cl: _Cluster) -> _Cluster:
        cl.team = _team(cl.asum, cl.wsum, cl.cam_lab, self.p.team_camera_conflict)
        return cl

    def _from_track(self, t: Track, members: list[int]) -> _Cluster:
        ks = np.array(sorted(t), int)
        xy = np.stack([t[k].xy for k in ks]).astype(float)
        cov = np.stack([self._point_cov(t[k]) for k in ks])
        cams = np.array([sum(self.cam_bit[c] for c in {cam for cam, _ in t[k].boxes}) for k in ks], int)
        vb, vf = _velocities(ks, xy, self.rate, self.p.vel_win)
        a_s, ws, per = self._team_evidence(t)
        rows = np.array([self.row_of[m] for m in members if m in self.row_of], int)
        wts = np.array([self.embs[m]["n"] for m in members if m in self.row_of], float)
        emb = self.embs[members[0]] if len(members) == 1 else None
        return self._finish(_Cluster(members, dict(t), ks, xy, cov, cams, vb, vf, a_s, ws, per, rows, wts, emb))

    def _merge(self, A: _Cluster, B: _Cluster) -> _Cluster:
        shared = np.intersect1d(A.ks, B.ks, assume_unique=True)
        track = dict(A.track)
        track.update(B.track)
        ks = np.concatenate([A.ks, B.ks])
        xy = np.concatenate([A.xy, B.xy])
        cov = np.concatenate([A.cov, B.cov])
        cams = np.concatenate([A.cams, B.cams])
        keep = np.ones(len(ks), bool)
        if len(shared):
            ia = np.searchsorted(A.ks, shared)
            ib = len(A.ks) + np.searchsorted(B.ks, shared)
            keep[ib] = False
            for q, k in enumerate(shared.tolist()):
                boxes = sorted(A.track[k].boxes + B.track[k].boxes)
                xy[ia[q]], cov[ia[q]] = fuse(self.cams_, boxes)
                track[k] = TrackPoint(xy[ia[q]].copy(), boxes)
                cams[ia[q]] |= cams[ib[q]]
        ks, xy, cov, cams = ks[keep], xy[keep], cov[keep], cams[keep]
        order = np.argsort(ks, kind="stable")
        ks, xy, cov, cams = ks[order], xy[order], cov[order], cams[order]
        vb, vf = _velocities(ks, xy, self.rate, self.p.vel_win)
        rows = np.concatenate([A.rows, B.rows])
        wts = np.concatenate([A.wts, B.wts])
        emb = _pool(A.emb, B.emb) if self.p.linkage == "medoid" else None
        cam_lab = {c: [A.cam_lab.get(c, [0, 0])[0] + B.cam_lab.get(c, [0, 0])[0],
                       A.cam_lab.get(c, [0, 0])[1] + B.cam_lab.get(c, [0, 0])[1]]
                   for c in A.cam_lab.keys() | B.cam_lab.keys()}
        return self._finish(_Cluster(A.members + B.members, track, ks, xy, cov, cams, vb, vf, A.asum + B.asum,
                                     A.wsum + B.wsum, cam_lab, rows, wts, emb))

    # --- pair cost ------------------------------------------------------------------------------------------
    def _app(self, A: _Cluster, B: _Cluster) -> float | None:
        if self.p.linkage == "medoid":
            return app_distance(A.emb, B.emb) if A.emb is not None and B.emb is not None else None
        if not len(A.rows) or not len(B.rows):
            return None
        return float(A.wts @ self.D[np.ix_(A.rows, B.rows)] @ B.wts / (A.wts.sum() * B.wts.sum()))

    def evaluate(self, A: _Cluster, B: _Cluster) -> _Pair:
        """Cost of merging A and B (INF when they cannot link now), the motion z-score of the merge, its largest
        hand-over gap, whether it rests on motion only, and whether the refusal is a hard cannot-link (A vs B teams,
        two boxes of one camera in one frame, co-located frames too far apart, an impossible run) or even a
        permanent one."""
        p = self.p
        ta, tb = A.team, B.team
        if {ta, tb} == {"A", "B"}:
            return _Pair(INF, INF, INF, False, True, True)
        nA, nB = len(A.ks), len(B.ks)
        z_overlap = pen_overlap = 0.0
        if A.ks[-1] < B.ks[0] or B.ks[-1] < A.ks[0]:               # fast path: one hand-over
            E, L = (A, B) if A.ks[-1] < B.ks[0] else (B, A)
            k1, k2 = E.ks[-1:], L.ks[:1]
            x1, x2, R = E.xy[-1:], L.xy[:1], E.cov[-1:] + L.cov[:1]
            v1, v2 = E.vb[-1:], L.vf[:1]
        else:
            shared = np.intersect1d(A.ks, B.ks, assume_unique=True)
            if len(shared):
                ia, ib = np.searchsorted(A.ks, shared), np.searchsorted(B.ks, shared)
                if np.any(A.cams[ia] & B.cams[ib]):
                    return _Pair(INF, INF, INF, False, True, True)
                dist = np.linalg.norm(A.xy[ia] - B.xy[ib], axis=1)
                med = float(np.median(dist))
                if med > p.overlap_gate_m:
                    return _Pair(INF, INF, INF, False, True, True)
                pen_overlap = (med / p.overlap_gate_m) ** 2
                z_overlap = med / float(np.sqrt(np.median(_lam_max(A.cov[ia] + B.cov[ib]))))
            ks = np.concatenate([A.ks, B.ks])
            own = np.concatenate([np.zeros(nA, np.int8), np.ones(nB, np.int8)])
            order = np.lexsort((own, ks))
            o = own[order]
            ch = np.flatnonzero(o[1:] != o[:-1])
            first, second = order[ch], order[ch + 1]
            XY, COV = np.concatenate([A.xy, B.xy]), np.concatenate([A.cov, B.cov])
            k1, k2 = ks[first], ks[second]
            x1, x2, R = XY[first], XY[second], COV[first] + COV[second]
            v1, v2 = np.concatenate([A.vb, B.vb])[first], np.concatenate([A.vf, B.vf])[second]
        dt = (k2 - k1) / self.rate
        lam = _lam_max(R)
        d = np.linalg.norm(x2 - x1, axis=1)
        if np.any(d > p.vmax * dt + p.slack + 3.0 * np.sqrt(lam)):
            return _Pair(INF, INF, INF, False, True, False)
        gap = float(dt.max())
        # motion term on the short hand-overs: constant-velocity prediction from both sides
        m = (dt > 0) & (dt <= p.motion_gap_s)
        z = z_overlap
        if m.any():
            dtm = dt[m][:, None]
            e1 = np.linalg.norm(x2[m] - (x1[m] + v1[m] * dtm), axis=1)
            e2 = np.linalg.norm(x1[m] - (x2[m] - v2[m] * dtm), axis=1)
            n1, n2 = np.isnan(e1), np.isnan(e2)
            e = np.where(n1, e2, np.where(n2, e1, 0.5 * (e1 + e2)))
            none = n1 & n2
            e = np.where(none, d[m], e)
            sv = np.where(none, p.sig_v_none, p.sig_v)
            z = max(z, float(np.max(e / np.sqrt(lam[m] + (sv * dt[m]) ** 2))))
        pen = max(min((z / 3.0) ** 2, 4.0), pen_overlap)
        if p.max_gap_s is not None and gap > p.max_gap_s:
            return _Pair(INF, z, gap, False, False, False)
        u = p.u_penalty if "U" in (ta, tb) else 0.0
        app = self._app(A, B)
        if app is not None:
            if gap > p.long_gap_s and app > p.long_thr:
                return _Pair(INF, z, gap, False, False, False)
            motion = p.w_motion * pen if gap <= p.motion_gap_s else 0.0
            return _Pair(app + motion + u, z, gap, False, False, False)
        if gap > p.noapp_gap_s or z > p.noapp_z:
            return _Pair(INF, z, gap, True, False, False)
        return _Pair(p.noapp_base + p.w_motion * pen + u, z, gap, True, False, False)

    # --- main loop ------------------------------------------------------------------------------------------
    def _candidates(self, ids: list[int]) -> dict[int, set[int]]:
        """Initial partner sets: every pair whose spans come within the motion / no-appearance gap of each other,
        plus embedded pairs at any gap whose tracklet appearance distance could pass merge_thr."""
        p = self.p
        s = np.array([self.clusters[i].ks[0] for i in ids])
        e = np.array([self.clusters[i].ks[-1] for i in ids])
        near = int(np.ceil(max(p.motion_gap_s, p.noapp_gap_s) * self.rate))
        partners: dict[int, set[int]] = {i: set() for i in ids}
        order = np.argsort(s)
        s_sorted = s[order]
        for q, i in enumerate(ids):
            rr = order[:np.searchsorted(s_sorted, e[q] + near, side="right")]
            for r in rr[e[rr] >= s[q] - near].tolist():
                if r != q:
                    partners[i].add(ids[r])
                    partners[ids[r]].add(i)
        emb_ids = [i for i in ids if i in self.row_of]
        if emb_ids:
            rows = np.array([self.row_of[i] for i in emb_ids])
            Dsub = self.D[np.ix_(rows, rows)]
            for a, b in zip(*np.nonzero(np.triu(Dsub <= p.merge_thr, 1))):
                partners[emb_ids[a]].add(emb_ids[b])
                partners[emb_ids[b]].add(emb_ids[a])
        return partners

    def _ambiguous(self, a: int, b: int, z: float, partners: dict[int, set[int]]) -> bool:
        """A motion-only merge of a and b is ambiguous when either side x has a rival c for the same slot (a short
        hand-over with x, within noapp_margin sigmas of the proposed one) that has a hard cannot-link with the other
        side y: c and y cannot both be x's continuation, and motion cannot tell which is."""
        near = int(np.ceil(self.p.noapp_gap_s * self.rate))
        for x, y in ((a, b), (b, a)):
            X, Y = self.clusters[x], self.clusters[y]
            lo, hi = Y.ks[0] - near, Y.ks[-1] + near
            for c in partners[x]:
                C = self.clusters.get(c)
                if c == y or C is None or C.ks[0] > hi or C.ks[-1] < lo:
                    continue
                r = self.evaluate(X, C)
                if r.cost == INF or r.gap > self.p.noapp_gap_s or r.z >= z + self.p.noapp_margin:
                    continue
                if self.evaluate(C, Y).hard:
                    return True
        return False

    def _absorb(self, a: int, b: int, partners: dict[int, set[int]] | None = None) -> int:
        """Merge clusters a and b into a new cluster; returns its id."""
        c = self.next_id
        self.next_id += 1
        self.clusters[c] = self._merge(self.clusters[a], self.clusters[b])
        del self.clusters[a], self.clusters[b]
        if partners is not None:
            partners[c] = (partners.pop(a, set()) | partners.pop(b, set())) - {a, b}
            for q in partners[c]:
                partners[q] -= {a, b}
                partners[q].add(c)
        return c

    def run(self) -> list[_Cluster]:
        """Agglomerative linking of the substantial tracklets (cheapest first, ambiguous motion-only merges retried
        in a few rounds), then debris attachment. Returns the clusters in order of first frame."""
        p = self.p
        debris = [i for i, cl in self.clusters.items()
                  if cl.ks[-1] - cl.ks[0] + 1 < p.debris_len_s * self.rate and not len(cl.rows)]
        self.st.debris = len(debris)
        dset = set(debris)
        ids = [i for i in self.clusters if i not in dset]
        partners = self._candidates(ids)
        heap: list[tuple[float, int, int]] = []

        def push(a: int, b: int) -> None:
            r = self.evaluate(self.clusters[a], self.clusters[b])
            if r.permanent:
                partners[a].discard(b)
                partners[b].discard(a)
            elif r.cost <= p.merge_thr:
                heapq.heappush(heap, (r.cost, a, b))

        for a in ids:
            for b in list(partners[a]):
                if a < b:
                    push(a, b)
        for _ in range(p.retry_rounds + 1):
            deferred: list[tuple[int, int]] = []
            while heap:
                _, a, b = heapq.heappop(heap)
                if a not in self.clusters or b not in self.clusters:
                    continue
                r = self.evaluate(self.clusters[a], self.clusters[b])
                if r.cost > p.merge_thr:
                    continue
                if r.motion_only and self._ambiguous(a, b, r.z, partners):
                    self.st.ambiguous += 1
                    deferred.append((a, b))
                    continue
                self.st.merges_motion += r.motion_only
                self.st.merges_app += not r.motion_only
                self.st.costs.append(round(r.cost, 3))
                c = self._absorb(a, b, partners)
                for q in list(partners[c]):
                    push(c, q)
            for a, b in deferred:
                if a in self.clusters and b in self.clusters:
                    push(a, b)
            if not heap:
                break
        if p.strict_debris:
            self._attach_debris_strict(debris)
        else:
            self._attach_debris_chained(debris)
        return sorted(self.clusters.values(), key=lambda cl: int(cl.ks[0]))

    def _attach_debris_strict(self, debris: list[int]) -> None:
        """Every debris piece is scored against the clusters as they were before any debris joined (debris never
        chains through debris): feasible (no hard cannot-link, gap <= strict_gap_s, z <= noapp_z) and every other
        feasible cluster strict_margin sigmas worse. The decisions are then applied, longest piece first; a piece
        that now collides with another piece on the same cluster (two boxes of one camera in one frame) stays out."""
        p = self.p
        near = int(np.ceil(p.strict_gap_s * self.rate))
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
                if not r.hard and r.gap <= p.strict_gap_s and r.z <= p.noapp_z:
                    scored.append((r.z, c))
            scored.sort()
            if scored and (len(scored) == 1 or scored[1][0] >= scored[0][0] + p.strict_margin):
                decisions.append((d, scored[0][1]))
        cur = {c: c for c in snap}
        for d, c in decisions:
            tgt = cur[c]
            if self.evaluate(self.clusters[d], self.clusters[tgt]).hard:
                continue
            cur[c] = self._absorb(d, tgt)
            self.st.debris_attached += 1

    def _attach_debris_chained(self, debris: list[int]) -> None:
        """The research final_b rule: longest piece first, each against the current clusters (so later pieces see
        the debris attached before them), gap <= debris_gap_s, runner-up noapp_margin sigmas worse."""
        p = self.p
        near = int(np.ceil(p.debris_gap_s * self.rate))
        pending = set(debris)                       # debris never absorbs debris
        for d in sorted(debris, key=lambda i: -len(self.clusters[i].ks)):
            Dc = self.clusters[d]
            lo, hi = Dc.ks[0] - near, Dc.ks[-1] + near
            scored = []
            for c, C in self.clusters.items():
                if c in pending or C.ks[0] > hi or C.ks[-1] < lo:
                    continue
                r = self.evaluate(Dc, C)
                if not r.hard and r.gap <= p.debris_gap_s and r.z <= p.noapp_z:
                    scored.append((r.z, c))
            scored.sort()
            if scored and (len(scored) == 1 or scored[1][0] >= scored[0][0] + p.noapp_margin):
                self._absorb(d, scored[0][1])
                self.st.debris_attached += 1
                pending.discard(d)


# ---------------------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------------------

def link_tracklets(tracklets: list[Track], cams: dict[str, CamBoxes], rate: float, params: GtaParams = GtaParams(),
                   border_exempt: dict[str, tuple[str, ...]] | None = None,
                   stats: GtaStats | None = None) -> list[Track]:
    """Purity split -> constrained agglomerative linking -> debris attachment -> min-length drop -> smoothing and
    gap filling (module docstring). `rate` is the grid rate in Hz; `border_exempt` goes to identity.clean_masks
    (None: link_closed.default_border_exempt, the top edge of every camera; {} for no exemption). Returns the
    identities in order of first frame (stats.output_team: each one's team 'A' / 'B' / 'U'); gap fills carry
    boxes=[]. Every box of the input appears at most once in the output."""
    st = stats if stats is not None else GtaStats()
    p = params
    if p.linkage not in ("average", "medoid"):
        raise ValueError(f"linkage must be 'average' or 'medoid', got {p.linkage!r}")
    if p.smooth not in ("pre", "post", "both", "none"):
        raise ValueError(f"smooth must be 'pre', 'post', 'both' or 'none', got {p.smooth!r}")
    st.inputs = len(tracklets)
    tr = [t for t in tracklets if t]
    if not tr:
        return []
    clean = clean_masks(cams, border_exempt=default_border_exempt(cams) if border_exempt is None else border_exempt)
    if p.split_team:
        tr = split_team_flips(tr, cams)
    if p.dbscan_eps is not None:
        tr = [piece for t in tr for piece in dbscan_split(t, cams, rate, clean, eps=p.dbscan_eps,
                                                          min_samples=p.dbscan_min_samples)]
    st.after_split = len(tr)
    clusters = Linker(tr, cams, rate, p, clean, st).run()
    if p.min_len_s > 0:
        keep = [cl for cl in clusters if len(cl.track) >= p.min_len_s * rate]
        st.dropped_points = sum(len(cl.track) for cl in clusters) - sum(len(cl.track) for cl in keep)
        clusters = keep
    out = [cl.track for cl in clusters]
    if p.smooth in ("pre", "both"):
        out = [savgol_smooth(t) for t in out]
    out = [fill_gaps(t, rate, p.fill_gap_s, p.vmax) if p.fill_gap_s > 0 else t for t in out]
    if p.smooth in ("post", "both"):
        out = [savgol_smooth(t) for t in out]
    st.outputs = len(out)
    st.output_team = [cl.team for cl in clusters]
    return out


def link_tracklets_windowed(tracklets: list[Track], cams: dict[str, CamBoxes], rate: float,
                            params: GtaParams = GtaParams(), window_s: float = 360.0, overlap_s: float = 60.0,
                            border_exempt: dict[str, tuple[str, ...]] | None = None, min_boxes: int = 10,
                            min_agree: float = 0.5, stats: WindowedStats | None = None) -> list[Track]:
    """link_tracklets for a whole match: overlapping windows linked independently and chained into identities by
    the boxes they share in the overlap; every frame belongs to one window (the cut is the middle of the overlap).
    Same scheme and helper as link_closed.assign_identities_windowed (link_closed.chain_windows; see its
    docstring). Why windows here: the appearance matrix and the candidate pairs grow with the square of the number
    of tracklets, and appearance drifts with the light over a match. stats.per_window gets each window's GtaStats,
    stats.output_team the team of each identity's first window. With window_s >= the input span (or less than half
    the overlap short of it) this is exactly one link_tracklets call."""
    st = stats if stats is not None else WindowedStats()

    def solve(clip: list[Track]) -> list[Track]:
        wst = GtaStats()
        st.per_window.append(wst)
        return link_tracklets(clip, cams, rate, params, border_exempt, wst)

    ids, origin = chain_windows(tracklets, rate, solve, window_s, overlap_s, min_boxes, min_agree, st)
    st.output_team = [st.per_window[w].output_team[j] for w, j in origin]
    return ids
