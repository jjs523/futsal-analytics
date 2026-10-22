"""Tracklet source of the open-set linker (ids='v2gta'): long AND pure fused tracklets.

The conservative per-camera builder (tracklets.conservative_tracklets) is pure but short: it cuts at every
occlusion, and the overlapped boxes of a crossing come out as one-sample tracklets. This source re-attaches the
continuation of a cut piece wherever that is unambiguous, before any long-range linking:

    per camera   conservative_tracklets -> relink (local re-link of the pieces around an occlusion: margin rule,
                 plus the agreement rule with a BoostTrack proposal tracker when boxmot is installed)
    two cameras  tracklets.pair_views (cut only where whole-tracklet fusion would collide)
                 -> relink of the fused tracklets (margin rule, pieces >= 5 samples)
                 -> absorb_short (1-2 sample pieces become the second view / a hole fill of the one long track
                 that clearly owns them)

    relink_source(cams, rate) -> list[Track]      the whole chain ('relink' source of the research code)
    relink, absorb_short, boosttrack_tracklets     the steps

Every box of the selected frames ends up in exactly one output tracklet; a TrackPoint carries one box per camera.

Ported from experiments/sources.py (best_source(data, 'relink'), percam_source, relink, absorb_short) and the
BoostTrack runner of experiments/percam.py (boxmot_tracklets). Teams are the generic 'A' / 'B' of
boxes.assign_teams; team votes go through identity.team_vote.

BoostTrack proposals need boxmot (the package that also computes the ReID embeddings, pipeline/reid.py). Without it
the per-camera re-link runs the margin rule only (proposals=None, the research 'relink_pure' source: ~40 % fewer
per-camera links, fewest witnessed swaps); proposals='auto' picks whichever is available and stats.proposals says
which ran.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .boxes import CamBoxes, Track, TrackPoint
from .identity import clean_masks, fuse, normalise, point_cov, team_vote
from .tracklets import XviewStats, conservative_tracklets, default_frames, iou_matrix, pair_views


# ---------------------------------------------------------------------------------------------------------------
# BoostTrack proposals (optional, boxmot)
# ---------------------------------------------------------------------------------------------------------------

def have_boxmot() -> bool:
    try:
        import boxmot.trackers.boosttrack.boosttrack  # noqa: F401
    except Exception:                                   # ImportError, or a broken install (missing light deps)
        return False
    return True


class _CachedReid:
    """Stands in for boxmot's ReID backend: no model is loaded; the tracker gets the boxes' cached embeddings
    through update(embs=...) and never asks for features itself."""

    def __init__(self, weights=None, device=None, half=False):
        self.model = self

    def get_features(self, xyxy: np.ndarray, img: np.ndarray) -> np.ndarray:
        raise RuntimeError("BoostTrack asked for ReID features although embeddings were passed")


def _make_boosttrack(**params):
    from boxmot.trackers.boosttrack import boosttrack as mod
    real = mod.ReidAutoBackend
    mod.ReidAutoBackend = _CachedReid                   # no model load, no device side effects
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return mod.BoostTrack(reid_weights=None, device="cpu", half=False, **params)
    finally:
        mod.ReidAutoBackend = real


def boosttrack_tracklets(cams: dict[str, CamBoxes], cam: str, rate: float, frames: Iterable[int] | None = None,
                         min_conf: float = 0.1, max_age_s: float = 2.0, det_thresh: float = 0.4,
                         iou_threshold: float = 0.3) -> list[Track]:
    """boxmot's BoostTrack over the grid frames of one camera, fed with the boxes' ReID embeddings and a dummy
    image (no camera-motion compensation: the phones stand on tripods). Defaults are the research settings, the
    30 fps ones rescaled to 10 Hz (max_age 20 samples = 2 s, min_hits 1). Output rows are mapped back to box
    indices by det_ind (the detection the tracker associated, even when its Kalman box drifted off it), else to
    the best-overlapping box with IoU > 0.5; a box is never given to two tracks. One box per TrackPoint."""
    c = cams[cam]
    if c.reid is None:
        raise ValueError(f"{cam} has no ReID embeddings")
    trk = _make_boosttrack(max_age=max(1, int(round(max_age_s * rate))), min_hits=1, det_thresh=det_thresh,
                           iou_threshold=iou_threshold, use_ecc=False, with_reid=True)
    W, H = c.frame_wh
    img = np.zeros((int(H), int(W), 3), np.uint8)
    frames = default_frames(cams) if frames is None else frames
    tracks: dict[int, Track] = {}
    sink = io.StringIO()
    for k in frames:
        idx = c.boxes_at(k, min_conf)
        dets = np.hstack([c.xyxy[idx], c.conf[idx][:, None], np.zeros((len(idx), 1))]) if len(idx) else np.empty((0, 6))
        embs = c.reid[idx].astype(np.float64) if len(idx) else np.empty((0, c.reid.shape[1]))
        with contextlib.redirect_stdout(sink):
            out = trk.update(dets, img, embs)
        if sink.tell() > 1 << 20:
            sink.seek(0)
            sink.truncate()
        out = np.asarray(out)
        if out.size == 0 or not len(idx):
            continue
        out = out.reshape(-1, out.shape[-1])
        ious = iou_matrix(out[:, :4].astype(float), c.xyxy[idx])
        taken = set()
        for r, row in enumerate(out):
            j = int(round(row[7])) if np.isfinite(row[7]) else -1
            if not (0 <= j < len(idx) and abs(row[7] - j) < 1e-6):
                j = int(ious[r].argmax())
                if ious[r, j] <= 0.5:
                    continue
            if j in taken:
                continue
            taken.add(j)
            i = int(idx[j])
            tracks.setdefault(int(row[4]), {})[k] = TrackPoint(c.xy[i].copy(), [(cam, i)])
    return sorted(tracks.values(), key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# Local re-link of pieces cut at occlusions
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _End:
    """One end (head or tail) of a track, as seen by the local re-linker."""
    k: int                                            # frame of the end sample
    xy: np.ndarray                                    # its pitch position
    cov: np.ndarray                                   # its covariance (identity.point_cov)
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


def _owner(tracks: list[Track], cams: dict[str, CamBoxes]) -> dict[str, np.ndarray]:
    """{cam: track index owning each box, -1 if none}."""
    own = {cam: np.full(len(c.k), -1) for cam, c in cams.items()}
    for q, t in enumerate(tracks):
        for p in t.values():
            for cam, i in p.boxes:
                own[cam][i] = q
    return own


def _lsq_velocity(ks: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares velocity (units per frame) of samples X at frames ks; zero with fewer than 2 samples."""
    if len(ks) < 2:
        return np.zeros(X.shape[1])
    t = ks - ks.mean()
    return (t[:, None] * (X - X.mean(0))).sum(0) / max(float((t ** 2).sum()), 1e-9)


def _end(track: Track, cams: dict[str, CamBoxes], clean: dict[str, np.ndarray], tail: bool, camera_conflict: bool,
         n_vel: int = 5, n_feat: int = 15) -> _End:
    """Tail (tail=True) or head of a track: last / first sample, outward image velocity per camera from the n_vel
    nearest samples, and the mean of the n_feat nearest clean crops (largest box per frame)."""
    ks = np.array(sorted(track), int)
    order = ks[::-1] if tail else ks
    k0 = int(order[0])
    e = _End(k0, track[k0].xy, point_cov(cams, track[k0]),
             team=team_vote(track, cams, camera_conflict=camera_conflict)[0])
    for cam, c in cams.items():
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
            if cams[cam].reid is not None and clean[cam][i] and (
                    best is None or cams[cam].h[i] > cams[best[0]].h[best[1]]):
                best = (cam, i)
        if best is not None:
            feats.append(cams[best[0]].reid[best[1]])
            if len(feats) >= n_feat:
                break
    if feats:
        e.feat = normalise(np.mean(feats, 0))
    return e


def _link_cost(a: _End, b: _End, rate: float, vmax: float, slack: float, w_dh: float, w_app: float,
               app_missing: float, w_gap: float, img_gate: float, app_gate: float) -> float:
    """Cost of continuing tail a with head b (b starts after a ends); np.inf when a hard gate fails: A vs B teams,
    an impossible run on the pitch, an image-space miss above img_gate box heights, or clearly different
    appearance. Cost = constant-velocity image miss in box heights, averaged over both directions, in the camera that
    sees the person largest + w_dh x log height change + w_app x ReID distance + w_gap per missing frame."""
    if {a.team, b.team} == {"A", "B"}:
        return np.inf
    g = b.k - a.k
    dt = g / rate
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


def _proposal_links(chains: list[Track], proposals: list[Track], cams: dict[str, CamBoxes],
                    max_gap: int) -> set[tuple[int, int]]:
    """Pairs (a, b) where a proposal track goes from the last box of chain a straight to the first box of chain b
    within max_gap frames."""
    chain_of = _owner(chains, cams)
    prop_of = _owner(proposals, cams)
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


def relink(tracks: list[Track], cams: dict[str, CamBoxes], rate: float, max_gap: int = 10, min_len: int = 3,
           vmax: float = 9.0, slack: float = 1.0, w_dh: float = 0.5, w_app: float = 1.0, app_missing: float = 0.15,
           w_gap: float = 0.01, img_gate: float = 1.0, app_gate: float = 0.35, max_cost: float = 0.6,
           margin: float = 0.15, proposals: list[Track] | None = None, prop_margin: float = 0.0,
           prop_min_len: int = 1, max_rounds: int = 5, clean: dict[str, np.ndarray] | None = None,
           team_camera_conflict: bool = True, stats: RelinkStats | None = None) -> list[Track]:
    """Re-attach the continuation of a track that a conservative builder cut at an occlusion, when it is unambiguous.
    `max_gap`, `min_len` and `prop_min_len` count grid samples (tuned at 10 Hz).

    Candidates: a head 1..max_gap frames after a tail, passing the gates of _link_cost. Two rules make links:
      margin rule    among tracks with >= min_len samples (shorter ones are neither linked nor competitors: they are
                     mostly the overlapped boxes of the occlusion itself), a -> b is the cheapest option of both a and
                     b by at least `margin` and costs <= max_cost;
      proposal rule  (with `proposals`, e.g. BoostTrack's output on the same boxes) the proposal tracker goes from
                     a's last box straight to b's first box, a -> b costs <= max_cost and is the mutual best over all
                     tracks with >= prop_min_len samples (short ones included) by at least prop_margin. Two
                     different association rules rarely make the same mistake, so their agreement can bridge
                     occlusions the margin rule cannot; prop_min_len=2 keeps it from chaining through 1-sample
                     overlapped boxes.
    Linked chains get fresh ends (motion and ReID of the joined track) and the matching repeats until nothing
    changes. Linked tracks never overlap in time, so every box stays in exactly one track. Clean crops default to
    identity.clean_masks without any border margin."""
    st = stats if stats is not None else RelinkStats()
    clean = clean_masks(cams, border=0.0) if clean is None else clean
    chains = [dict(t) for t in tracks if t]
    tails: dict[int, _End] = {}
    heads: dict[int, _End] = {}
    for _ in range(max_rounds):
        st.rounds += 1
        longq = {q for q, t in enumerate(chains) if len(t) >= min_len}
        idx = [q for q, t in enumerate(chains) if len(t) >= prop_min_len] if proposals is not None else sorted(longq)
        for q in idx:
            if q not in tails:
                tails[q] = _end(chains[q], cams, clean, True, team_camera_conflict)
                heads[q] = _end(chains[q], cams, clean, False, team_camera_conflict)
        by_start: dict[int, list[int]] = {}
        for q in idx:
            by_start.setdefault(heads[q].k, []).append(q)
        cost: dict[tuple[int, int], float] = {}
        for a in idx:
            for g in range(1, max_gap + 1):
                for b in by_start.get(tails[a].k + g, []):
                    c = _link_cost(tails[a], heads[b], rate, vmax, slack, w_dh, w_app, app_missing, w_gap,
                                   img_gate, app_gate)
                    if np.isfinite(c):
                        cost[(a, b)] = c
        st.candidates += len(cost)

        def mutual_best(pairs: dict[tuple[int, int], float], marg: float,
                        only: set | None = None) -> list[tuple[int, int]]:
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
            links = mutual_best(cost, prop_margin, _proposal_links(chains, proposals, cams, max_gap))
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


# ---------------------------------------------------------------------------------------------------------------
# Short pieces into their host
# ---------------------------------------------------------------------------------------------------------------

def absorb_short(tracks: list[Track], cams: dict[str, CamBoxes], max_len: int = 2, host_min_len: int = 10,
                 gate_m: float = 1.0, margin_m: float = 0.7, max_hole: int = 3,
                 team_camera_conflict: bool = True, moved: list | None = None) -> list[Track]:
    """Give the boxes of very short tracklets (<= max_len samples, mostly the overlapped boxes of an occlusion) to a
    long track (>= host_min_len samples) when that is unambiguous at every one of their samples: the host either has
    a point at that frame without a box of this camera (the box becomes its second view) or a hole of <= max_hole
    frames there (the box fills it, compared with the linear interpolation), the box lies within gate_m of the host,
    every other long track is at least margin_m farther, and the teams do not contradict. A short tracklet moves
    whole or not at all, so a box is never split from its tracklet's other samples by guesswork. `moved` (a list)
    receives the indices of the absorbed tracklets."""
    hosts = [q for q, t in enumerate(tracks) if len(t) >= host_min_len]
    team = {q: team_vote(tracks[q], cams, camera_conflict=team_camera_conflict)[0] for q in hosts}
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

    done = set()
    for s, t in enumerate(tracks):
        if len(t) > max_len or s in done:
            continue
        plan = []
        for k, p in sorted(t.items()):
            if len(p.boxes) != 1:
                plan = None
                break
            cam, i = p.boxes[0]
            tb = cams[cam].team[i]
            cands = []
            for q in at.get(k, []):
                xy, has = host_xy(q, k)
                if xy is None:
                    continue
                if k in out[q] and any(c == cam for c, _ in out[q][k].boxes):
                    d_ok = False                         # the host already has a box of this camera here
                else:
                    d_ok = not (tb and team[q] in ("A", "B") and tb != team[q])
                cands.append((float(np.linalg.norm(xy - p.xy)), q, d_ok))
            cands.sort()
            if not cands or not cands[0][2] or cands[0][0] > gate_m or (
                    len(cands) > 1 and cands[1][0] - cands[0][0] < margin_m):
                plan = None
                break
            plan.append((k, cands[0][1], (cam, i)))
        if not plan or len({q for _, q, _ in plan}) != 1:
            continue
        q = plan[0][1]
        for k, _, cb in plan:
            boxes = sorted((out[q][k].boxes if k in out[q] else []) + [cb])
            out[q][k] = TrackPoint(fuse(cams, boxes)[0], boxes)
        out[s] = {}
        done.add(s)
    if moved is not None:
        moved.extend(sorted(done))
    return sorted((dict(sorted(t.items())) for t in out if t), key=lambda t: min(t))


# ---------------------------------------------------------------------------------------------------------------
# The whole source
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class SourceStats:
    proposals: str | None = None                       # proposal tracker of the per-camera re-link, None = margin only
    conservative: dict = field(default_factory=dict)   # cam -> conservative tracklets
    percam: dict = field(default_factory=dict)         # cam -> tracklets after the per-camera re-link
    relink_percam: dict = field(default_factory=dict)  # cam -> RelinkStats
    pairing: XviewStats = field(default_factory=XviewStats)
    paired: int = 0                                    # tracklets after pair_views
    relink_fused: RelinkStats = field(default_factory=RelinkStats)
    absorbed: int = 0                                  # 1-2 sample tracklets given to a host
    tracklets: int = 0                                 # output


def relink_source(cams: dict[str, CamBoxes], rate: float, frames: Iterable[int] | None = None,
                  proposals: str | None = "auto", team_override_sim: float | None = 0.75,
                  team_camera_conflict: bool = True, stats: SourceStats | None = None) -> list[Track]:
    """The fused tracklet source of the open-set linker (module docstring): per camera conservative_tracklets ->
    relink (margin rule; with proposals='boosttrack' also the BoostTrack agreement rule, never through 1-sample
    pieces), then pair_views (cut_on_partner_change=False) -> relink (min_len 5) -> absorb_short.

    `frames` defaults to tracklets.default_frames(cams). `proposals`: 'boosttrack' (needs boxmot and ReID), None
    (margin rule only) or 'auto' (BoostTrack when boxmot imports and every camera has ReID). `team_override_sim` goes
    to pair_views and `team_camera_conflict` to every team vote (identity.team_vote); None / False are the research
    behaviour. With one camera the per-camera tracklets go straight to the fused re-link."""
    st = stats if stats is not None else SourceStats()
    if len(cams) > 2:
        raise NotImplementedError("pair_views pairs two cameras")
    if proposals == "auto":
        proposals = "boosttrack" if all(c.reid is not None for c in cams.values()) and have_boxmot() else None
    if proposals not in (None, "boosttrack"):
        raise ValueError(f"proposals must be 'auto', 'boosttrack' or None, got {proposals!r}")
    st.proposals = proposals
    frames = list(default_frames(cams) if frames is None else frames)
    per = {}
    for cam in cams:
        base = conservative_tracklets(cams, cam, rate, frames)
        prop = boosttrack_tracklets(cams, cam, rate, frames) if proposals else None
        rs = RelinkStats()
        per[cam] = relink(base, cams, rate, proposals=prop, prop_min_len=2, team_camera_conflict=team_camera_conflict,
                          stats=rs)
        st.conservative[cam], st.percam[cam], st.relink_percam[cam] = len(base), len(per[cam]), rs
    if len(per) == 2:
        c1, c2 = list(cams)
        t = pair_views(per[c1], per[c2], cams, cut_on_partner_change=False, team_override_sim=team_override_sim,
                       stats=st.pairing)
    else:
        t = [tr for ts in per.values() for tr in ts]
    st.paired = len(t)
    t = relink(t, cams, rate, min_len=5, team_camera_conflict=team_camera_conflict, stats=st.relink_fused)
    moved: list = []
    out = absorb_short(t, cams, team_camera_conflict=team_camera_conflict, moved=moved)
    st.absorbed, st.tracklets = len(moved), len(out)
    return out
