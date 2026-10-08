"""Final candidates: the best tracklet source (sources.py) x the best linker (link_closed.py / link_gta.py) x the
appearance embedding (raw OSNet / the per-match SSL head of ssl_embed.py).

    python experiments/harness.py final_a combine
    python experiments/metrics2.py baseline final_a final_b final_c final_d --parts first,second
    python experiments/combine.py baseline final_a final_b final_c final_d     # witnessed swaps per half

Candidates (parameters lightly tuned on part='first' only, see FINAL_CLOSED / FINAL_GTA):
  final_a  sources.best_source('relink') -> link_closed (closed set, 5 + 1 slots per team), raw OSNet
  final_b  sources.best_source('relink') -> link_gta (open-set agglomerative linker), raw OSNet
  final_c  final_a with the SSL head in the linker (the tracklet source stays on raw OSNet). The registered variant
           trains the head on the first half only, so its second-half numbers are held out; in production train on
           the whole window (ssl_head_window='all').
  final_d  final_a's linker on xview('conservative') instead of the relink source: the source control

Result (metrics2, first / second half): final_d and final_a are the two best, about equal on identity (10 long
ids + 2 short spare ids, events <= 0.7/min, team impurity <= 0.011, count_5v5 0.82-0.87); final_d has the higher
frames_in_top10 (0.968 / 0.95) and fewer speed violations, final_a the higher box coverage (0.95 vs 0.925). The SSL
head (final_c) helps only on the half it was trained on. The open-set final_b keeps 98% of the boxes but shows
6th 'N' people (count_over5 0.14 / 0.10).

Positions are the linkers' own (B2 re-fusion, Hermite gap fill, Savitzky-Golay), so the speed metrics of all
candidates stay comparable with the earlier variants. `refine` (registered as final_a_rts) is a covariance-aware
RTS smoother for the analytics output: it removes the far-side foot-point jitter, but it also removes what the
speed_jumps / teleports metrics measure, so never score link quality on its output.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

from harness import Data, Track, TrackPoint, register
from blocks import fuse

# ---------------------------------------------------------------------------------------------------------------
# Tuned linker settings (part='first')
# ---------------------------------------------------------------------------------------------------------------

def _closed_params():
    from link_closed import ClosedParams
    # with the cam2 top-edge exemption (closed_top_exempt, count_5v5 0.66 -> 0.72 on the relink source):
    # coverage 0.5 -> 0.7: a tracklet without clean crops is worth assigning (0.6-0.8 gave identical results;
    # count_5v5 -> 0.80, the spare slots start to take the 6th-person tracklets);
    # attach_gap 1 -> 2 s: short pieces next to a slot's points join it (count_5v5 +0.07, speed jumps +1.7/min).
    # d_none 0.3, min_len 0.5 / 2 s, vmax 9 / slack 1.5 and spare=0 were neutral or worse
    return ClosedParams(coverage=0.7, attach_gap_s=2.0)


def _gta_params():
    from link_gta import GtaParams
    # min_len 1 -> 5 s: unattached debris and short off-court people stop counting as identities (ids 21 -> 16,
    # events 2.7 -> 0.7/min, count_over5 0.19 -> 0.14); 10 s gave the same result, 3 s one more id.
    # merge_thr 0.3-0.4, medoid linkage and DBSCAN 0.25 changed nothing or made it worse on this source
    return GtaParams(min_len_s=5.0)


@dataclass(frozen=True)
class Candidate:
    source: str = "relink"            # sources.best_source name, or 'xv_conservative'
    linker: str = "closed"            # 'closed' (link_closed) | 'gta' (link_gta)
    embedding: str = "raw"            # 'raw' OSNet | 'ssl' head | 'mix' (head and OSNet 1:1)
    ssl_head_window: str = "first"    # head training frames: 'first' half (held-out evaluation) or 'all'
    cam2_top_exempt: bool = True      # link_closed: cam2 crops cut at the top edge still count as clean


# ---------------------------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------------------------

def source_tracklets(data: Data, name: str) -> list[Track]:
    """Tracklet source on raw OSNet (call it outside any embedding swap: the builders' gates are OSNet-tuned)."""
    if name == "xv_conservative":
        from xview import xview
        return xview(data, "conservative")
    from sources import best_source
    return best_source(data, name)


@contextlib.contextmanager
def closed_top_exempt(enabled: bool = True):
    """Make link_closed use link_gta's clean-crop rule without cam2's top-border test. cam2 is aimed low and cuts
    most heads at the top edge, so blocks.clean_masks leaves it almost no clean crops (phase-1 insight, link_gta
    insight 4); link_closed has no parameter for it, so its module-level name is swapped for the duration."""
    if not enabled:
        yield
        return
    import link_closed as lc
    from link_gta import clean_crops
    orig = lc.clean_masks
    lc.clean_masks = lambda data, *a, **kw: clean_crops(data, ("cam2",))
    try:
        yield
    finally:
        lc.clean_masks = orig


@contextlib.contextmanager
def embedding(data: Data, kind: str, window: str = "first"):
    """Swap CamData.reid for the SSL head ('ssl') or head + OSNet ('mix') while linking; 'raw' does nothing."""
    if kind == "raw":
        yield
        return
    from ssl_embed import load_or_train, mix_feats, raw_feats, with_embedding
    head = load_or_train(data, **({"train_lo": 0, "train_hi": data.n} if window == "all" else {}))
    feats = mix_feats(head, raw_feats(data)) if kind == "mix" else head
    with with_embedding(data, feats):
        yield


def run_candidate(data: Data, c: Candidate, closed_params=None, gta_params=None) -> list[Track]:
    """Source (raw OSNet) -> linker under the chosen embedding. Linker stats go to stderr."""
    tracklets = source_tracklets(data, c.source)
    with embedding(data, c.embedding, c.ssl_head_window):
        if c.linker == "closed":
            from link_closed import ClosedStats, link_closed
            st = ClosedStats()
            with closed_top_exempt(c.cam2_top_exempt):
                out = link_closed(tracklets, data, closed_params or _closed_params(), st)
            print(f"[closed] tracklets {st.tracklets} solved {st.solved} assigned {st.assigned} unassigned_long "
                  f"{st.unassigned_long} ({st.unassigned_long_samples} samples) short attached {st.attached_short}/"
                  f"{st.short_total} share {st.assigned_samples_share} exact5v5 {st.exact_5v5_slots}/"
                  f"{st.exact_5v5_filled} spare {st.spare_samples} reach_viol {st.reach_violations} "
                  f"merged_dups {st.merged_dups} solver {st.solver_s} unpinned {st.unpinned}", file=sys.stderr)
            return out
        if c.linker == "gta":
            from link_gta import GtaStats, link
            st = GtaStats()
            out = link(tracklets, data, gta_params or _gta_params(), st)
            print(f"[gta] inputs {st.inputs} after_split {st.after_split} embedded {st.embedded} merges app "
                  f"{st.merges_app} motion {st.merges_motion} debris {st.debris_attached} ambiguous {st.ambiguous} "
                  f"outputs {st.outputs} dropped_points {st.dropped_points}", file=sys.stderr)
            return out
    raise ValueError(f"unknown linker {c.linker!r}")


# ---------------------------------------------------------------------------------------------------------------
# Optional position refinement for analytics: covariance-weighted RTS smoothing of the final identities
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RefineParams:
    sigma_a: float = 4.0          # m/s^2: white-noise acceleration of the constant-velocity model
    gate_d2: float = 9.21         # innovations beyond this Mahalanobis^2 get their covariance inflated (soft gate)
    max_gap_s: float = 3.0        # a longer gap without points restarts the filter
    sigma_v0: float = 3.0         # m/s: initial velocity uncertainty


def rts_smooth(track: Track, data: Data, p: RefineParams = RefineParams()) -> Track:
    """Constant-velocity Kalman filter + RTS smoother over every frame of each segment of `track`. Points with boxes
    are measurements (B2-fused position and covariance, so a far-side foot point only counts along its precise
    direction); points without boxes (gap fills) are predicted only. Same frames and boxes, new positions."""
    if not track:
        return {}
    dt = 1.0 / data.rate
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = p.sigma_a ** 2 * G @ G.T
    H = np.eye(2, 4)
    out: Track = {}
    ks_all = np.array(sorted(track), int)
    for seg in np.split(ks_all, np.flatnonzero(np.diff(ks_all) > p.max_gap_s * data.rate) + 1):
        k0, n = int(seg[0]), int(seg[-1] - seg[0]) + 1
        z = np.full((n, 2), np.nan)
        R = np.zeros((n, 2, 2))
        for k in seg.tolist():
            if track[k].boxes:
                z[k - k0], R[k - k0] = fuse(data, track[k].boxes)
        meas = np.flatnonzero(np.isfinite(z[:, 0]))
        if not len(meas):                                   # nothing to filter: keep the points as they are
            out.update({int(k): track[int(k)] for k in seg})
            continue
        f0 = int(meas[0])
        x = np.r_[z[f0], 0.0, 0.0]
        P = np.diag([*np.diag(R[f0]) + (25.0 if f0 else 0.0), p.sigma_v0 ** 2, p.sigma_v0 ** 2])
        xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))
        xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
        for i in range(n):
            if i:
                x, P = F @ x, F @ P @ F.T + Q
            xp[i], Pp[i] = x, P
            if np.isfinite(z[i, 0]):
                y = z[i] - x[:2]
                d2 = float(y @ np.linalg.solve(P[:2, :2] + R[i], y))
                S = P[:2, :2] + R[i] * max(1.0, d2 / p.gate_d2)
                K = np.linalg.solve(S, P[:2, :]).T
                x, P = x + K @ y, P - K @ H @ P
            xf[i], Pf[i] = x, P
        xs = xf.copy()
        for i in range(n - 2, -1, -1):
            C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
            xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        for k in seg.tolist():
            out[k] = TrackPoint(xs[k - k0, :2].copy(), list(track[k].boxes))
    return dict(sorted(out.items()))


def refine(tracks: list[Track], data: Data, p: RefineParams | None = None) -> list[Track]:
    return [rts_smooth(t, data, p or RefineParams()) for t in tracks if t]


# ---------------------------------------------------------------------------------------------------------------
# Extra label-free check: identity swaps witnessed by the other camera (sources.xref_swaps), per time half
# ---------------------------------------------------------------------------------------------------------------

def witnessed_swaps(tracks: list[Track], data: Data, part: str = "all", ref: list[Track] | None = None) -> dict:
    """sources.xref_swaps on the tracks clipped to `part`, against the xview conservative tracklets. Tracks built
    from that same source cannot swap inside one of its tracklets, so they score slightly better by construction."""
    import metrics2
    from sources import xref_swaps
    lo, hi = metrics2.part_range(data, part)
    ref = ref if ref is not None else source_tracklets(data, "xv_conservative")
    r = xref_swaps(metrics2.clip(tracks, lo, hi), data, ref)
    return r | {"xref_swaps_per_min": round(r["xref_swaps"] / ((hi - lo) / data.rate / 60.0), 2)}


# ---------------------------------------------------------------------------------------------------------------
# Registered candidates
# ---------------------------------------------------------------------------------------------------------------

FINALS: dict[str, Candidate] = {
    "final_a": Candidate("relink", "closed", "raw"),
    "final_b": Candidate("relink", "gta", "raw"),
    "final_c": Candidate("relink", "closed", "ssl"),
    "final_d": Candidate("xv_conservative", "closed", "raw"),
}

for _name, _c in FINALS.items():
    register(_name)(lambda data, _c=_c: run_candidate(data, _c))


@register("final_a_rts")
def final_a_rts(data: Data) -> list[Track]:
    """final_a with RTS-smoothed positions (analytics output; not a candidate, see the module docstring)."""
    return refine(run_candidate(data, FINALS["final_a"], replace(_closed_params(), refuse=False, smooth=False)), data)


if __name__ == "__main__":
    import harness
    data = Data()
    ref = source_tracklets(data, "xv_conservative")
    for name in sys.argv[1:]:
        tr = harness.load_tracks(os.path.join(harness.ROOT, "experiments", "results", f"{name}.json"))
        print(name, {part: witnessed_swaps(tr, data, part, ref) for part in ("first", "second")})
