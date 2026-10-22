"""Cross-camera self-calibration from the players themselves.

Two homographies tapped by hand disagree by a metre or more over parts of the pitch (tap error, lens distortion,
feet hidden behind other players). Confident boxes of the two cameras that are mutual nearest neighbours on the
pitch and look alike (ReID) are the same person, so their foot points should coincide. A smooth cubic correction
field fitted on those pairs, each way, removes most of the disagreement; both cameras are moved half way, to the
midpoint, because neither calibration is known to be the better one.
"""
from __future__ import annotations

import numpy as np

from ..court import Court
from .boxes import COURT_MARGIN_M, CamBoxes

MIN_PAIRS = 200
VALID_MARGIN_M = 3.0          # the correction field is trusted up to this far outside the lines (pairs are in-court)


def _features(xy: np.ndarray, court: Court) -> np.ndarray:
    """Cubic polynomial terms of the pitch position, normalised by the pitch size."""
    x, y = xy[:, 0] / court.length, xy[:, 1] / court.width
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], 1)


def _fit(src: np.ndarray, dst: np.ndarray, court: Court) -> np.ndarray:
    """Least squares correction field src -> dst, refitted 3 times without the worst 15 % (wrong pairs: two
    players standing close together who also look alike)."""
    F = _features(src, court)
    W = np.linalg.lstsq(F, dst - src, rcond=None)[0]
    for _ in range(3):
        r = np.linalg.norm(src + F @ W - dst, axis=1)
        keep = r < np.percentile(r, 85)
        W = np.linalg.lstsq(F[keep], (dst - src)[keep], rcond=None)[0]
    return W


def _assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:                       # scipy is optional; same optimum, slower
        from ..sim.evaluate import _hungarian
        pairs = sorted(_hungarian(cost))
        return np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int)
    return linear_sum_assignment(cost)


def matched_pairs(a: CamBoxes, b: CamBoxes, court: Court, min_conf: float = 0.5, max_d: float = 2.5,
                  min_sim: float = 0.7, k_range: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw pitch positions (P in b, Q in a) of boxes that are the same person in both cameras: per grid frame a
    Hungarian match on distance - 0.5 * ReID similarity, kept when close, alike and mutual nearest on the pitch."""
    ina, inb = court.contains(a.xy_raw, COURT_MARGIN_M), court.contains(b.xy_raw, COURT_MARGIN_M)
    ks = sorted(set(a.by_k) & set(b.by_k))
    if k_range is not None:
        ks = [k for k in ks if k_range[0] <= k < k_range[1]]
    P, Q = [], []
    for k in ks:
        ia, ib = a.by_k[k], b.by_k[k]
        ia = ia[(a.conf[ia] >= min_conf) & ina[ia]]
        ib = ib[(b.conf[ib] >= min_conf) & inb[ib]]
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy_raw[ia][:, None] - b.xy_raw[ib][None], axis=2)
        S = a.reid[ia] @ b.reid[ib].T
        r, c = _assignment(D - 0.5 * S)
        for i, j in zip(r, c):
            if D[i, j] < max_d and S[i, j] > min_sim and D[i].argmin() == j and D[:, j].argmin() == i:
                P.append(b.xy_raw[ib[j]])
                Q.append(a.xy_raw[ia[i]])
    return np.array(P, float).reshape(-1, 2), np.array(Q, float).reshape(-1, 2)


def apply_alignment(cams: dict[str, CamBoxes], coef: dict[str, list], court: Court = Court()) -> None:
    """Move each camera in `coef` by half its correction field (in place), starting from xy_raw, so applying
    twice does not compound. in_court follows the moved positions.

    The cubic is fitted on in-court pairs only and blows up outside them (an off-pitch box went to 3e8 m), so it is
    evaluated exactly within the pitch + VALID_MARGIN_M and, further out, at the nearest point of that zone: the
    shift stays bounded and continuous. Cutting it to zero there instead would jump by up to ~3 m at the zone edge
    (the test match's far corner) and push real players there out of in_court."""
    for name, W in coef.items():
        cam = cams.get(name)
        if cam is None or not len(cam):
            continue
        m = VALID_MARGIN_M
        at = np.stack([np.clip(cam.xy_raw[:, 0], -m, court.length + m), np.clip(cam.xy_raw[:, 1], -m, court.width + m)], 1)
        cam.xy = cam.xy_raw + 0.5 * (_features(at, court) @ np.asarray(W, float))
        cam.in_court = court.contains(cam.xy, COURT_MARGIN_M)


def align_cameras(cams: dict[str, CamBoxes], court: Court, min_conf: float = 0.5, max_d: float = 2.5,
                  min_sim: float = 0.7, k_range: tuple[int, int] | None = None) -> dict | None:
    """Fit the cross-camera correction and move both cameras' xy to the midpoint (in place; xy_raw is kept).

    Needs exactly two cameras with ReID and at least MIN_PAIRS matched pairs; otherwise the cameras are left as
    they are and None is returned (a poor fit from a handful of pairs would bend the pitch). `k_range` limits the
    pairs to grid frames lo <= k < hi."""
    if len(cams) != 2:
        return None
    (na, a), (nb, b) = cams.items()
    if a.reid is None or b.reid is None:
        return None
    P, Q = matched_pairs(a, b, court, min_conf, max_d, min_sim, k_range)
    if len(P) < MIN_PAIRS:
        return None
    Wba, Wab = _fit(P, Q, court), _fit(Q, P, court)        # b -> a's frame, a -> b's frame
    before = float(np.median(np.linalg.norm(P - Q, axis=1)))
    coef = {na: Wab.tolist(), nb: Wba.tolist()}
    apply_alignment(cams, coef, court)
    Pm = P + 0.5 * (_features(P, court) @ Wba)
    Qm = Q + 0.5 * (_features(Q, court) @ Wab)
    after = float(np.median(np.linalg.norm(Pm - Qm, axis=1)))
    return {"pairs": len(P), "median_before_m": round(before, 3), "median_after_m": round(after, 3), "coef": coef}


# ---------------------------------------------------------------------------------------------------------------
# Clock refinement from the players
# ---------------------------------------------------------------------------------------------------------------

def _shift_residual(a: CamBoxes, b: CamBoxes, ks: list[int], d: int, min_conf: float, max_d: float,
                    min_sim: float) -> tuple[float, int]:
    """Mean distance between the same players (mutual nearest + ReID-alike) when camera b is read d grid frames
    later than camera a."""
    ds = []
    for k in ks:
        ia, ib = a.boxes_at(k, min_conf), b.boxes_at(k + d, min_conf)
        if not len(ia) or not len(ib):
            continue
        D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
        S = a.reid[ia] @ b.reid[ib].T
        r, c = _assignment(D - 0.5 * S)
        for i, j in zip(r, c):
            if D[i, j] < max_d and S[i, j] > min_sim and D[i].argmin() == j and D[:, j].argmin() == i:
                ds.append(float(D[i, j]))
    return (float(np.mean(ds)) if len(ds) >= 100 else float("nan")), len(ds)


def estimate_clock(a: CamBoxes, b: CamBoxes, rate: float, window_s: float = 300.0, max_shift_s: float = 1.5,
                   step: int = 3, min_conf: float = 0.5, max_d: float = 2.5, min_sim: float = 0.7) -> dict | None:
    """How far camera b's clock is off camera a's, from the players: per window, b is read d grid frames earlier or
    later and the d at which the same players coincide best on the pitch is kept (sub-frame by a parabola); a line
    through those shifts gives an extra offset and drift for b.

    Why: the audio sync is good to ~0.1 s at one moment but cannot see a slow drift through the sound-travel noise,
    and comparing players a second apart looks like 1 m of position error for a runner. On the 10/8 test this
    measured -389 ppm (1.2 s over 50 minutes); that turned out to be grid_frames' old push-forward rule on the 60 fps
    phone (boxes.one_source_frame_per_k), not the phones.
    Returns {'shift_s': [(t_mid_s, best_shift_s), ...], 'offset_s', 'drift', 'span_s'} where b's corrected grid is
    k_b(t) = k_b - (offset_s + drift * t_grid) * rate, or None when fewer than 2 windows have enough pairs."""
    if a.reid is None or b.reid is None or not a.by_k or not b.by_k:
        return None
    n = 1 + max(max(a.by_k), max(b.by_k))
    W = max(1, int(round(window_s * rate)))
    D = int(round(max_shift_s * rate))
    rows = []
    for lo in range(0, n, W):
        ks = [k for k in range(lo, min(lo + W, n), step) if k in a.by_k]
        res = {d: _shift_residual(a, b, ks, d, min_conf, max_d, min_sim)[0] for d in range(-D, D + 1)}
        ok = {d: r for d, r in res.items() if np.isfinite(r)}
        if not ok:
            continue
        d0 = min(ok, key=ok.get)
        frac = 0.0
        if d0 - 1 in ok and d0 + 1 in ok:
            y0, y1, y2 = ok[d0 - 1], ok[d0], ok[d0 + 1]
            den = y0 - 2 * y1 + y2
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den > 0 else 0.0
        rows.append(((lo + min(lo + W, n)) / 2 / rate, (d0 + frac) / rate))
    if len(rows) < 2:
        return None
    t, s = np.array(rows).T
    drift, off = np.polyfit(t, s, 1)
    return {"shift_s": [(round(float(x), 1), round(float(y), 3)) for x, y in rows], "offset_s": float(off),
            "drift": float(drift), "span_s": float(t[-1] - t[0])}


def apply_clock(cam: CamBoxes, clock: dict, rate: float) -> None:
    """Move camera `cam` onto the corrected grid of estimate_clock (in place): k_new = k - shift(t) * rate with
    shift(t) = offset_s + drift * t, t the grid time. At most one source frame per grid frame, as in grid_frames:
    with drift > 0 the corrected grid is compressed and the surplus frames are left off it (k = -1)."""
    from .boxes import _index_by_k, one_source_frame_per_k
    on = cam.k >= 0
    k0 = {int(f): int(k) for f, k in zip(cam.fid[on], cam.k[on])}
    keys = list(k0)
    k_of = one_source_frame_per_k(keys, [k0[f] - (clock["offset_s"] + clock["drift"] * k0[f] / rate) * rate for f in keys])
    cam.k = np.array([k_of[int(f)] if o else -1 for f, o in zip(cam.fid, on)], int)
    cam.by_k = _index_by_k(cam.k)


MIN_CLOCK_SPAN_S = 600.0      # a line through window shifts closer together than this is no better than the audio sync


def refine_clock(cams: dict[str, CamBoxes], rate: float, min_span_s: float = MIN_CLOCK_SPAN_S, **kw) -> dict | None:
    """estimate_clock of the second camera against the first and, when the window shifts span at least min_span_s,
    apply_clock to the second camera (in place). Each window's best shift is good to a few hundredths of a second,
    so a drift fitted over two or three windows a few minutes apart is mostly noise; shorter clips keep the audio
    sync. Returns the estimate plus "camera" (the corrected one) and "applied", or None (not two cameras with ReID,
    fewer than 2 windows)."""
    if len(cams) != 2:
        return None
    (_, a), (nb, b) = cams.items()
    c = estimate_clock(a, b, rate, **kw)
    if c is None:
        return None
    c["camera"] = nb
    c["applied"] = bool(c["span_s"] >= min_span_s)
    if c["applied"]:
        apply_clock(b, c, rate)
    return c


def corrected_sync(offset: float, drift: float, clock: dict, start: float = 0.0) -> tuple[float, float]:
    """The (offset, drift) that reads a camera corrected by apply_clock: its frame for reference time T is at its own
    t = (T - offset) / (1 + drift) (trackview.render's convention). `offset`, `drift` are its sync before the
    correction and `start` the reference time of grid frame 0, since the correction is a line in grid time."""
    c, d = clock["offset_s"], clock["drift"]
    return offset - c + d * (start - offset), (1 - d) * (1 + drift) - 1
