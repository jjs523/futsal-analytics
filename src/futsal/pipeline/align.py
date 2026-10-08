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
    twice does not compound. in_court follows the moved positions."""
    for name, W in coef.items():
        cam = cams.get(name)
        if cam is None or not len(cam):
            continue
        cam.xy = cam.xy_raw + 0.5 * (_features(cam.xy_raw, court) @ np.asarray(W, float))
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
