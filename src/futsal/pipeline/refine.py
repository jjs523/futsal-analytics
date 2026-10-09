"""Position refinement for analytics: covariance-aware constant-velocity Kalman + RTS smoothing of final identities.

The linker's positions (link_closed: re-fused boxes, Hermite gap fills, Savitzky-Golay) keep the foot-point jitter
of far-side players, which inflates distance and sprint counts. Here every point with boxes is a measurement with
its fused anisotropic covariance (identity.fuse), so a far-side foot point only counts along the direction it is
precise in, and the RTS smoother uses the whole track, past and future. Identities and boxes are never changed.

Use the output for distances, speeds and heat maps only: smoothing removes exactly what the link-quality metrics
(speed jumps, teleports) look for, so never score a linker on refined positions.

Ported from experiments/combine.py (rts_smooth / refine, RefineParams).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import CamBoxes, Track, TrackPoint
from .identity import fuse


@dataclass(frozen=True)
class RefineParams:
    sigma_a: float = 4.0          # m/s^2: white-noise acceleration of the constant-velocity model
    gate_d2: float = 9.21         # innovations beyond this Mahalanobis^2 get their covariance inflated (soft gate)
    max_gap_s: float = 3.0        # a longer gap without points restarts the filter
    sigma_v0: float = 3.0         # m/s: initial velocity uncertainty


def rts_smooth(track: Track, cams: dict[str, CamBoxes], rate: float, p: RefineParams = RefineParams()) -> Track:
    """Constant-velocity Kalman filter + RTS smoother over every frame of each segment of `track` (segments split at
    gaps > max_gap_s). Points with boxes are measurements (fused position and covariance of their boxes); points
    without boxes (gap fills) are predicted only. An innovation beyond gate_d2 inflates its covariance instead of
    being dropped, so an outlier pulls little but a real turn is still followed. Same frames and boxes, new xy."""
    if not track:
        return {}
    dt = 1.0 / rate
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = p.sigma_a ** 2 * G @ G.T
    H = np.eye(2, 4)
    out: Track = {}
    ks_all = np.array(sorted(track), int)
    for seg in np.split(ks_all, np.flatnonzero(np.diff(ks_all) > p.max_gap_s * rate) + 1):
        k0, n = int(seg[0]), int(seg[-1] - seg[0]) + 1
        z = np.full((n, 2), np.nan)
        R = np.zeros((n, 2, 2))
        for k in seg.tolist():
            if track[k].boxes:
                z[k - k0], R[k - k0] = fuse(cams, track[k].boxes)
        meas = np.flatnonzero(np.isfinite(z[:, 0]))
        if not len(meas):                                   # nothing to filter: keep the points as they are
            out.update({int(k): track[int(k)] for k in seg})
            continue
        f0 = int(meas[0])
        x = np.r_[z[f0], 0.0, 0.0]
        # starting at the first measurement but filtering from the segment start: be vague until it is reached
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


def refine_positions(tracks: list[Track], cams: dict[str, CamBoxes], rate: float,
                     params: RefineParams = RefineParams()) -> list[Track]:
    """rts_smooth on every non-empty track (`rate` is the grid rate in Hz). Same tracks in the same order, same
    frames and boxes; only positions change. Best on the linker output without its own smoothing
    (ClosedParams(refuse=False, smooth=False)), which the filter replaces."""
    return [rts_smooth(t, cams, rate, params) for t in tracks if t]
