"""Player metrics from a pitch trajectory sampled at a fixed rate (NaN = not seen).

Raw per-frame positions jitter by tens of centimetres (more on the far side), and summing jittery
steps inflates distance several-fold for slow players such as goalkeepers. So metrics are computed
on a constant-velocity Kalman + RTS smoothed track whose measurement noise is estimated per player.
On synthetic matches this keeps distance within ~0-13 % and top speed within ~0.7 m/s of the truth.
"""
from __future__ import annotations

import numpy as np

SPRINT_SPEED = 5.5          # m/s, a common amateur-futsal sprint threshold (tune with real data)
SPRINT_MIN_S = 1.0          # a sprint must last at least this long
SPEED_SPAN_S = 0.5          # speed = displacement over this window of the smoothed track
MAX_GAP_S = 2.0             # do not bridge longer unseen gaps when measuring distance
PLAYER_ACCEL = 2.0          # m/s^2, process noise of the smoother


def _noise_std(z: np.ndarray) -> float:
    """Measurement noise from the second differences of the seen samples (robust, MAD based)."""
    z = z[np.isfinite(z)]
    if len(z) < 5:
        return 0.3
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    return max(0.03, 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / np.sqrt(6))


def kalman_smooth(xy: np.ndarray, dt: float, accel: float = PLAYER_ACCEL) -> np.ndarray:
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother, per axis. NaN inputs are
    predicted through; samples inside unseen gaps longer than MAX_GAP_S come back as NaN."""
    xy = np.asarray(xy, float)
    n = len(xy)
    out = np.full_like(xy, np.nan)
    q11, q12, q22 = accel ** 2 * dt ** 4 / 4, accel ** 2 * dt ** 3 / 2, accel ** 2 * dt ** 2
    for c in range(xy.shape[1]):
        z = xy[:, c]
        ok = np.isfinite(z)
        if ok.sum() < 2:
            continue
        R = _noise_std(z) ** 2
        xs = np.empty((n, 2)); P = np.empty((n, 3)); xp = np.empty((n, 2)); Pp = np.empty((n, 3))
        x0, x1 = z[np.flatnonzero(ok)[0]], 0.0
        p00, p01, p11 = R, 0.0, 25.0
        for i in range(n):
            if i:
                x0 = x0 + dt * x1
                p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q11, p01 + dt * p11 + q12, p11 + q22
            xp[i] = (x0, x1); Pp[i] = (p00, p01, p11)
            if ok[i]:
                s = p00 + R
                k0, k1 = p00 / s, p01 / s
                r = z[i] - x0
                x0, x1 = x0 + k0 * r, x1 + k1 * r
                p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
            xs[i] = (x0, x1); P[i] = (p00, p01, p11)
        for i in range(n - 2, -1, -1):                       # RTS backward pass, 2x2 algebra by hand
            a00, a01, a11 = P[i]
            b00, b01, b11 = Pp[i + 1]
            # C = P_i F^T inv(Pp_{i+1}),  F = [[1, dt], [0, 1]]
            f00, f01, f10, f11 = a00 + dt * a01, a01, a01 + dt * a11, a11
            det = b00 * b11 - b01 * b01
            i00, i01, i11 = b11 / det, -b01 / det, b00 / det
            c00, c01 = f00 * i00 + f01 * i01, f00 * i01 + f01 * i11
            c10, c11 = f10 * i00 + f11 * i01, f10 * i01 + f11 * i11
            d0, d1 = xs[i + 1, 0] - xp[i + 1, 0], xs[i + 1, 1] - xp[i + 1, 1]
            xs[i] = (xs[i, 0] + c00 * d0 + c01 * d1, xs[i, 1] + c10 * d0 + c11 * d1)
            e00, e01, e11 = P[i + 1, 0] - b00, P[i + 1, 1] - b01, P[i + 1, 2] - b11
            P[i] = (a00 + c00 * (c00 * e00 + c01 * e01) + c01 * (c00 * e01 + c01 * e11),
                    a01 + c00 * (c10 * e00 + c11 * e01) + c01 * (c10 * e01 + c11 * e11),
                    a11 + c10 * (c10 * e00 + c11 * e01) + c11 * (c10 * e01 + c11 * e11))
        out[:, c] = xs[:, 0]
    out[_long_gaps(np.isfinite(xy[:, 0]), int(MAX_GAP_S / dt))] = np.nan
    return out


def _long_gaps(seen: np.ndarray, max_len: int) -> np.ndarray:
    """True on unseen samples that belong to a gap longer than max_len (or before the first / after the last sighting)."""
    bad = np.zeros_like(seen)
    idx = np.flatnonzero(seen)
    if not len(idx):
        return ~bad
    bad[:idx[0]] = True; bad[idx[-1] + 1:] = True
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a - 1 > max_len:
            bad[a + 1:b] = True
    return bad


def speeds(smoothed: np.ndarray, dt: float, span_s: float = SPEED_SPAN_S) -> np.ndarray:
    k = max(1, int(round(span_s / dt)))
    v = np.full(len(smoothed), np.nan)
    if len(smoothed) > k:
        v[k // 2: k // 2 + len(smoothed) - k] = np.linalg.norm(smoothed[k:] - smoothed[:-k], axis=1) / (k * dt)
    return v


def summary(xy: np.ndarray, dt: float) -> dict:
    s = kalman_smooth(xy, dt)
    step = np.linalg.norm(np.diff(s, axis=0), axis=1)
    v = speeds(s, dt)
    ok = np.isfinite(v)
    fast = ok & (v >= SPRINT_SPEED)
    sprints, run, need = 0, 0, int(round(SPRINT_MIN_S / dt))
    for f in fast:
        run = run + 1 if f else 0
        sprints += run == need
    return {
        "distance_m": float(np.nansum(step)),
        "max_speed_ms": float(np.nanmax(v)) if ok.any() else 0.0,
        "mean_speed_ms": float(np.nanmean(v)) if ok.any() else 0.0,
        "sprints": int(sprints),
        "seen_ratio": float(np.isfinite(xy[:, 0]).mean()) if len(xy) else 0.0,
    }


def heatmap(xy: np.ndarray, length: float, width: float, cell: float = 1.0) -> np.ndarray:
    """Occupancy grid (rows = y, cols = x) normalised to sum to 1."""
    xy = xy[np.isfinite(xy).all(1)]
    nx, ny = int(np.ceil(length / cell)), int(np.ceil(width / cell))
    h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[ny, nx], range=[[0, width], [0, length]])
    return h / h.sum() if h.sum() else h
