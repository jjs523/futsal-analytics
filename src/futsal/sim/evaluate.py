"""ID tracking scores against synthetic ground truth (same idea as the MOT benchmarks' IDF1)."""
from __future__ import annotations

import numpy as np

from ..tracks import TrackSet


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular matrix (Kuhn-Munkres, O(n^3)); avoids a scipy dependency."""
    n_r, n_c = cost.shape
    n = max(n_r, n_c)
    C = np.zeros((n, n)); C[:n_r, :n_c] = cost
    u, v, p, way = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j] and p[j] - 1 < n_r and j - 1 < n_c]


def id_scores(truth: TrackSet, est: TrackSet, radius: float = 1.0) -> dict:
    """IDF1: share of all (true + estimated) positions that are matched under ONE global player<->track
    mapping. ID switches: how often the track closest to a true player changes while it stays matched."""
    T = np.stack([p.xy for p in truth.players])            # (G, F, 2)
    E = np.stack([p.xy for p in est.players]) if est.players else np.zeros((0, T.shape[1], 2))
    G, F = T.shape[:2]
    team_ok = np.array([[tp.team == ep.team or ep.team == "?" for ep in est.players] for tp in truth.players], bool).reshape(G, len(est.players))
    overlap = np.zeros((G, len(est.players)))
    switches, last = 0, [None] * G
    for f in range(F):
        d = np.linalg.norm(T[:, f, None, :] - E[None, :, f, :], axis=2) if len(est.players) else np.zeros((G, 0))
        d = np.where(np.isfinite(d) & team_ok, d, np.inf)
        within = d <= radius
        overlap += within
        for g in range(G):
            if np.isfinite(T[g, f, 0]) and within[g].any():
                k = int(np.argmin(d[g]))
                if last[g] is not None and last[g] != k:
                    switches += 1
                last[g] = k
    n_true = int(np.isfinite(T[..., 0]).sum())
    n_est = int(np.isfinite(E[..., 0]).sum())
    idtp = sum(overlap[g, e] for g, e in _hungarian(-overlap)) if len(est.players) else 0
    return {"idf1": float(2 * idtp / max(n_true + n_est, 1)), "id_switches": int(switches),
            "tracks": len(est.players), "players": G, "minutes": F / truth.fps / 60}
