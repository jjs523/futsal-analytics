"""Grid analysis of a layout: who sees each spot of the pitch and how many metres a 1 px error costs there."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..court import Court
from ..homography import metres_per_pixel
from .layout import TRIPOD_HEIGHT, Layout

BOTH, ONLY_FIRST, ONLY_SECOND, BLIND = 0, 1, 2, 3


@dataclass
class Coverage:
    court: Court
    cell: float
    xs: np.ndarray                 # cell centres along x
    ys: np.ndarray                 # cell centres along y
    zone: np.ndarray               # (ny, nx) BOTH / ONLY_FIRST / ONLY_SECOND / BLIND
    err: np.ndarray                # (ny, nx) metres per px using the better camera; NaN if blind

    @property
    def blind_m2(self) -> float:
        return float((self.zone == BLIND).sum() * self.cell ** 2)

    def share(self, z: int) -> float:
        return float((self.zone == z).mean() * 100)

    def summary(self) -> dict:
        e = self.err[np.isfinite(self.err)]
        return {"both_pct": self.share(BOTH), "single_pct": self.share(ONLY_FIRST) + self.share(ONLY_SECOND),
                "blind_m2": self.blind_m2, "err_median": float(np.median(e)) if e.size else np.nan,
                "err_p95": float(np.percentile(e, 95)) if e.size else np.nan, "err_max": float(e.max()) if e.size else np.nan,
                "weak_pct": float((self.err > 0.4).mean() * 100)}


def analyse(court: Court, layout: Layout, hfov: float, cell: float = 0.25, height: float = TRIPOD_HEIGHT) -> Coverage:
    xs = np.arange(cell / 2, court.length, cell)
    ys = np.arange(cell / 2, court.width, cell)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    cams = list(layout.build(hfov, height).values())
    seen, errs = [], []
    for cam in cams:
        s = cam.sees_player(grid)
        uv, _ = cam.project(grid)
        e = np.full(len(grid), np.inf)
        e[s] = metres_per_pixel(cam.floor_homography(), uv[s])
        seen.append(s); errs.append(e)
    a, b = seen[0], seen[1] if len(seen) > 1 else np.zeros_like(seen[0])
    zone = np.full(len(grid), BLIND)
    zone[a & b], zone[a & ~b], zone[~a & b] = BOTH, ONLY_FIRST, ONLY_SECOND
    best = np.min(np.stack(errs), axis=0)
    best[~np.isfinite(best)] = np.nan
    return Coverage(court, cell, xs, ys, zone.reshape(gx.shape), best.reshape(gx.shape))
