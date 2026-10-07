"""Camera layouts and the brute-force search for the best two-phone placement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import Camera, yaw_towards
from ..court import Court

# 16:9 video horizontal FOV of the 1x lens, computed from 35 mm-equivalent focal lengths (see docs/simulation.md)
HFOV_NARROW = 67.3      # 26 mm: iPhone 16 / 16e / 17
HFOV_AVERAGE = 71.5     # 24 mm average of current flagships
HFOV_WIDE = 73.9        # 23 mm: Galaxy S26 / Z Flip7
CAMERA_OFFSET = 1.3     # tripod stands this far outside the touch/goal line, inside the fence
TRIPOD_HEIGHT = 2.0


@dataclass
class CameraSpec:
    name: str
    xy: tuple[float, float]
    yaw_deg: float

    def build(self, hfov: float, height: float = TRIPOD_HEIGHT) -> Camera:
        return Camera.on_tripod(self.xy, height, self.yaw_deg, hfov)


@dataclass
class Layout:
    name: str
    cameras: list[CameraSpec]

    def build(self, hfov: float, height: float = TRIPOD_HEIGHT) -> dict[str, Camera]:
        return {c.name: c.build(hfov, height) for c in self.cameras}


def diagonal(court: Court, twist_deg: float = 0.0, offset: float = CAMERA_OFFSET) -> Layout:
    """Phones at the bottom-left and top-right corners. twist 0 = facing each other; a positive twist turns
    both phones counter-clockwise (the other phone then appears right of the frame centre)."""
    a = (-offset, -offset)
    b = (court.length + offset, court.width + offset)
    return Layout(f"diagonal(twist={twist_deg:g})",
                  [CameraSpec("cam1", a, yaw_towards(a, b, twist_deg)), CameraSpec("cam2", b, yaw_towards(b, a, twist_deg))])


def safe_twist(court: Court, hfov: float = HFOV_NARROW, step: float = 0.5, max_twist: float = 10.0) -> float:
    """Largest twist that still leaves no blind area for a phone with `hfov` (the app shows where the other
    phone must appear on screen). Coverage is not monotonic in the twist, so stop at the first blind spot."""
    from .coverage import analyse
    best = 0.0
    for t in np.arange(0.0, max_twist + 1e-9, step):
        if analyse(court, diagonal(court, t), hfov, cell=0.25).blind_m2 > 0:
            break
        best = float(t)
    return best


def other_phone_screen_x(court: Court, twist_deg: float, hfov: float) -> float:
    """Where the opposite phone appears across the frame (0 = left edge, 1 = right edge)."""
    lay = diagonal(court, twist_deg)
    cams = lay.build(hfov)
    uv, _ = cams["cam1"].project(np.array([[*lay.cameras[1].xy, TRIPOD_HEIGHT]]))
    return float(uv[0, 0] / cams["cam1"].width)


def perimeter(court: Court, step: float = 2.0, offset: float = CAMERA_OFFSET, avoid_goals: float = 3.5) -> np.ndarray:
    L, W = court.length, court.width
    x0, x1, y0, y1 = -offset, L + offset, -offset, W + offset
    pts = [(x, y) for x in np.arange(x0, x1 + 1e-6, step) for y in (y0, y1)]
    pts += [(x, y) for y in np.arange(y0 + step, y1 - 1e-6, step) for x in (x0, x1)]
    pts = [p for p in pts if not ((p[0] < 0 or p[0] > L) and abs(p[1] - W / 2) < avoid_goals)]   # not behind a goal
    return np.unique(np.round(np.array(pts), 2), axis=0)


def search(court: Court, hfov: float, yaw_step: float = 4.0, pos_step: float = 2.0, top: int = 5, cell: float = 0.5):
    """Every pair of tripod positions on the perimeter x every inward yaw. Ranked by union coverage,
    then by area seen by both phones. Returns [(CameraSpec, CameraSpec, union %, both %)]."""
    xs, ys = np.meshgrid(np.arange(cell / 2, court.length, cell), np.arange(cell / 2, court.width, cell))
    grid = np.stack([xs.ravel(), ys.ravel()], 1)
    specs, vis = [], []
    for p in perimeter(court, pos_step):
        inward = yaw_towards(p, court.centre)
        for d in np.arange(-80, 81, yaw_step):
            s = CameraSpec("cam", (float(p[0]), float(p[1])), float(inward + d))
            specs.append(s); vis.append(s.build(hfov).sees_player(grid))
    V = np.array(vis, np.float32)
    n = V.sum(1)
    both = V @ V.T
    union = n[:, None] + n[None, :] - both
    score = union * V.shape[1] + both
    pos = np.array([hash(s.xy) for s in specs])
    _, pid = np.unique(pos, return_inverse=True)
    best = {}
    for i in range(len(specs)):
        row = score[i].copy(); row[pid <= pid[i]] = -1          # each position pair once, different positions
        j = int(np.argmax(row))
        if row[j] < 0:
            continue
        key = (pid[i], pid[j])
        if key not in best or row[j] > best[key][0]:
            best[key] = (row[j], i, j)
    ranked = sorted(best.values(), key=lambda t: -t[0])[:top]
    N = V.shape[1]
    return [(specs[i], specs[j], float(union[i, j] / N * 100), float(both[i, j] / N * 100)) for _, i, j in ranked]
