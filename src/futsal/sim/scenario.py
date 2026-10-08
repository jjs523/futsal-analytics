"""Synthetic futsal match: 2 x (1 goalkeeper + 4 outfield players) moving plausibly around the pitch."""
from __future__ import annotations

import numpy as np

from ..court import Court
from ..tracks import PlayerTrack, TrackSet

KOREAN_NAMES = ["김민준", "이서준", "박도윤", "최예준", "정시우", "강하준", "조주원", "윤지호", "장지후", "임준서"]

# Amateur futsal: same bib per team, but shorts (and height) differ per player -> appearance cues for ID tracking.
VEST_BGR = {"A": (40, 40, 230), "B": (230, 110, 40)}                 # red / blue bibs
GK_VEST_BGR = {"A": (0, 220, 240), "B": (200, 60, 200)}              # yellow / purple goalkeeper shirts
SHORTS_BGR = [(25, 25, 25), (235, 235, 235), (90, 40, 20), (128, 128, 128), (40, 30, 120), (150, 190, 210), (120, 120, 0)]

# (speed m/s, probability) for each new run; amateur futsal outfield players cover roughly 80-110 m/min
GAITS = [(1.2, 0.20), (2.8, 0.45), (4.5, 0.25), (6.5, 0.10)]
MAX_ACCEL = 3.0     # m/s^2


def synthetic_match(court: Court, seconds: float = 120.0, fps: float = 10.0, seed: int = 0) -> TrackSet:
    """Each player repeatedly picks a waypoint (a mix of their role position, a wandering ball and some
    randomness) and runs to it at a sampled gait with limited acceleration."""
    rng = np.random.default_rng(seed)
    look = np.random.default_rng(seed + 10_000)      # separate stream: appearance must not change the motion
    n, dt = int(seconds * fps), 1.0 / fps
    L, W = court.length, court.width
    t = np.arange(n) * dt
    ph = rng.uniform(0, 2 * np.pi, 4)
    ball = np.stack([L / 2 + 0.42 * L * np.sin(t / 4.1 + ph[0]) * np.cos(t / 11.0 + ph[1]),
                     W / 2 + 0.40 * W * np.sin(t / 2.7 + ph[2]) * np.cos(t / 7.3 + ph[3])], 1)
    roles = {"A": [(0.015, 0.5), (0.22, 0.25), (0.22, 0.75), (0.40, 0.5), (0.50, 0.35)],
             "B": [(0.985, 0.5), (0.78, 0.25), (0.78, 0.75), (0.60, 0.5), (0.50, 0.65)]}
    speeds, probs = np.array([g[0] for g in GAITS]), np.array([g[1] for g in GAITS])
    players, pid = [], 1
    for team, slots in roles.items():
        for k, (rx, ry) in enumerate(slots):
            gk = k == 0
            home = np.array([rx * L, ry * W])
            p, v = home.copy(), np.zeros(2)
            xy = np.empty((n, 2))
            next_change, target, speed = 0, home, 0.0
            for i in range(n):
                if i >= next_change:
                    if gk:
                        target = np.array([home[0], np.clip(ball[i, 1], W / 2 - 3, W / 2 + 3)])
                        speed = rng.choice([0.8, 2.0, 3.5], p=[.5, .35, .15])
                    else:
                        w = rng.uniform(0.3, 0.8)
                        target = (1 - w) * home + w * ball[i] + rng.normal(0, 3.0, 2)
                        speed = rng.choice(speeds, p=probs)
                    target = np.clip(target, [0.5, 0.5], [L - 0.5, W - 0.5])
                    next_change = i + int(rng.uniform(1.5, 5.0) * fps)
                d = target - p
                dist = np.linalg.norm(d)
                desired = d / dist * min(speed, dist * 1.5) if dist > 1e-6 else np.zeros(2)
                dv = desired - v
                dvn = np.linalg.norm(dv)
                if dvn > MAX_ACCEL * dt:
                    dv *= MAX_ACCEL * dt / dvn
                v = v + dv
                p = np.clip(p + v * dt, [0.3, 0.3], [L - 0.3, W - 0.3])
                xy[i] = p
            meta = {"height": float(np.clip(look.normal(1.75, 0.06), 1.60, 1.92)),
                    "vest": GK_VEST_BGR[team] if gk else VEST_BGR[team],
                    "shorts": SHORTS_BGR[int(look.integers(len(SHORTS_BGR)))]}    # duplicates happen, as in real life
            players.append(PlayerTrack(pid, team, xy, name=KOREAN_NAMES[pid - 1], meta=meta))
            pid += 1
    return TrackSet(L, W, fps, players)
