"""Identity tracking on the pitch plane (after both cameras are fused).

Tracking in metres instead of per-camera pixels means one tracker covers both phones, and a player
who walks out of one phone's view keeps the same ID as long as the other phone still sees them.
Constant-velocity prediction + greedy nearest-neighbour association with a distance gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..fusion import Fused
from ..tracks import PlayerTrack


@dataclass
class _Track:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    last: int
    points: dict[int, np.ndarray] = field(default_factory=dict)
    teams: Counter = field(default_factory=Counter)


def track(frames: list[list[Fused]], fps: float, gate: float = 1.5, max_gap_s: float = 2.0,
          min_length_s: float = 1.0) -> list[PlayerTrack]:
    dt = 1.0 / fps
    max_gap = int(max_gap_s * fps)
    alive: list[_Track] = []
    done: list[_Track] = []
    next_id = 1
    for i, dets in enumerate(frames):
        preds = [t.pos + t.vel * (i - t.last) * dt for t in alive]
        pairs = sorted((float(np.linalg.norm(d.xy - p)), ti, di) for ti, p in enumerate(preds) for di, d in enumerate(dets))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > gate * (1 + 0.5 * (i - alive[ti].last - 1)):   # widen the gate while a track coasts
                continue
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            t, d = alive[ti], dets[di]
            gap = i - t.last
            new_vel = (d.xy - t.pos) / (gap * dt)
            t.vel = 0.6 * t.vel + 0.4 * new_vel
            t.pos, t.last = d.xy.copy(), i
            t.points[i] = d.xy.copy()
            t.teams.update(o.team for o in d.sources if o.team)
        for di, d in enumerate(dets):
            if di not in used_d:
                t = _Track(next_id, d.xy.copy(), np.zeros(2), i, {i: d.xy.copy()})
                t.teams.update(o.team for o in d.sources if o.team)
                alive.append(t); next_id += 1
        still = []
        for t in alive:
            (still if i - t.last <= max_gap else done).append(t)
        alive = still
    done += alive
    n = len(frames)
    out = []
    for t in sorted(done, key=lambda t: t.id):
        if len(t.points) < min_length_s * fps:
            continue
        xy = np.full((n, 2), np.nan)
        for i, p in t.points.items():
            xy[i] = p
        team = t.teams.most_common(1)[0][0] if t.teams else "?"
        out.append(PlayerTrack(len(out) + 1, team, xy))
    return out


def stitch(tracks: list[PlayerTrack], fps: float, max_gap_s: float = 4.0, max_speed: float = 7.0,
           slack: float = 1.5) -> list[PlayerTrack]:
    """Offline ID stitching (a light version of global tracklet association): join a track that ends to one
    that starts later if the same team could have run between the two points in the gap. Shortest jumps first."""
    def ends(t):
        idx = np.flatnonzero(np.isfinite(t.xy[:, 0]))
        return idx[0], idx[-1]
    tracks = [PlayerTrack(t.id, t.team, t.xy.copy(), t.name) for t in tracks]
    merged = True
    while merged:
        merged = False
        cands = []
        for a in tracks:
            a0, a1 = ends(a)
            for b in tracks:
                if a is b or a.team != b.team:
                    continue
                b0, b1 = ends(b)
                gap = b0 - a1
                if gap <= 0 or gap > max_gap_s * fps:
                    continue
                d = float(np.linalg.norm(b.xy[b0] - a.xy[a1]))
                if d <= max_speed * gap / fps + slack:
                    cands.append((d / (gap / fps + 1e-9), d, id(a), id(b)))
        if cands:
            _, _, ia, ib = min(cands)
            a = next(t for t in tracks if id(t) == ia)
            b = next(t for t in tracks if id(t) == ib)
            fill = np.isfinite(b.xy[:, 0])
            a.xy[fill] = b.xy[fill]
            tracks.remove(b)
            merged = True
    tracks.sort(key=lambda t: -np.isfinite(t.xy[:, 0]).sum())
    for i, t in enumerate(tracks, 1):
        t.id = i
    return tracks
