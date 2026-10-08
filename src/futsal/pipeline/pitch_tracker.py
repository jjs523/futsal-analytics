"""Identity tracking on the pitch plane (after both cameras are fused).

Tracking in metres instead of per-camera pixels means one tracker covers both phones, and a player
who walks out of one phone's view keeps the same ID as long as the other phone still sees them.

Two trackers live here:
  track() + stitch()          motion only: constant-velocity prediction, greedy matching, gap stitching
  tracklets() + associate()   default: unambiguous short tracklets re-linked over the whole match using
                              appearance (shorts colour etc.) + motion + no-overlap constraints
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


# ---------------------------------------------------------------------------------------------------------
# Appearance-aware IDs: 1) cut the match into short tracklets that are almost certainly one person,
# 2) re-link the tracklets over the whole match with appearance + motion + "one person cannot be in two
# places at once". Runs after the match, so future frames are available.
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Tracklet:
    id: int
    team: str | None
    points: dict[int, np.ndarray] = field(default_factory=dict)
    feats: list[np.ndarray] = field(default_factory=list)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def start(self) -> int:
        return min(self.points)

    @property
    def end(self) -> int:
        return max(self.points)

    def feat(self) -> np.ndarray | None:
        return np.median(self.feats, axis=0) if self.feats else None       # robust to a few bad crops


def _same_team(a, b) -> bool:
    return a is None or b is None or a == b


def tracklets(frames: list[list[Fused]], fps: float, gate: float = 1.5, ambiguity: float = 0.7,
              max_gap_s: float = 1.0) -> list[Tracklet]:
    """Greedy frame-to-frame linking that refuses to guess: when a track has two plausible detections, or a
    detection is plausible for two tracks (players crossing, merged boxes), those tracks end and new
    tracklets start. Every tracklet is then very likely a single person; joining them is associate()'s job."""
    dt = 1.0 / fps
    alive: list[Tracklet] = []
    done: list[Tracklet] = []
    next_id = 1
    for i, dets in enumerate(frames):
        preds = [t.points[t.end] + t.vel * (i - t.end) * dt for t in alive]
        dist = np.full((len(alive), len(dets)), np.inf)
        for ti, (t, p) in enumerate(zip(alive, preds)):
            g = gate * (1 + 0.5 * (i - t.end - 1))
            for di, d in enumerate(dets):
                if _same_team(t.team, d.team):
                    dd = float(np.linalg.norm(d.xy - p))
                    if dd <= g:
                        dist[ti, di] = dd
        ambiguous_t, ambiguous_d = set(), set()
        for ti in range(len(alive)):
            c = np.sort(dist[ti][np.isfinite(dist[ti])])
            if len(c) >= 2 and c[1] - c[0] < ambiguity:
                ambiguous_t.add(ti)
        for di in range(len(dets)):
            c = np.sort(dist[:, di][np.isfinite(dist[:, di])])
            if len(c) >= 2 and c[1] - c[0] < ambiguity:
                ambiguous_d.add(di)
                ambiguous_t.update(np.flatnonzero(np.isfinite(dist[:, di]) & (dist[:, di] - c[0] < ambiguity)).tolist())
        used_t, used_d = set(), set()
        for ti, di in sorted(((ti, di) for ti, di in zip(*np.nonzero(np.isfinite(dist)))), key=lambda p: dist[p]):
            if ti in used_t or di in used_d or ti in ambiguous_t or di in ambiguous_d:
                continue
            used_t.add(ti); used_d.add(di)
            t, d = alive[ti], dets[di]
            gap = i - t.end
            t.vel = 0.6 * t.vel + 0.4 * (d.xy - t.points[t.end]) / (gap * dt)
            t.points[i] = d.xy.copy()
            if d.feat is not None:
                t.feats.append(d.feat)
            if t.team is None:
                t.team = d.team
        still = []
        for ti, t in enumerate(alive):
            if ti in ambiguous_t or i - t.end > max_gap_s * fps:
                done.append(t)
            else:
                still.append(t)
        alive = still
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Tracklet(next_id, d.team, {i: d.xy.copy()}, [d.feat] if d.feat is not None else [])
                alive.append(t); next_id += 1
    return done + alive


def feat_distance(a: np.ndarray | None, b: np.ndarray | None, lower_weight: float = 0.7) -> float:
    """Hellinger distance (0 = identical, 1 = disjoint) between appearance descriptors; the lower body
    (shorts) counts more because bibs are shared within a team."""
    if a is None or b is None:
        return 0.5
    h = len(a) // 2

    def hel(p, q):
        p, q = p / max(p.sum(), 1e-9), q / max(q.sum(), 1e-9)
        return float(np.sqrt(max(0.0, 1 - np.sqrt(p * q).sum())))
    return (1 - lower_weight) * hel(a[:h], b[:h]) + lower_weight * hel(a[h:], b[h:])


class _Cluster:
    def __init__(self, t: Tracklet):
        self.parts = [t]
        self.team = t.team
        self._feat = t.feat()

    def feat(self):
        return self._feat

    def _update_feat(self):
        fs = [f for p in self.parts for f in p.feats]
        self._feat = np.median(fs, axis=0) if fs else None

    def frames(self) -> int:
        return sum(len(p.points) for p in self.parts)

    def merge(self, other: "_Cluster") -> None:
        self.parts = sorted(self.parts + other.parts, key=lambda p: p.start)
        self.team = self.team or other.team
        self._update_feat()


def associate(tracks: list[Tracklet], n_frames: int, fps: float, max_speed: float = 7.0, slack: float = 1.5,
              long_gap_s: float = 6.0, w_app: float = 1.0, w_motion: float = 0.35, max_cost: float = 0.55,
              long_gap_max_app: float = 0.2, min_length_s: float = 1.0, max_overlap: int = 5,
              dup_dist: float = 1.5) -> list[PlayerTrack]:
    """Agglomerative linking of tracklets into players (cheapest join first). A join is allowed only if the
    two groups never overlap in time, the team agrees, and at every hand-over the player could have run the
    distance (or, after a long absence such as a substitution, the appearance is a close match)."""
    clusters = [_Cluster(t) for t in tracks if t.points]

    def junction_cost(c1: _Cluster, c2: _Cluster) -> float:
        parts = sorted(c1.parts + c2.parts, key=lambda p: p.start)
        mine = {id(p) for p in c1.parts}
        own = {id(p): (c1 if id(p) in mine else c2) for p in parts}
        worst = 0.0
        for a, b in zip(parts, parts[1:]):
            if own[id(a)] is own[id(b)]:
                continue
            if b.start <= a.end:                                 # overlap: two people, unless it is a short
                shared = [f for f in b.points if f in a.points]  # duplicate of the same player (unfused cameras)
                if len(shared) > max_overlap or any(np.linalg.norm(a.points[f] - b.points[f]) > dup_dist for f in shared):
                    return np.inf
                continue
            gap = (b.start - a.end) / fps
            d = float(np.linalg.norm(b.points[b.start] - a.points[a.end]))
            reach = max_speed * gap + slack
            if gap > long_gap_s:
                worst = max(worst, 0.5)                          # long absence: motion says little
            elif d > reach:
                return np.inf
            else:
                worst = max(worst, d / reach)
        return worst

    def cost(c1: _Cluster, c2: _Cluster) -> float:
        if not _same_team(c1.team, c2.team):
            return np.inf
        motion = junction_cost(c1, c2)
        if not np.isfinite(motion):
            return np.inf
        app = feat_distance(c1.feat(), c2.feat())
        if motion >= 0.5 and app > long_gap_max_app:             # long gap needs a confident appearance match
            return np.inf
        return w_app * app + w_motion * motion

    n = len(clusters)
    C = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = cost(clusters[i], clusters[j])
    alive = np.ones(n, bool)
    while True:
        k = int(np.argmin(C))
        i, j = divmod(k, n)
        if not np.isfinite(C[i, j]) or C[i, j] > max_cost:
            break
        clusters[i].merge(clusters[j])
        alive[j] = False
        C[j, :] = np.inf; C[:, j] = np.inf
        for m in np.flatnonzero(alive):
            if m != i:
                c = cost(clusters[min(i, m)], clusters[max(i, m)])
                C[min(i, m), max(i, m)] = c
    out = []
    for c in sorted((clusters[i] for i in np.flatnonzero(alive)), key=lambda c: -c.frames()):
        if c.frames() < min_length_s * fps:
            continue
        xy = np.full((n_frames, 2), np.nan)
        for p in c.parts:
            for f, q in p.points.items():
                xy[f] = q
        out.append(PlayerTrack(len(out) + 1, c.team or "?", xy))
    return out


def absorb_duplicates(tracks: list[PlayerTrack], dist: float = 1.5, min_share: float = 0.6,
                      min_together: int = 5) -> list[PlayerTrack]:
    """A shorter track that, whenever it coexists with a longer same-team track, sits within `dist` metres
    of it is the same player seen twice (cameras not fused, or a split cluster): fold it into the longer one."""
    tracks = sorted(tracks, key=lambda t: -np.isfinite(t.xy[:, 0]).sum())
    keep: list[PlayerTrack] = []
    for t in tracks:
        seen = np.isfinite(t.xy[:, 0])
        host = None
        for k in keep:
            if not _same_team(k.team, t.team) and k.team != "?" and t.team != "?":
                continue
            both = seen & np.isfinite(k.xy[:, 0])
            if both.sum() < min_together:
                continue
            close = np.linalg.norm(t.xy[both] - k.xy[both], axis=1) <= dist
            if close.mean() >= min_share:
                host = k
                break
        if host is None:
            keep.append(PlayerTrack(t.id, t.team, t.xy.copy(), t.name))
        else:
            fill = seen & ~np.isfinite(host.xy[:, 0])
            host.xy[fill] = t.xy[fill]
    for i, t in enumerate(keep, 1):
        t.id = i
    return keep
