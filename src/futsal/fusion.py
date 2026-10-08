"""Merging what two cameras see into one set of pitch positions.

Each camera's detections are already in pitch coordinates (via its homography) and carry an
uncertainty sigma (metres), usually `detector jitter (px) * metres_per_pixel` at the foot point.
Detections from different cameras closer than `gate` metres are treated as the same player and
merged by inverse-variance weighting, so the nearer camera dominates automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    xy: np.ndarray           # (2,) pitch position, metres
    sigma: float             # 1-sigma position uncertainty, metres
    camera: str
    track_id: int | None = None
    team: str | None = None
    feat: np.ndarray | None = None     # appearance descriptor (pipeline.detect.appearance), if computed


@dataclass
class Fused:
    xy: np.ndarray
    sigma: float
    sources: list[Observation]

    @property
    def team(self) -> str | None:
        teams = [o.team for o in self.sources if o.team]
        return max(set(teams), key=teams.count) if teams else None

    @property
    def feat(self) -> np.ndarray | None:
        feats = [o.feat for o in self.sources if o.feat is not None]
        return np.mean(feats, axis=0) if feats else None


def inverse_variance_mean(obs: list[Observation]) -> tuple[np.ndarray, float]:
    w = np.array([1.0 / max(o.sigma, 1e-6) ** 2 for o in obs])
    xy = (np.stack([o.xy for o in obs]) * w[:, None]).sum(0) / w.sum()
    return xy, float(np.sqrt(1.0 / w.sum()))


def fuse_frame(per_camera: dict[str, list[Observation]], gate: float = 1.0) -> list[Fused]:
    """Greedy cross-camera association for one timestamp (2 cameras, ~10 players: greedy is enough).
    Pairs are matched in order of increasing distance; anything left unmatched passes through.
    The gate widens to 3 sigma where the cameras are imprecise (far side), never below `gate`."""
    cams = list(per_camera)
    if not cams:
        return []
    if len(cams) == 1:
        return [Fused(o.xy, o.sigma, [o]) for o in per_camera[cams[0]]]
    if len(cams) > 2:
        raise NotImplementedError("fuse_frame handles one or two cameras")
    a, b = per_camera[cams[0]], per_camera[cams[1]]
    pairs = sorted(((float(np.linalg.norm(oa.xy - ob.xy)), i, j) for i, oa in enumerate(a) for j, ob in enumerate(b)))
    used_a, used_b, out = set(), set(), []
    for d, i, j in pairs:
        if d > max(gate, 3 * np.hypot(a[i].sigma, b[j].sigma)):
            continue
        if a[i].team and b[j].team and a[i].team != b[j].team:
            continue
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        xy, s = inverse_variance_mean([a[i], b[j]])
        out.append(Fused(xy, s, [a[i], b[j]]))
    out += [Fused(o.xy, o.sigma, [o]) for i, o in enumerate(a) if i not in used_a]
    out += [Fused(o.xy, o.sigma, [o]) for j, o in enumerate(b) if j not in used_b]
    return out
