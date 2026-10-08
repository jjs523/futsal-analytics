"""Tracker variants for experiments/harness.py. Each is registered by name and maps Data -> list[Track]."""
from __future__ import annotations

import numpy as np

from harness import Data, Track, TrackPoint, register

from futsal.fusion import Observation, fuse_frame
from futsal.pipeline import pitch_tracker


def observations(data: Data, k: int, min_conf: float = 0.3) -> dict[str, list[Observation]]:
    """Observations of grid frame k per camera; Observation.track_id carries the box index."""
    out = {}
    for cam, c in data.cams.items():
        obs = []
        for i in c.boxes_at(k, min_conf):
            ok_feat = c.h[i] >= 24 and c.w[i] <= 0.6 * c.h[i]
            obs.append(Observation(c.xy[i].copy(), float(c.sigma[i]), cam, track_id=int(i),
                                   feat=c.color[i] if ok_feat else None))
        out[cam] = obs
    return out


def fused_frames(data: Data, min_conf: float = 0.3, gate: float = 1.0):
    return [fuse_frame(observations(data, k, min_conf), gate=gate) for k in range(data.n)]


def from_player_tracks(players, frames) -> list[Track]:
    """PlayerTrack xy arrays back to Tracks with box references (match each point to the fused detection it came from)."""
    out = []
    for p in players:
        tr: Track = {}
        for k in np.flatnonzero(np.isfinite(p.xy[:, 0])):
            boxes = []
            for f in frames[k]:
                if np.allclose(f.xy, p.xy[k], atol=1e-9):
                    boxes = [(o.camera, o.track_id) for o in f.sources]
                    break
            tr[int(k)] = TrackPoint(p.xy[k].copy(), boxes)
        out.append(tr)
    return out


@register("baseline")
def baseline(data: Data) -> list[Track]:
    """The repository's current default (pipeline.run.build_tracks, ids='appearance') on conf >= 0.3 boxes."""
    frames = fused_frames(data, 0.3)
    players = pitch_tracker.associate(pitch_tracker.tracklets(frames, data.rate, ambiguity=0.4), len(frames), data.rate)
    players = pitch_tracker.absorb_duplicates(players)
    return from_player_tracks(players, frames)


@register("baseline_motion")
def baseline_motion(data: Data) -> list[Track]:
    frames = fused_frames(data, 0.3)
    players = pitch_tracker.stitch(pitch_tracker.track(frames, data.rate), data.rate)
    return from_player_tracks(players, frames)
