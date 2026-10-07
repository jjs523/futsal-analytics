"""Synthetic end-to-end run of the geometric half of the pipeline.

ground-truth tracks -> each phone projects the feet (+ detector jitter) -> calibration from noisy taps
-> pitch coordinates per phone -> inverse-variance fusion -> estimated tracks.
Player identities are taken as known: this measures geometry (placement, calibration, fusion),
not detection or ID tracking.
"""
from __future__ import annotations

import numpy as np

from ..court import Court
from ..fusion import Observation, inverse_variance_mean
from ..homography import Calibration, calibrate
from ..tracks import PlayerTrack, TrackSet
from .layout import Layout


def tap_keypoints(cam, court: Court, click_px: float, rng) -> dict[str, tuple[float, float]]:
    kp = court.keypoints()
    uv, ok = cam.project(np.array(list(kp.values())))
    uv = uv + rng.normal(0, click_px, uv.shape)
    return {n: (float(p[0]), float(p[1])) for n, p, o in zip(kp, uv, ok) if o}


def run(truth: TrackSet, layout: Layout, hfov: float, det_px: float = 3.0, click_px: float = 2.0,
        seed: int = 0) -> tuple[TrackSet, dict[str, Calibration]]:
    rng = np.random.default_rng(seed)
    court = Court(truth.length, truth.width)
    cams = layout.build(hfov)
    cals = {name: calibrate(tap_keypoints(cam, court, click_px, rng), court) for name, cam in cams.items()}
    out = []
    for p in truth.players:
        obs_per_cam = {}
        for name, cam in cams.items():
            seen = cam.sees_player(p.xy)
            uv, _ = cam.project(p.xy)
            uv = uv + rng.normal(0, det_px, uv.shape)
            xy = np.full_like(p.xy, np.nan)
            sig = np.full(len(p.xy), np.nan)
            if seen.any():
                xy[seen] = cals[name].to_pitch(uv[seen])
                sig[seen] = det_px * cals[name].metres_per_pixel(uv[seen])
            obs_per_cam[name] = (xy, sig)
        est = np.full_like(p.xy, np.nan)
        for i in range(len(p.xy)):
            obs = [Observation(xy[i], float(s[i]), n) for n, (xy, s) in obs_per_cam.items() if np.isfinite(s[i])]
            if obs:
                est[i] = inverse_variance_mean(obs)[0]
        out.append(PlayerTrack(p.id, p.team, est, name=p.name))
    return TrackSet(truth.length, truth.width, truth.fps, out, truth.start), cals


def position_errors(truth: TrackSet, est: TrackSet) -> np.ndarray:
    e = np.concatenate([np.linalg.norm(a.xy - b.xy, axis=1) for a, b in zip(truth.players, est.players)])
    return e[np.isfinite(e)]
