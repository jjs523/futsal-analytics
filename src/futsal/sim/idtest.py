"""End-to-end ID tracking experiment on synthetic video: render both phones, detect, fuse, track,
score against ground truth. Compares the motion-only tracker with the appearance-aware one."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from ..court import Court
from ..homography import calibrate
from ..pipeline import detect, run
from . import evaluate, layout, observe, scenario, video


def run_experiment(court: Court, seconds: float = 120, seed: int = 5, hfov: float = layout.HFOV_AVERAGE,
                   twist: float = 5.0, workdir: str | None = None) -> dict:
    truth = scenario.synthetic_match(court, seconds, fps=10, seed=seed)
    cams = layout.diagonal(court, twist).build(hfov)
    workdir = workdir or tempfile.mkdtemp(prefix="idtest_")
    rng = np.random.default_rng(seed)
    dets, cals = {}, {}
    for name, cam in cams.items():
        det = detect.synthetic_detector(video.background(cam, court, 0.5), video.VEST_TO_TEAM)
        path = os.path.join(workdir, f"{name}.mp4")
        video.render(cam, court, truth, path, scale=0.5)
        dets[name], _, _ = detect.detect_video(path, det, stride=1, feat_every=2, scale=2.0)
        cals[name] = calibrate(observe.tap_keypoints(cam, court, 2.0, rng), court)
    out = {}
    for mode in ("motion", "appearance"):
        est = run.build_tracks(court, dets, cals, fps_out=truth.fps, ids=mode)
        out[mode] = evaluate.id_scores(truth, est)
    return out
