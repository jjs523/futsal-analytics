"""Render a synthetic phone video (pitch lines, goals, players as boxes: bib colour on top, the player's own
shorts colour below, box height from the player's height) for pipeline tests and demos without real footage."""
from __future__ import annotations

import cv2
import numpy as np

from ..camera import PLAYER_HEIGHT, Camera
from ..court import Court
from ..tracks import TrackSet

from .scenario import GK_VEST_BGR, VEST_BGR

TEAM_BGR = VEST_BGR                                        # kept for older callers
GRASS_BGR = (59, 122, 42)
VEST_TO_TEAM = {**{c: t for t, c in VEST_BGR.items()}, **{c: t for t, c in GK_VEST_BGR.items()}}


def _polyline(cam: Camera, P3: np.ndarray, scale: float) -> np.ndarray | None:
    Xc = (cam.R @ (P3 - cam.C).T).T
    if (Xc[:, 2] < .1).any():
        return None
    uv = (cam.K @ Xc.T).T
    return (uv[:, :2] / uv[:, 2:] * scale).astype(np.int32)


def background(cam: Camera, court: Court, scale: float = 0.5) -> np.ndarray:
    """The empty pitch as seen by `cam` (what a background model would learn from a fixed tripod)."""
    w, h = int(cam.width * scale), int(cam.height * scale)
    base = np.full((h, w, 3), GRASS_BGR, np.uint8)
    for m in court.markings():
        q = np.vstack([np.linspace(m[i], m[i + 1], 12) for i in range(len(m) - 1)])
        pts = _polyline(cam, np.hstack([q, np.zeros((len(q), 1))]), scale)
        if pts is not None:
            cv2.polylines(base, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    for g in court.goal_frames():
        pts = _polyline(cam, g, scale)
        if pts is not None:
            cv2.polylines(base, [pts], False, (240, 240, 240), 2, cv2.LINE_AA)
    return base


def render(cam: Camera, court: Court, truth: TrackSet, path: str, scale: float = 0.5, start: int = 0, end: int | None = None):
    """Write frames start..end of `truth` as seen by `cam`, at truth.fps, downscaled by `scale`."""
    base = background(cam, court, scale)
    h, w = base.shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), truth.fps, (w, h))
    end = truth.n_frames if end is None else end
    for i in range(start, end):
        img = base.copy()
        order = sorted(truth.players, key=lambda p: -np.linalg.norm(p.xy[i] - cam.C[:2]))   # far players first
        for p in order:
            foot, ok = cam.project(p.xy[i])
            if not ok[0]:
                continue
            height = p.meta.get("height", PLAYER_HEIGHT)
            head, _ = cam.project(np.r_[p.xy[i], height])
            ph = (foot[0, 1] - head[0, 1]) * scale
            x, y = foot[0, 0] * scale, foot[0, 1] * scale
            x0, x1, top, waist = int(round(x - ph * .17)), int(round(x + ph * .17)), int(round(y - ph)), int(round(y - ph * .45))
            cv2.rectangle(img, (x0, top), (x1, waist), p.meta.get("vest", TEAM_BGR[p.team]), -1)
            cv2.rectangle(img, (x0, waist), (x1, int(round(y))), p.meta.get("shorts", (25, 25, 25)), -1)
        vw.write(img)
    vw.release()
