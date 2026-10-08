"""Player detection on video frames.

Each detection keeps the foot point (bottom-centre of the box, in pixels), the box height and, every few
frames, an appearance descriptor (colour histograms of the upper and lower body) used later to keep player
IDs apart: bibs are the same within a team, but shorts / socks / skin usually are not.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

# A detector takes a BGR frame and returns boxes [(x1, y1, x2, y2, confidence, team_or_None), ...]
Box = tuple[float, float, float, float, float, "str | None"]
Detector = Callable[[np.ndarray], list[Box]]

HUE_BINS = 12            # chromatic hue bins; plus black / grey / white bins for achromatic pixels
FEAT_LEN = 2 * (HUE_BINS + 3)


@dataclass
class Detection:
    frame: int
    t: float              # seconds since this camera started recording (segment t0 already added)
    u: float
    v: float
    conf: float
    team: str | None = None
    h_px: float = 0.0     # box height in pixels
    feat: list[int] | None = None    # upper + lower body histograms, each scaled to sum 255 (see appearance())


def _hist(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel().astype(int), hsv[..., 2].ravel().astype(int)
    out = np.zeros(HUE_BINS + 3)
    if not len(h):
        return out
    black = v < 60
    achrom = ~black & (s < 50)
    white = achrom & (v > 170)
    grey = achrom & ~white
    chrom = ~black & ~achrom
    out[:HUE_BINS] = np.bincount((h[chrom].astype(int) * HUE_BINS) // 180, minlength=HUE_BINS)[:HUE_BINS]
    out[HUE_BINS:] = black.sum(), grey.sum(), white.sum()
    return out / max(out.sum(), 1)


def appearance(frame: np.ndarray, box) -> np.ndarray:
    """Upper-body (10-50 % of the box height) and lower-body (55-85 %) colour histograms from the central
    60 % of the box width, which keeps most of the background out."""
    x1, y1, x2, y2 = (float(b) for b in box[:4])
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    xa, xb = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
    parts = []
    for a, b in ((0.10, 0.50), (0.55, 0.85)):
        ya, yb = int(max(0, y1 + a * h)), int(min(H, y1 + b * h))
        crop = frame[ya:max(yb, ya + 1), xa:max(xb, xa + 1)]
        parts.append(_hist(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)) if crop.size else np.zeros(HUE_BINS + 3))
    return np.concatenate(parts)


def quantize(f: np.ndarray) -> list[int]:
    return [int(round(x * 255)) for x in f]


def dequantize(q) -> np.ndarray | None:
    return None if q is None else np.asarray(q, float) / 255.0


def detect_video(path: str, detector: Detector, stride: int = 3, t0: float = 0.0, feat_every: int = 2,
                 scale: float = 1.0, min_feat_px: float = 24, max_aspect: float = 0.6,
                 rate_hz: float | None = None, start_s: float = 0.0, end_s: float | None = None,
                 progress: bool = False) -> tuple[list[Detection], float, int]:
    """Run `detector` on every `stride`-th frame (appearance on every `feat_every`-th of those).
    `rate_hz` sets the stride from the video's frame rate instead (10 -> every 3rd frame at 30 fps, every 6th at 60),
    so two phones recording at different frame rates give detections at the same rate.
    `start_s` / `end_s` limit the work to part of the video (times stay those of the whole video).
    Appearance is skipped for boxes too small to have meaningful colours (< `min_feat_px` tall in the video)
    and for boxes wider than `max_aspect` x height, which usually hold two overlapping players.
    `scale` maps pixel coordinates back to the calibrated resolution (e.g. 2.0 if the video is half size).
    Returns detections, the video fps and its frame count."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if rate_hz:
        stride = max(1, int(round(fps / rate_hz)))
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    out, i, k = [], 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if end_s is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end_s:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            # Phone camera apps often record variable frame rate: use each frame's own timestamp, not i / fps.
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms > 0 or (i == 0 and start_s <= 0) else start_s + i / fps
            with_feat = k % feat_every == 0
            for x1, y1, x2, y2, c, team in detector(frame):
                good = with_feat and (y2 - y1) >= min_feat_px and (x2 - x1) <= max_aspect * (y2 - y1)
                feat = quantize(appearance(frame, (x1, y1, x2, y2))) if good else None
                out.append(Detection(i, t0 + t, (x1 + x2) / 2 * scale, y2 * scale, float(c), team, (y2 - y1) * scale, feat))
            k += 1
            if progress and k % 200 == 0:
                span = f"/{end_s - start_s:.0f}" if end_s is not None else ""
                print(f"    {t - start_s:.0f}{span}초")
        i += 1
    cap.release()
    return out, float(fps), i


def save(dets: list[Detection], path: str, **meta) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "detections": [asdict(d) for d in dets]}, f, separators=(",", ":"))


def load(path: str) -> tuple[list[Detection], dict]:
    with open(path) as f:
        d = json.load(f)
    return [Detection(**x) for x in d["detections"]], d["meta"]


def yolo_detector(model: str = "yolo11s.pt", conf: float = 0.3, imgsz: int = 1280, tiles: int = 1) -> Detector:
    """Pretrained COCO 'person' detector from Ultralytics (pip install '.[vision]').
    tiles=2 runs a 2x2 overlapping tiling for small far-side players (4x slower)."""
    from ultralytics import YOLO   # optional dependency
    net = YOLO(model)

    def boxes(img):
        r = net.predict(img, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def detect(frame):
        if tiles == 1:
            xyxy, cf = boxes(frame)
        else:
            h, w = frame.shape[:2]
            th, tw = int(h / tiles * 1.2), int(w / tiles * 1.2)
            all_b, all_c = [], []
            for ty in np.linspace(0, h - th, tiles).astype(int):
                for tx in np.linspace(0, w - tw, tiles).astype(int):
                    b, c = boxes(frame[ty:ty + th, tx:tx + tw])
                    all_b.append(b + [tx, ty, tx, ty]); all_c.append(c)
            xyxy, cf = np.vstack(all_b), np.concatenate(all_c)
            if len(xyxy):
                rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy]
                keep = np.array(cv2.dnn.NMSBoxes(rects, cf.tolist(), conf, 0.5)).reshape(-1)
                xyxy, cf = xyxy[keep], cf[keep]
        return [(float(x1), float(y1), float(x2), float(y2), float(c), None) for (x1, y1, x2, y2), c in zip(xyxy, cf)]
    return detect


def synthetic_detector(background: np.ndarray, vest_to_team: dict, min_area: int = 30, tol: int = 60) -> Detector:
    """Detector for videos made by futsal.sim.video: whatever differs from the empty-pitch background is a
    player box; the team comes from the colour at the top of the box. Overlapping players merge into one
    box, much like a real detector struggles with occlusion."""
    bg = background.astype(int)
    vests = np.array(list(vest_to_team), float)
    teams = list(vest_to_team.values())
    kernel = np.ones((3, 3), np.uint8)

    def detect(frame):
        fg = (np.abs(frame.astype(int) - bg).sum(2) > tol).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)              # drop compression speckle
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
        out = []
        for x, y, w, h, area in stats[1:]:
            if area < min_area or h < 6:
                continue
            top = frame[y + max(1, h // 10): y + max(2, h // 3), x + w // 4: x + max(w - w // 4, w // 4 + 1)].reshape(-1, 3)
            team = teams[int(np.argmin(np.linalg.norm(vests - top.mean(0), axis=1)))] if len(top) else None
            # pixel centres: the last foreground row is y + h - 1, so the box bottom edge is at y + h - 0.5
            out.append((x - 0.5, y - 0.5, x + w - 0.5, y + h - 0.5, 1.0, team))
        return out
    return detect


def color_blob_detector(team_colors: dict[str, tuple[int, int, int]], tol: int = 40, min_area: int = 30) -> Detector:
    """Older synthetic detector: solid team-coloured boxes only."""
    def detect(frame):
        out = []
        for team, bgr in team_colors.items():
            lo = np.clip(np.array(bgr) - tol, 0, 255).astype(np.uint8)
            hi = np.clip(np.array(bgr) + tol, 0, 255).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(frame, lo, hi))
            for x, y, w, h, area in stats[1:]:
                if area >= min_area:
                    out.append((float(x), float(y), float(x + w), float(y + h), 1.0, team))
        return out
    return detect
