"""Player detection on video frames. Output is the foot point (bottom-centre of the box) in pixels."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

# A detector takes a BGR frame and returns [(u_foot, v_foot, confidence, team_or_None), ...]
Detector = Callable[[np.ndarray], list[tuple[float, float, float, str | None]]]


@dataclass
class Detection:
    frame: int
    t: float              # seconds since this camera started recording (segment t0 already added)
    u: float
    v: float
    conf: float
    team: str | None = None


def detect_video(path: str, detector: Detector, stride: int = 3, t0: float = 0.0) -> tuple[list[Detection], float, int]:
    """Run `detector` on every `stride`-th frame. Returns detections, the video fps and its frame count."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            for u, v, c, team in detector(frame):
                out.append(Detection(i, t0 + i / fps, float(u), float(v), float(c), team))
        i += 1
    cap.release()
    return out, float(fps), i


def save(dets: list[Detection], path: str, **meta) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "detections": [asdict(d) for d in dets]}, f)


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
        return [((x1 + x2) / 2, y2, float(c), None) for (x1, y1, x2, y2), c in zip(xyxy, cf)]
    return detect


def color_blob_detector(team_colors: dict[str, tuple[int, int, int]], tol: int = 40, min_area: int = 30) -> Detector:
    """Finds solid-coloured player boxes in synthetic videos (tests and demos). team_colors are BGR."""
    def detect(frame):
        out = []
        for team, bgr in team_colors.items():
            lo = np.clip(np.array(bgr) - tol, 0, 255).astype(np.uint8)
            hi = np.clip(np.array(bgr) + tol, 0, 255).astype(np.uint8)
            mask = cv2.inRange(frame, lo, hi)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, area in stats[1:]:
                if area >= min_area:
                    out.append((x + w / 2, y + h - 0.5, 1.0, team))
        return out
    return detect
