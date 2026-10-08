"""원본 해상도 조각 검출과 공 궤적 정리.

공은 한 프레임에 하나뿐이고(연습 공 제외), 날아가면 선수보다 훨씬 빠르고(초속 20~30m), 자주 가려집니다.
그래서 프레임마다 후보 여러 개 중 '궤적과 이어지는 하나'를 고르고, 짧게 놓친 구간은 앞뒤로 채웁니다.
좌표는 픽셀 기준입니다: 공은 공중에 뜨므로 바닥 호모그래피로 미터 변환하면 뜬 공의 위치가 틀어집니다
(두 카메라로 높이까지 계산하는 것은 다음 단계).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from .dataset import tile_origins

# 조각 검출기: BGR 조각 이미지 → [(x1, y1, x2, y2, 확신도), ...]  (조각 안 좌표)
TileDetector = Callable[[np.ndarray], list[tuple[float, float, float, float, float]]]


def detect_tiled(frame: np.ndarray, detector: TileDetector, tile: int = 640, overlap: float = 0.2,
                 nms_iou: float = 0.3) -> list[tuple[float, float, float, float, float]]:
    """원본 해상도 프레임을 조각으로 나눠 검출하고 겹친 결과를 NMS로 합칩니다.
    조각 경계에 잘린 공은 이웃 조각에서 온전히 보이므로, NMS에서 잘린 쪽이 지도록 순위를 낮춥니다."""
    h, w = frame.shape[:2]
    boxes, scores, rank = [], [], []
    for x0, y0 in tile_origins(w, h, tile, overlap):
        crop = frame[y0:y0 + tile, x0:x0 + tile]
        th, tw = crop.shape[:2]
        for x1, y1, x2, y2, c in detector(crop):
            cut = ((x1 <= 1 and x0 > 0) or (y1 <= 1 and y0 > 0)
                   or (x2 >= tw - 1 and x0 + tw < w) or (y2 >= th - 1 and y0 + th < h))
            boxes.append([x1 + x0, y1 + y0, x2 - x1, y2 - y1]); scores.append(float(c))
            rank.append(float(c) * (0.5 if cut else 1.0))
    if not boxes:
        return []
    keep = np.array(cv2.dnn.NMSBoxes(boxes, rank, 0.0, nms_iou)).reshape(-1)
    return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], scores[i]) for i in keep]


def yolo_tile_detector(model: str, conf: float = 0.15, imgsz: int = 640) -> TileDetector:
    from ultralytics import YOLO      # 선택 의존성: pip install '.[vision]'
    net = YOLO(model)
    # 공 전용 모델은 'ball' 하나, COCO 사전학습 모델(비교 기준선)은 'sports ball'(32)만 남김
    ball_ids = [k for k, v in net.names.items() if v in ("ball", "sports ball")] or None

    def detect(img):
        r = net.predict(img, conf=conf, imgsz=imgsz, classes=ball_ids, verbose=False)[0]
        return [(*map(float, b), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return detect


@dataclass
class BallPoint:
    frame: int
    t: float
    u: float | None          # 공 중심 픽셀 (없으면 None)
    v: float | None
    conf: float
    source: str              # "detected" | "interpolated" | "missing"


def link(candidates: list[list[tuple[float, float, float]]], fps: float, max_speed_px: float = 60.0,
         min_conf: float = 0.2, max_gap: int = 8) -> list[tuple[float, float, float, str] | None]:
    """프레임별 후보 [(u, v, 확신도)] → 프레임별 공 위치 하나.
    1) 확신도 높은 후보에서 시작해 앞뒤로 '한 프레임에 max_speed_px 이내로 움직인' 후보를 이어 궤적 조각을 만들고
    (방향을 아는 뒤로는 등속 예측 근처만 받음), 2) 확신도 합이 큰 조각부터 빈 프레임을 채우고, 3) `max_gap` 프레임 이하 빈칸은 직선으로 채움."""
    n = len(candidates)
    used = [set() for _ in range(n)]
    pieces = []
    order = sorted(((c[2], i, j) for i, cs in enumerate(candidates) for j, c in enumerate(cs) if c[2] >= min_conf), reverse=True)
    for _, i0, j0 in order:
        if j0 in used[i0]:
            continue
        piece = {i0: candidates[i0][j0]}; used[i0].add(j0)
        for direction in (1, -1):
            last_i, last, vel = i0, candidates[i0][j0], None      # vel: 프레임당 픽셀 이동 (등속 예측)
            while True:
                nxt = _next_point(candidates, used, last_i, last, vel, direction, max_speed_px, min_conf, max_gap)
                if nxt is None:
                    break
                i, j = nxt
                c, k = candidates[i][j], abs(i - last_i)
                vel = ((c[0] - last[0]) / k, (c[1] - last[1]) / k)
                piece[i] = c; used[i].add(j); last_i, last = i, c
        pieces.append(piece)
    pieces.sort(key=lambda p: -sum(c[2] for c in p.values()))
    out: list = [None] * n
    reserved = [False] * n                    # 채택된 조각 안의 빈칸: 그 조각의 보간으로 채울 자리
    for p in pieces:
        if len(p) < 3 and max(c[2] for c in p.values()) < 0.5:      # 짧고 약한 조각은 잡음
            continue
        frames = sorted(i for i in p if out[i] is None and not reserved[i])   # 더 강한 조각이 차지한 곳은 건너뜀
        for i in frames:
            out[i] = (p[i][0], p[i][1], p[i][2], "detected")
        for a, b in zip(frames, frames[1:]):
            for i in range(a + 1, b):
                reserved[i] = True
    idx = [i for i in range(n) if out[i] is not None]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for i in range(a + 1, b):
                r = (i - a) / (b - a)
                out[i] = (out[a][0] + r * (out[b][0] - out[a][0]), out[a][1] + r * (out[b][1] - out[a][1]), 0.0, "interpolated")
    return out


def _next_point(candidates, used, last_i, last, vel, direction, max_speed_px, min_conf, max_gap):
    """궤적 조각의 다음 점 (프레임, 후보 번호). 방향을 모르면 가장 가까운 프레임의 가장 가까운 후보,
    방향을 알면 앞으로 `max_gap` 프레임 안에서 등속 예측에 가장 잘 맞는 후보를 고릅니다
    (공이 가려진 동안 그 자리를 지나간 머리·라인 오검출을 집지 않도록)."""
    best, best_cost = None, np.inf
    for k in range(1, max_gap + 1):
        i = last_i + direction * k
        if not 0 <= i < len(candidates):
            break
        if vel is None and best is not None:
            break
        pred = (last[0], last[1]) if vel is None else (last[0] + vel[0] * k, last[1] + vel[1] * k)
        gate = max_speed_px * k if vel is None else max_speed_px * (0.25 + 0.25 * k)
        for j, c in enumerate(candidates[i]):
            if j in used[i] or c[2] < min_conf or np.hypot(c[0] - last[0], c[1] - last[1]) > max_speed_px * k:
                continue
            d = np.hypot(c[0] - pred[0], c[1] - pred[1])
            if d > gate:                        # 예측에서 너무 벗어나면 다른 물체
                continue
            cost = d / gate + 0.05 * (k - 1)    # 예측에 잘 맞을수록, 덜 건너뛸수록 좋음
            if cost < best_cost:
                best, best_cost = (i, j), cost
    return best


def track_video(path: str, detector: TileDetector, stride: int = 1, tile: int = 640, max_speed_px: float = 60.0) -> list[BallPoint]:
    """영상 전체에서 공 찾기 → 궤적 정리. 공은 빨라서 선수보다 촘촘하게(stride 1~2) 보는 것을 권장합니다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cands, frames, times, i = [], [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            times.append(ms / 1000.0 if ms > 0 or i == 0 else i / fps)
            frames.append(i)
            cands.append([((x1 + x2) / 2, (y1 + y2) / 2, c) for x1, y1, x2, y2, c in detect_tiled(frame, detector, tile)])
        i += 1
    cap.release()
    linked = link(cands, fps / stride, max_speed_px * stride)
    return [BallPoint(f, t, *(p[:2] if p else (None, None)), p[2] if p else 0.0, p[3] if p else "missing")
            for f, t, p in zip(frames, times, linked)]


def save(points: list[BallPoint], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in points], f, separators=(",", ":"))
