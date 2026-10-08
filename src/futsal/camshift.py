"""촬영 중 카메라가 움직였는지 자동으로 찾기: python -m futsal.camshift cam1.mp4

일반 사용자는 삼각대를 건드리거나, 폰을 만졌다가 다시 놓거나, 바람에 흔들리는 일이 흔합니다.
영상 전체를 0.5초 간격으로 훑어서 배경(라인·펜스·골대)이 통째로 밀린 순간을 찾고, 얼마나 밀렸는지 잽니다.
- 밝기 변화(자동 노출)는 무시합니다: 밝기 대신 윤곽(경계선)끼리 비교합니다.
- 카메라 앞을 지나가는 선수는 무시합니다: 밀린 상태가 `hold_s`초 이상 같은 값으로 유지될 때만 움직임으로 봅니다.
- 화면이 많이 돌아가거나 확대돼서 밀림만으로 설명이 안 되면 `retap`으로 표시합니다 (기준점을 다시 탭해야 함).

결과로 시간 구간별 좌표 보정(CalibrationTimeline)을 바로 만들 수 있습니다 (`timeline_from_moves`).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .homography import Calibration, CalibrationTimeline


def _edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return mag / (mag.mean() + 1e-6)


def camera_moves(path: str, step_s: float = 0.5, width: int = 640, min_px: float = 3.0, hold_s: float = 2.0,
                 min_response: float = 0.05, progress: bool = False) -> dict:
    """Times where the whole picture slid, with the slide (dx, dy) in original-resolution pixels.
    Returns {"moves": [{"t", "dx", "dy", "total_dx", "total_dy", "response", "retap"}], "samples", "duration_s"}."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps * step_s)))
    ref = win = None
    scale = 1.0
    pending: list[tuple[float, float, float, float, np.ndarray]] = []
    moves, total_dx, total_dy, i, n = [], 0.0, 0.0, 0, 0
    while cap.grab():
        if i % stride:
            i += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000 if ms > 0 or i == 0 else i / fps
        h, w = frame.shape[:2]
        scale = width / w
        cur = _edges(cv2.cvtColor(cv2.resize(frame, (width, int(round(h * scale)))), cv2.COLOR_BGR2GRAY))
        n += 1
        if ref is None:
            ref, win = cur, cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
        else:
            (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
            dx, dy = dx / scale, dy / scale
            if np.hypot(dx, dy) >= min_px or resp < min_response:
                pending.append((t, dx, dy, resp, cur))
            else:
                pending.clear()
            if pending and pending[-1][0] - pending[0][0] >= hold_s:
                d = np.array([(p[1], p[2]) for p in pending])
                if np.all(np.abs(d - np.median(d, axis=0)) < max(2.0, 0.1 * np.hypot(*np.median(d, axis=0)))):
                    mdx, mdy = (float(v) for v in np.median(d, axis=0))
                    resp_med = float(np.median([p[3] for p in pending]))
                    total_dx += mdx; total_dy += mdy
                    moves.append({"t": round(pending[0][0], 2), "dx": round(mdx, 1), "dy": round(mdy, 1),
                                  "total_dx": round(total_dx, 1), "total_dy": round(total_dy, 1),
                                  "response": round(resp_med, 3), "retap": resp_med < min_response})
                    ref = pending[-1][4]                 # the new resting position is the new reference
                    pending.clear()
                else:
                    pending.pop(0)                       # not settled yet (still moving, or a player in front)
        if progress and total and n % 600 == 0:
            print(f"    {i / total * 100:.0f}%")
        i += 1
    cap.release()
    return {"moves": moves, "samples": n, "duration_s": round(i / fps, 1)}


def timeline_from_moves(base: Calibration, moves: list[dict], tapped_at_s: float = 0.0) -> CalibrationTimeline:
    """Calibration tapped on the frame at `tapped_at_s` + detected moves -> one calibration per resting position."""
    def total_at(t):
        tot = (0.0, 0.0)
        for m in moves:
            if m["t"] <= t:
                tot = (m["total_dx"], m["total_dy"])
        return tot
    bx, by = total_at(tapped_at_s)
    starts = [(0.0, (0.0, 0.0))] + [(m["t"], (m["total_dx"], m["total_dy"])) for m in moves]
    return [(t, base.shifted(x - bx, y - by)) for t, (x, y) in starts]


def describe(result: dict) -> list[str]:
    lines = []
    for m in result["moves"]:
        mm, ss = divmod(int(m["t"]), 60)
        note = " → 화면이 돌거나 확대됨, 이 시점 이후로 기준점 다시 탭" if m["retap"] else " → 자동 보정 가능"
        lines.append(f"{mm}:{ss:02d} 카메라 움직임: 가로 {m['dx']:+.0f}px, 세로 {m['dy']:+.0f}px{note}")
    return lines or ["카메라 움직임 없음"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.camshift", description="촬영 중 카메라가 움직인 순간 찾기")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--step", type=float, default=0.5, help="몇 초마다 볼지")
    ap.add_argument("--min-px", type=float, default=3.0, help="이보다 작게 밀린 건 무시 (원본 픽셀)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = {}
    for v in a.videos:
        print(f"{v} 훑는 중... (영상 길이에 따라 몇 분)")
        out[v] = camera_moves(v, a.step, min_px=a.min_px, progress=True)
        for line in describe(out[v]):
            print("  " + line)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
