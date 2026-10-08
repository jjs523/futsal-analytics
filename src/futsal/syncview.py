"""시간 맞추기 눈으로 확인: python -m futsal.syncview cam1.mp4 cam2.mp4 --offset 124.29 --drift 0 --out sync_check

구한 시간 차(t_cam1 = t_cam2 + offset + drift x t_cam2)로 두 영상의 같은 순간을 찾아, 위(cam1)·아래(cam2)로
몇 프레임씩 나란히 붙인 이미지를 만듭니다. 아래에는 그 순간 앞뒤 1초의 소리 변화(공 차는 소리 등)를 겹쳐 그려서,
두 선의 뾰족한 부분이 같은 자리에 오는지도 볼 수 있습니다.

--at 을 주지 않으면 겹치는 구간의 시작·25%·50%·75%·끝에서 소리가 가장 뚜렷한 순간(공 차는 소리 등)을 고릅니다.
--offsets 로 후보 값을 여러 개 주면 같은 순간을 후보마다 그려서 어느 값이 맞는지 비교할 수 있습니다.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from . import sync

_SR = 16000


def to_other(t_ref: float, offset: float, drift: float) -> float:
    """cam1 시각 → cam2 시각 (t_ref = t_other + offset + drift * t_other 의 역)."""
    return (t_ref - offset) / (1 + drift)


def frames_at(path: str, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """각 시각에 가장 가까운 프레임과 그 프레임의 실제 시각. 앞쪽으로 이동한 뒤 차례로 읽어 시각을 정확히 맞춥니다."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(min(times) - 1.0, 0) * 1000)
    best: list[tuple[float, float, np.ndarray | None]] = [(np.inf, 0.0, None) for _ in times]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        for i, want in enumerate(times):
            if abs(t - want) < best[i][0]:
                best[i] = (abs(t - want), t, frame)
        if t > max(times) + 0.1:
            break
    cap.release()
    return [(t, f) for _, t, f in best]


def loudest_onset(audio: np.ndarray, t0: float, t1: float) -> float:
    """t0~t1 사이에서 소리가 가장 갑자기 커지는 시각 (공 차는 소리, 휘슬, 손뼉)."""
    env = sync._envelope(audio[int(t0 * _SR):int(t1 * _SR)], _SR)
    return t0 + int(np.argmax(env)) * sync._HOP / _SR


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _audio_panel(a: np.ndarray, b: np.ndarray, t_ref: float, offset: float, drift: float,
                 width: int, height: int = 140, span: float = 1.0) -> np.ndarray:
    """cam1(주황)·cam2(파랑) 소리 변화를 cam1 시간 축에 겹쳐 그림. 가운데 흰 선이 고른 순간."""
    panel = np.full((height, width, 3), 30, np.uint8)
    n = int(2 * span * _SR / sync._HOP)
    for audio, t0, color in ((a, t_ref - span, (0, 140, 255)),
                             (b, to_other(t_ref, offset, drift) - span, (255, 160, 0))):
        seg = audio[max(int(t0 * _SR), 0):max(int(t0 * _SR), 0) + n * sync._HOP]
        if len(seg) < n * sync._HOP:
            continue
        env = sync._envelope(seg, _SR)
        env = np.clip(env / (env.max() + 1e-9), 0, 1)
        xs = np.linspace(0, width - 1, len(env)).astype(np.int32)
        ys = (height - 10 - env * (height - 30)).astype(np.int32)
        cv2.polylines(panel, [np.stack([xs, ys], 1)], False, color, 2)
    cv2.line(panel, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)
    for k in range(-int(span * 10), int(span * 10) + 1):           # 0.1초 눈금
        x = int(width / 2 + k / (2 * span) * width)
        cv2.line(panel, (x, height - 6), (x, height), (180, 180, 180), 1)
    cv2.putText(panel, "sound: cam1 orange / cam2 blue, ticks 0.1s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return panel


def sheet(ref: str, other: str, t_ref: float, offset: float, drift: float, a: np.ndarray, b: np.ndarray,
          steps: int = 2, step_s: float = 1 / 30, tile_w: int = 480) -> np.ndarray:
    """한 순간: 위 줄 cam1, 아래 줄 cam2, 가운데 칸이 고른 순간, 양옆으로 step_s 간격 프레임."""
    ks = list(range(-steps, steps + 1))
    ref_times = [t_ref + k * step_s for k in ks]
    rows = []
    for path, times, name in ((ref, ref_times, "cam1"),
                              (other, [to_other(t, offset, drift) for t in ref_times], "cam2")):
        tiles = []
        for k, (t, f) in zip(ks, frames_at(path, times)):
            tile = np.zeros((int(tile_w * 9 / 16), tile_w, 3), np.uint8) if f is None else \
                cv2.resize(f, (tile_w, int(f.shape[0] * tile_w / f.shape[1])))
            _label(tile, f"{name} {t:.3f}s ({k:+d})")
            if k == 0:
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 255, 255), 3)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    width = rows[0].shape[1]
    title = np.full((40, width, 3), 0, np.uint8)
    cv2.putText(title, f"cam1 {int(t_ref // 60)}:{t_ref % 60:06.3f}  offset {offset:+.3f}s  drift {drift * 1e6:+.1f}ppm",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([title, rows[0], rows[1], _audio_panel(a, b, t_ref, offset, drift, width)])


def auto_moments(a: np.ndarray, b: np.ndarray, offset: float, drift: float, n: int = 5, search: float = 15.0) -> list[float]:
    """겹치는 구간에 고르게 n곳, 각 자리 ±search초 안에서 소리가 가장 뚜렷한 순간 (cam1 시각)."""
    start = max(offset, 0.0) + 5
    end = min(len(a) / _SR, len(b) / _SR * (1 + drift) + offset) - 5
    out = []
    for k in range(n):
        c = start + search + (end - start - 2 * search) * k / max(n - 1, 1)
        out.append(round(loudest_onset(a, c - search, c + search), 3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.syncview", description="시간 맞추기 결과를 영상으로 확인")
    ap.add_argument("ref", help="기준 영상 (cam1)")
    ap.add_argument("other", help="맞출 영상 (cam2)")
    ap.add_argument("--offset", type=float, nargs="+", required=True, help="시간 차(초). 여러 개면 후보끼리 비교")
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--at", type=float, nargs="*", help="확인할 cam1 시각(초). 생략하면 자동으로 5곳")
    ap.add_argument("--steps", type=int, default=2, help="고른 순간 앞뒤로 몇 칸씩")
    ap.add_argument("--step", type=float, default=1 / 30, help="칸 간격(초)")
    ap.add_argument("--out", default="sync_check")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    print("소리 읽는 중...")
    aa, bb = sync.read_audio(a.ref, _SR), sync.read_audio(a.other, _SR)
    moments = a.at or auto_moments(aa, bb, a.offset[0], a.drift)
    for t in moments:
        for off in a.offset:
            img = sheet(a.ref, a.other, t, off, a.drift, aa, bb, a.steps, a.step)
            path = os.path.join(a.out, f"t{int(t):04d}s_off{off:+.3f}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  {path}")


if __name__ == "__main__":
    main()
