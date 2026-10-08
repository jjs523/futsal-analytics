"""Time-aligning the two phones' videos from their audio tracks (whistles, ball strikes, voices)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np


def estimate_offset(a: np.ndarray, b: np.ndarray, sample_rate: int, max_lag_s: float = 30.0) -> tuple[float, float]:
    """Seconds to ADD to b's timestamps to align it with a (positive: b started recording later),
    plus a peak-to-sidelobe confidence score. Uses the FFT cross-correlation of the onset envelopes."""
    ea, eb = _envelope(a, sample_rate), _envelope(b, sample_rate)
    hop_rate = sample_rate / _HOP
    n = len(ea) + len(eb)
    size = 1 << (n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(ea, size) * np.conj(np.fft.rfft(eb, size)), size)
    lags = np.r_[np.arange(0, len(ea)), np.arange(-len(eb) + 1, 0)]
    vals = np.r_[xc[:len(ea)], xc[size - len(eb) + 1:]]
    keep = np.abs(lags) <= max_lag_s * hop_rate
    lags, vals = lags[keep], vals[keep]
    i = int(np.argmax(vals))
    peak = vals[i]
    side = np.delete(vals, slice(max(i - 5, 0), i + 6))
    conf = float((peak - side.mean()) / (side.std() + 1e-9)) if len(side) else 0.0
    frac = 0.0
    if 0 < i < len(vals) - 1 and lags[i + 1] - lags[i - 1] == 2:      # parabolic peak -> sub-hop precision
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    return float((lags[i] + frac) / hop_rate), conf


def align(a: np.ndarray, b: np.ndarray, sample_rate: int, window_s: float = 120.0, max_lag_s: float = 30.0) -> dict:
    """Clock mapping between two recordings of the same match:  t_a = t_b + offset + drift * t_b.
    The offset is measured in a window at the start and another at the end (claps / whistles there help);
    their difference is the clock drift between the two phones (typically tens of ppm, ~0.1 s per hour)."""
    sr, w = sample_rate, int(window_s * sample_rate)
    o_start, c_start = estimate_offset(a[:w], b[:w], sr, max_lag_s)
    res = {"offset": o_start, "drift": 0.0, "offset_start": o_start, "conf_start": c_start,
           "offset_end": None, "conf_end": None}
    if len(b) < 3 * w or len(a) < 3 * w:          # too short to see drift
        return res
    tb = len(b) / sr - window_s                    # end window of b, and where it should sit in a
    ta = min(max(tb + o_start, 0.0), len(a) / sr - window_s)
    local, c_end = estimate_offset(a[int(ta * sr):int(ta * sr) + w], b[int(tb * sr):int(tb * sr) + w], sr, max_lag_s)
    o_end = local + ta - tb
    centre_start, centre_end = window_s / 2, tb + window_s / 2
    drift = (o_end - o_start) / (centre_end - centre_start)
    res.update(offset=o_start - drift * centre_start, drift=drift, offset_end=o_end, conf_end=c_end)
    return res


_HOP = 256


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Positive changes of short-time energy: sharp sounds stand out, steady crowd noise does not."""
    x = np.asarray(x, float)
    n = len(x) // _HOP
    e = np.sqrt(np.mean(x[: n * _HOP].reshape(n, _HOP) ** 2, axis=1) + 1e-12)
    d = np.maximum(np.diff(np.log(e), prepend=np.log(e[0])), 0)
    return (d - d.mean()) / (d.std() + 1e-9)


def creation_time(video_path: str) -> str | None:
    """Recording start time stored by the phone (second precision; a sanity check, not for frame sync)."""
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time,com.apple.quicktime.creationdate",
                          "-of", "default=nw=1", video_path], capture_output=True, text=True).stdout
    return out.strip() or None


def read_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 audio from a video file via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video_path, "-ac", "1", "-ar", str(sample_rate),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def main(argv=None):
    """python -m futsal.sync cam1.mp4 cam2.mp4  ->  cam2 시각을 cam1 시각으로 바꾸는 식"""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m futsal.sync", description="두 폰 영상의 시간 차이를 소리로 계산")
    ap.add_argument("ref", help="기준 영상 (cam1)")
    ap.add_argument("other", help="맞출 영상 (cam2)")
    ap.add_argument("--window", type=float, default=120.0, help="시작·끝에서 비교할 구간 길이(초)")
    ap.add_argument("--max-lag", type=float, default=30.0, help="두 영상 시작 차이의 최대값(초)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sr = 16000
    res = align(read_audio(a.ref, sr), read_audio(a.other, sr), sr, a.window, a.max_lag)
    res["creation_time"] = {"ref": creation_time(a.ref), "other": creation_time(a.other)}
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    print(f"시작 구간 시간 차: {res['offset_start']:+.3f} s  (신뢰도 {res['conf_start']:.1f})")
    if res["offset_end"] is not None:
        print(f"끝 구간 시간 차:   {res['offset_end']:+.3f} s  (신뢰도 {res['conf_end']:.1f})")
        print(f"시계 속도 차이:    {res['drift'] * 1e6:+.0f} ppm")
    import os
    ra, rb = os.path.basename(a.ref), os.path.basename(a.other)
    print(f"=> {ra} 시각 = {rb} 시각 {res['offset']:+.3f} s {res['drift']:+.2e} x ({rb} 시각)")
    print(f"   분석에 넣을 값: {{\"offset\": {res['offset']:.3f}, \"drift\": {res['drift']:.3e}}}")
    if min(c for c in (res["conf_start"], res["conf_end"]) if c is not None) < 6:
        print("주의: 신뢰도가 낮습니다 (6 미만). 손뼉 장면이나 시계 화면으로 직접 확인하세요.")
    print(f"녹화 시작 시각(메타데이터): {res['creation_time']}")


if __name__ == "__main__":
    main()
