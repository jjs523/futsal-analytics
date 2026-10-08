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


def align(a: np.ndarray, b: np.ndarray, sample_rate: int, window_s: float = 120.0, max_lag_s: float = 30.0,
          hint_s: float = 0.0) -> dict:
    """Clock mapping between two recordings of the same match:  t_a = t_b + offset + drift * t_b.
    The offset is measured in a window at the start and another at the end (claps / whistles there help);
    their difference is the clock drift between the two phones (typically tens of ppm, ~0.1 s per hour).
    `hint_s` is a rough offset (e.g. from the recording start times in the file names) when the phones were
    started minutes apart: the start windows are shifted by it and only +-`max_lag_s` around it is searched."""
    sr, w = sample_rate, int(window_s * sample_rate)
    sa, sb = max(hint_s, 0.0), max(-hint_s, 0.0)                 # where the shared start window begins in a / b
    ia, ib = int(sa * sr), int(sb * sr)
    local, c_start = estimate_offset(a[ia:ia + w], b[ib:ib + w], sr, max_lag_s)
    o_start = local + sa - sb
    res = {"offset": o_start, "drift": 0.0, "offset_start": o_start, "conf_start": c_start,
           "offset_end": None, "conf_end": None}
    if len(b) < 3 * w or len(a) < 3 * w:          # too short to see drift
        return res
    tb = min(len(b) / sr, len(a) / sr - o_start) - window_s     # end window inside both recordings
    ta = min(max(tb + o_start, 0.0), len(a) / sr - window_s)
    local, c_end = estimate_offset(a[int(ta * sr):int(ta * sr) + w], b[int(tb * sr):int(tb * sr) + w], sr, max_lag_s)
    o_end = local + ta - tb
    centre_start, centre_end = sb + window_s / 2, tb + window_s / 2
    drift = (o_end - o_start) / (centre_end - centre_start)
    res.update(offset=o_start - drift * centre_start, drift=drift, offset_end=o_end, conf_end=c_end)
    return res


def align_segments(a: np.ndarray, b: np.ndarray, sample_rate: int, hint_s: float = 0.0, max_lag_s: float = 5.0,
                   seg_s: float = 60.0, fine_lag_s: float = 1.0, min_conf: float = 3.0) -> dict:
    """Robust version of `align` for recordings without claps: the offset is measured in every `seg_s` window of
    the overlap (searching +-`max_lag_s` around `hint_s`), and the median over windows is taken.
    One window alone can lock onto a wrong, similar-looking sound, and with the phones ~45 m apart each sound
    reaches them up to +-0.13 s apart depending on where it was made; the median over the whole match cancels both.
    A second pass searches only +-`fine_lag_s` around the first answer and fits offset + drift through the
    windows that agree (least squares after dropping outliers; a median alone would snap to the 16 ms hop grid).
    Same result keys as `align`, plus `n_used`, `n_windows` and `stderr`."""
    sr = sample_rate
    dur_a, dur_b = len(a) / sr, len(b) / sr

    def measure(centre_of, lag):
        rows = []
        t_b = max(0.0, -hint_s) + lag
        while t_b + seg_s + lag <= dur_b:
            guess = centre_of(t_b)                               # offset expected for this window
            t_a = t_b + guess
            if t_a - lag < 0 or t_a + seg_s + lag > dur_a:
                t_b += seg_s
                continue
            ia, ib = int((t_a - lag) * sr), int(t_b * sr)        # a: window with `lag` margin on both sides
            local, conf = estimate_offset(a[ia:ia + int((seg_s + 2 * lag) * sr)], b[ib:ib + int(seg_s * sr)], sr, 2 * lag)
            rows.append((t_b + seg_s / 2, local + (t_a - lag) - t_b, conf))
            t_b += seg_s
        return np.array(rows).reshape(-1, 3)

    first = measure(lambda t: hint_s, max_lag_s)
    good = first[first[:, 2] >= min_conf] if len(first) else first
    if len(good) == 0:
        good = first
    if len(good) == 0:
        raise ValueError("the two recordings do not overlap (check hint_s)")
    o1 = float(np.median(good[:, 1]))
    second = measure(lambda t: o1, fine_lag_s)
    ok = second[(second[:, 2] >= min_conf) & (np.abs(second[:, 1] - o1) < 0.5)] if len(second) else second
    if len(ok) < 3:
        return {"offset": o1, "drift": 0.0, "offset_start": o1, "conf_start": float(np.median(good[:, 2])),
                "offset_end": None, "conf_end": None, "n_used": int(len(ok)), "n_windows": int(len(second)), "stderr": None}
    t, o = ok[:, 0], ok[:, 1]
    keep = np.abs(o - np.median(o)) < 3 * 1.4826 * np.median(np.abs(o - np.median(o))) + 1e-3   # drop outliers (MAD)
    t, o = t[keep], o[keep]
    drift, offset = 0.0, float(np.mean(o))
    if len(t) >= 6 and t.max() - t.min() > 600:     # straight line through the windows, kept only if the drift is
        (d, c), cov = np.polyfit(t, o, 1, cov=True)  # clearly measurable (otherwise its noise would shift the offset)
        if abs(d) > 2.5 * np.sqrt(cov[0, 0]):
            drift, offset = float(d), float(c)
    resid = o - (offset + drift * t)
    stderr = float(np.std(resid) / np.sqrt(len(resid)))
    return {"offset": offset, "drift": drift,
            "offset_start": offset + drift * float(t.min()), "conf_start": float(np.median(ok[:, 2])),
            "offset_end": offset + drift * float(t.max()), "conf_end": float(np.median(ok[:, 2])),
            "n_used": int(len(t)), "n_windows": int(len(second)), "stderr": stderr}


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


def filename_time(video_path: str):
    """Recording start time from phone file names like 20261008_170919.mp4 (Samsung) or
    VID_20261008_170919.mp4 / PXL_20261008_170919123.mp4 (other Android); None if the name has no time."""
    import datetime as dt
    import os
    import re
    m = re.search(r"(20\d{6})_(\d{6})", os.path.basename(video_path))
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def start_hint(ref: str, other: str) -> float | None:
    """Rough offset (s) from the file names: how much later `other` started than `ref`."""
    ta, tb = filename_time(ref), filename_time(other)
    return None if ta is None or tb is None else (tb - ta).total_seconds()


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
    import os
    ap = argparse.ArgumentParser(prog="python -m futsal.sync", description="두 폰 영상의 시간 차이를 소리로 계산")
    ap.add_argument("ref", help="기준 영상 (cam1)")
    ap.add_argument("other", help="맞출 영상 (cam2)")
    ap.add_argument("--hint", type=float, default=None,
                    help="대략적인 시작 차이(초, other가 늦게 시작하면 +). 생략하면 파일 이름의 녹화 시각으로 계산")
    ap.add_argument("--max-lag", type=float, default=None,
                    help="대략적인 차이에서 더 찾아볼 범위(초). 기본: 파일 이름으로 잡았으면 5, 아니면 30")
    ap.add_argument("--segment", type=float, default=60.0, help="한 번에 비교할 구간 길이(초)")
    ap.add_argument("--method", choices=["segments", "ends"], default="segments",
                    help="segments: 경기 전체를 1분씩 재서 중앙값 (손뼉 없어도 됨, 기본) / ends: 시작·끝 2분만 (손뼉 필요)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sr = 16000
    from_name = start_hint(a.ref, a.other)
    hint = a.hint if a.hint is not None else (from_name or 0.0)
    lag = a.max_lag if a.max_lag is not None else (5.0 if (a.hint is not None or from_name is not None) else 30.0)
    print(f"대략적인 시작 차이 {hint:+.0f} s 주변 ±{lag:.0f} s에서 찾습니다 (소리 읽는 중, 긴 영상은 몇 분 걸림)")
    xa, xb = read_audio(a.ref, sr), read_audio(a.other, sr)
    if a.method == "ends":
        res = align(xa, xb, sr, 120.0, lag, hint)
    else:
        res = align_segments(xa, xb, sr, hint, lag, a.segment)
    res["hint"] = hint
    res["creation_time"] = {"ref": creation_time(a.ref), "other": creation_time(a.other)}
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    ra, rb = os.path.basename(a.ref), os.path.basename(a.other)
    if "n_used" in res:
        se = f", 표준오차 {res['stderr']:.3f} s" if res["stderr"] is not None else ""
        print(f"구간 {res['n_windows']}개 중 {res['n_used']}개가 일치{se}")
    else:
        print(f"시작 구간 시간 차: {res['offset_start']:+.3f} s  (신뢰도 {res['conf_start']:.1f})")
        if res["offset_end"] is not None:
            print(f"끝 구간 시간 차:   {res['offset_end']:+.3f} s  (신뢰도 {res['conf_end']:.1f})")
    print(f"시계 속도 차이:    {res['drift'] * 1e6:+.0f} ppm" + ("  (측정 오차 안이라 0으로 둠)" if "n_used" in res and res["drift"] == 0 else ""))
    print(f"=> {ra} 시각 = {rb} 시각 {res['offset']:+.3f} s {res['drift']:+.2e} x ({rb} 시각)")
    print(f"   분석에 넣을 값: {{\"offset\": {res['offset']:.3f}, \"drift\": {res['drift']:.3e}}}")
    if "n_used" in res:
        if res["n_used"] < 5 or (res["stderr"] or 1) > 0.05:
            print("주의: 일치하는 구간이 적거나 흩어져 있습니다. python -m futsal.syncview 로 장면을 보고 확인하세요.")
    elif min(c for c in (res["conf_start"], res["conf_end"]) if c is not None) < 6:
        print("주의: 신뢰도가 낮습니다 (6 미만). 손뼉 장면이나 시계 화면으로 직접 확인하세요.")
    print(f"녹화 시각(메타데이터, 삼성은 종료 시각): {res['creation_time']}")


if __name__ == "__main__":
    main()
