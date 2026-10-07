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
    return float(lags[i] / hop_rate), conf


_HOP = 256


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Positive changes of short-time energy: sharp sounds stand out, steady crowd noise does not."""
    x = np.asarray(x, float)
    n = len(x) // _HOP
    e = np.sqrt(np.mean(x[: n * _HOP].reshape(n, _HOP) ** 2, axis=1) + 1e-12)
    d = np.maximum(np.diff(np.log(e), prepend=np.log(e[0])), 0)
    return (d - d.mean()) / (d.std() + 1e-9)


def read_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 audio from a video file via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video_path, "-ac", "1", "-ar", str(sample_rate),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)
