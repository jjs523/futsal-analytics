"""align.estimate_clock / apply_clock: recover a known clock offset + drift between two cameras from the players."""
import numpy as np
import pytest

from futsal.court import Court
from futsal.homography import Calibration
from futsal.pipeline import align
from futsal.pipeline.boxes import CamBoxes, _index_by_k

RATE = 10.0
COURT = Court()


def path(p: int, t: np.ndarray) -> np.ndarray:
    """Player p's pitch position at time t (s): smooth loops at ~2-4 m/s, players kept apart."""
    cx, cy = 6 + 6 * (p % 5), 5 + 10 * (p // 5)
    w = 0.25 + 0.03 * p
    return np.stack([cx + 4 * np.sin(w * t + p), cy + 3 * np.cos(0.8 * w * t + 2 * p)], 1)


def camera(name: str, n: int, players: int, emb: np.ndarray, shift=lambda t: 0.0 * t) -> CamBoxes:
    """Boxes of `players` people on grid frames 0..n-1; the position seen at grid frame k is the one at true time
    k / RATE + shift(k / RATE), i.e. this camera's clock is off by shift."""
    ks = np.repeat(np.arange(n), players)
    pid = np.tile(np.arange(players), n)
    t_true = ks / RATE + shift(ks / RATE)
    xy = np.concatenate([path(p, t_true[pid == p])[None] for p in range(players)], 0)       # (players, n, 2)
    xy = xy.transpose(1, 0, 2).reshape(-1, 2)
    m = len(ks)
    xyxy = np.stack([np.full(m, 100.0), np.full(m, 100.0), np.full(m, 140.0), np.full(m, 220.0)], 1)
    return CamBoxes(name=name, fid=ks.copy(), t=ks / RATE, k=ks.copy(), xyxy=xyxy, conf=np.full(m, 0.9),
                    reid=emb[pid].astype(np.float32), color=None, foot=np.zeros((m, 2)), h=np.full(m, 120.0),
                    w=np.full(m, 40.0), xy=xy.copy(), xy_raw=xy.copy(), sigma=np.full(m, 0.1),
                    in_court=COURT.contains(xy, 1.0), team=np.full(m, "", dtype="<U1"),
                    cal=Calibration(np.eye(3), 0.0), frame_wh=(1920, 1080), by_k=_index_by_k(ks))


@pytest.mark.parametrize("off,drift", [(-0.12, -400e-6), (0.12, 400e-6)])
def test_estimate_and_apply_clock_recovers_offset_and_drift(off, drift):
    """Either sign: a negative drift stretches camera 2's grid, a positive one compresses it (surplus frames are left
    off the grid; pushing them on would cancel the drift)."""
    rng = np.random.default_rng(0)
    players, n = 8, 9000                                    # 15 minutes at 10 Hz
    emb = rng.normal(size=(players, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    a = camera("cam1", n, players, emb)
    b = camera("cam2", n, players, emb, shift=lambda t: -(off + drift * t))
    clock = align.estimate_clock(a, b, RATE, window_s=180.0, max_shift_s=1.0, step=10)
    assert clock is not None and len(clock["shift_s"]) == 5
    assert abs(clock["offset_s"] - off) < 0.03 and abs(clock["drift"] - drift) < 60e-6
    before = align._shift_residual(a, b, list(range(0, n, 10)), 0, 0.5, 2.5, 0.7)[0]
    align.apply_clock(b, clock, RATE)
    after = align._shift_residual(a, b, list(range(0, n, 10)), 0, 0.5, 2.5, 0.7)[0]
    assert after < 0.25 * before
    on = b.k >= 0
    assert len(set(b.k[on].tolist())) == len(set(b.fid[on].tolist()))                 # one source frame per grid frame
    assert (~on).sum() <= (5 * players if drift > 0 else 0)
    again = align.estimate_clock(a, b, RATE, window_s=180.0, max_shift_s=1.0, step=10)
    assert abs(again["offset_s"]) < 0.03 and abs(again["drift"]) < 60e-6


def test_estimate_clock_needs_reid_and_enough_windows():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(4, 8)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    a, b = camera("cam1", 300, 4, emb), camera("cam2", 300, 4, emb)
    assert align.estimate_clock(a, b, RATE, window_s=10.0) is not None      # 3 windows of 10 s
    b.reid = None
    assert align.estimate_clock(a, b, RATE) is None
    a2, b2 = camera("cam1", 300, 4, emb), camera("cam2", 300, 4, emb)
    assert align.estimate_clock(a2, b2, RATE, window_s=300.0) is None      # one window: no line through one point


def test_refine_clock_applies_only_over_a_long_enough_span():
    rng = np.random.default_rng(2)
    players, n = 8, 9000
    emb = rng.normal(size=(players, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    shift = lambda t: -(-0.12 - 400e-6 * t)
    a, b = camera("cam1", n, players, emb), camera("cam2", n, players, emb, shift=shift)
    k0 = b.k.copy()
    short = align.refine_clock({"cam1": a, "cam2": b}, RATE, min_span_s=1000.0, window_s=180.0, max_shift_s=1.0, step=10)
    assert short["camera"] == "cam2" and not short["applied"] and np.array_equal(b.k, k0)     # 720 s of shifts < 1000
    done = align.refine_clock({"cam1": a, "cam2": b}, RATE, window_s=180.0, max_shift_s=1.0, step=10)
    assert done["applied"] and not np.array_equal(b.k, k0) and np.array_equal(a.k, np.repeat(np.arange(n), players))
    assert align.refine_clock({"cam1": a}, RATE) is None


@pytest.mark.parametrize("drift_c", [-400e-6, 400e-6])
def test_corrected_sync_reads_the_corrected_camera_at_its_own_frames(drift_c):
    """A camera synced with (offset, drift) on a grid starting at `start`, then corrected: corrected_sync gives the
    (offset, drift) at which trackview.render finds, for each grid frame, the camera frame whose boxes are there."""
    emb = np.ones((1, 4)) / 2.0
    off, drift, start = 124.27, 50e-6, 125.0
    cam = camera("cam2", 30000, 1, emb)
    t = np.arange(30000) / RATE + 5.0                                    # the camera's own clock (s)
    cam.t, cam.fid = t, np.arange(30000)
    cam.k = np.round((t + off + drift * t - start) * RATE).astype(int)
    clock = {"offset_s": -0.12, "drift": drift_c}
    align.apply_clock(cam, clock, RATE)
    o2, d2 = align.corrected_sync(off, drift, clock, start)
    on = cam.k >= 0
    t_hat = (start + cam.k[on] / RATE - o2) / (1 + d2)
    assert np.max(np.abs(t_hat - cam.t[on])) <= 1.0 / RATE + 1e-9          # two roundings of half a frame each
    assert np.allclose(align.corrected_sync(off, drift, {"offset_s": 0.0, "drift": 0.0}, start), (off, drift), rtol=0, atol=1e-12)
