import numpy as np
import pytest

from futsal import Court
from futsal.camera import Camera, yaw_towards
from futsal.fusion import Observation, fuse_frame
from futsal.homography import calibrate
from futsal import metrics
from futsal.sync import estimate_offset
from futsal.tracks import PlayerTrack, TrackSet

COURT = Court(40, 20)


def corner_camera(hfov=71.5):
    pos = (-1.3, -1.3)
    return Camera.on_tripod(pos, 2.0, yaw_towards(pos, COURT.centre, 5), hfov)


def test_keypoints_follow_law_1():
    kp = COURT.keypoints()
    assert len(kp) == 29
    assert kp["penalty_left"] == (6.0, 10.0) and kp["second_penalty_right"] == (30.0, 10.0)
    assert kp["area_line_top_left"] == pytest.approx((6.0, 11.58))
    assert kp["post_top_right"] == (40.0, 11.5)


def test_calibration_recovers_exact_homography():
    cam = corner_camera()
    names = list(COURT.keypoints())
    world = np.array([COURT.keypoints()[n] for n in names])
    uv, ok = cam.project(world)
    taps = {n: tuple(p) for n, p, o in zip(names, uv, ok) if o}
    assert len(taps) >= 15
    cal = calibrate(taps, COURT)
    assert cal.rms_px < 1e-3
    pts = np.array([[5.0, 5.0], [20.0, 10.0], [33.0, 17.0]])
    uv_p, _ = cam.project(pts)
    assert np.allclose(cal.to_pitch(uv_p), pts, atol=1e-4)
    assert np.allclose(cal.H, cam.floor_homography(), atol=1e-5)


def test_calibration_rejects_a_mistapped_point():
    cam = corner_camera()
    kp = COURT.keypoints()
    uv, ok = cam.project(np.array(list(kp.values())))
    taps = {n: tuple(p) for n, p, o in zip(kp, uv, ok) if o}
    bad = next(iter(taps))
    taps[bad] = (taps[bad][0] + 120, taps[bad][1] + 40)
    cal = calibrate(taps, COURT, ransac_px=8)
    assert bad not in cal.used and cal.rms_px < 1e-3


def test_calibration_needs_four_points():
    with pytest.raises(ValueError):
        calibrate({"corner_bl": (0, 0), "corner_br": (1, 0), "centre": (2, 2)}, COURT)


def test_fusion_merges_close_and_keeps_far():
    a = [Observation(np.array([10.0, 5.0]), 0.1, "cam1"), Observation(np.array([30.0, 15.0]), 0.6, "cam1")]
    b = [Observation(np.array([10.4, 5.0]), 0.3, "cam2"), Observation(np.array([2.0, 2.0]), 0.2, "cam2")]
    out = fuse_frame({"cam1": a, "cam2": b}, gate=1.0)
    assert len(out) == 3
    merged = next(f for f in out if len(f.sources) == 2)
    assert merged.xy[0] == pytest.approx(10.04, abs=1e-6)         # weighted towards the sharper camera
    assert merged.sigma < 0.1


def test_metrics_straight_run():
    dt = 0.1
    t = np.arange(0, 10, dt)
    xy = np.stack([3.0 * t, np.full_like(t, 10.0)], 1)              # 3 m/s for 10 s
    s = metrics.summary(xy, dt)
    assert s["distance_m"] == pytest.approx(29.7, abs=0.1)
    assert s["max_speed_ms"] == pytest.approx(3.0, abs=0.05)
    assert s["sprints"] == 0
    h = metrics.heatmap(xy, 40, 20)
    assert h.shape == (20, 40) and h.sum() == pytest.approx(1.0)


def test_metrics_do_not_count_jitter_as_running():
    rng = np.random.default_rng(0)
    xy = np.array([5.0, 5.0]) + rng.normal(0, 0.4, (600, 2))        # standing still, 40 cm noise, 60 s
    raw = np.nansum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    s = metrics.summary(xy, 0.1)
    assert raw > 250 and s["distance_m"] < 0.1 * raw
    assert s["max_speed_ms"] < 2 and s["sprints"] == 0


def test_metrics_sprint_and_gaps():
    dt = 0.1
    v = np.r_[np.full(50, 2.0), np.full(20, 6.5), np.full(50, 2.0)]   # one 2 s sprint
    xy = np.stack([np.cumsum(v) * dt, np.full(len(v), 5.0)], 1)
    xy[90:120] = np.nan                                                # 3 s unseen at the end -> not bridged
    s = metrics.summary(xy, dt)
    assert s["sprints"] == 1 and 6.0 < s["max_speed_ms"] < 7.0


def test_audio_offset():
    sr, rng = 16000, np.random.default_rng(0)
    n = sr * 40
    base = rng.normal(0, 0.01, n)
    for t in rng.uniform(1, 38, 25):                                  # ball strikes / whistles
        i = int(t * sr); base[i:i + 400] += rng.normal(0, 0.8, 400)
    lag = 2.5                                                          # phone 2 started 2.5 s later
    a = base
    b = np.r_[base[int(lag * sr):], np.zeros(int(lag * sr))] + rng.normal(0, 0.01, n)
    off, conf = estimate_offset(a, b, sr)
    assert off == pytest.approx(lag, abs=0.02) and conf > 5


def test_trackset_roundtrip(tmp_path):
    xy = np.array([[1.0, 2.0], [np.nan, np.nan], [1.5, 2.5]])
    ts = TrackSet(40, 20, 10, [PlayerTrack(1, "A", xy)])
    ts.compute_stats()
    ts.dump(tmp_path / "t.json")
    back = TrackSet.load(tmp_path / "t.json")
    assert np.allclose(back.players[0].xy, xy, equal_nan=True)
    assert back.players[0].stats["seen_ratio"] == pytest.approx(2 / 3)
