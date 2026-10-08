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


def _match_audio(seconds, sr, rng):
    x = rng.normal(0, 0.01, int(seconds * sr))
    for t in np.r_[[1.0, 1.5, 2.0], rng.uniform(3, seconds - 3, int(seconds / 4)), [seconds - 2.5, seconds - 2.0, seconds - 1.5]]:
        i = int(t * sr); x[i:i + 300] += rng.normal(0, 0.8, 300)     # claps at both ends + ball strikes
    return x


def test_align_recovers_offset_and_clock_drift():
    from futsal.sync import align
    sr, rng = 8000, np.random.default_rng(1)
    a = _match_audio(600, sr, rng)                     # 10 min, phone 1 clock
    off, drift = 3.2, 80e-6                            # phone 2 started 3.2 s later and runs 80 ppm slow
    tb = np.arange(int((600 - 5) * sr)) / sr           # phone 2 sample times
    ta = tb + off + drift * tb                         # same instants on phone 1's clock
    b = np.interp(ta, np.arange(len(a)) / sr, a) + rng.normal(0, 0.01, len(tb))
    r = align(a, b, sr, window_s=60)
    assert r["offset"] == pytest.approx(off, abs=0.01)
    assert r["drift"] == pytest.approx(drift, abs=20e-6)
    assert r["conf_start"] > 6 and r["conf_end"] > 6


def test_drift_is_applied_when_bucketing_frames():
    from futsal.pipeline.run import clock
    assert clock(1.5) == (1.5, 0.0)
    assert clock({"offset": 1.5, "drift": 1e-4}) == (1.5, 1e-4)


def test_align_with_start_hint_when_phones_started_minutes_apart():
    from futsal.sync import align, start_hint
    sr, rng = 8000, np.random.default_rng(2)
    a = _match_audio(900, sr, rng)                     # phone 1: 15 min
    off, drift = 123.4, -60e-6                         # phone 2 started 2 min later (as in the file names)
    tb = np.arange(int((900 - off - 10) * sr)) / sr
    b = np.interp(tb + off + drift * tb, np.arange(len(a)) / sr, a) + rng.normal(0, 0.01, len(tb))
    assert start_hint("x/20261008_170919.mp4", "y/20261008_171122.mp4") == 123.0
    r = align(a, b, sr, window_s=60, hint_s=123.0)
    assert r["offset"] == pytest.approx(off, abs=0.01)
    assert r["drift"] == pytest.approx(drift, abs=20e-6)
    assert start_hint("cam1.mp4", "cam2.mp4") is None


def test_detection_rate_is_per_second_whatever_the_frame_rate(tmp_path):
    import cv2
    from futsal.pipeline.detect import detect_video
    counts = {}
    for fps in (30, 60):
        path = str(tmp_path / f"v{fps}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
        for _ in range(fps * 3):
            vw.write(np.zeros((90, 160, 3), np.uint8))
        vw.release()
        dets, _, _ = detect_video(path, lambda f: [(10, 10, 20, 40, 0.9, None)], rate_hz=10)
        counts[fps] = len(dets)
    assert counts[30] == counts[60] == 30


def _game_audio_no_claps(seconds, off, sr, rng, spread=0.13):
    """Match sound without claps: many similar strikes and voices, a third heard only by one phone, and each shared
    sound reaching the two phones (~45 m apart) up to +-`spread` s apart depending on where it was made."""
    a = rng.normal(0, 0.05, int(seconds * sr))
    b = rng.normal(0, 0.05, int((seconds - off - 10) * sr))
    for t in rng.uniform(0, seconds, int(seconds * 3)):
        s = rng.normal(0, 1, 300) * np.exp(-np.arange(300) / 80) * rng.uniform(0.1, 0.6)
        who = rng.random()
        i, j = int(t * sr), int((t + rng.uniform(-spread, spread) - off) * sr)
        if who > 0.3 and i + 300 <= len(a):
            a[i:i + 300] += s
        if (who < 0.3 or who > 0.6) and 0 <= j and j + 300 <= len(b):
            b[j:j + 300] += s
    return a, b


def test_align_segments_without_claps():
    from futsal.sync import align_segments
    sr = 8000
    errs = []
    for seed in range(3):
        a, b = _game_audio_no_claps(1500, 124.29, sr, np.random.default_rng(seed))
        r = align_segments(a, b, sr, hint_s=123.0, max_lag_s=5.0)
        errs.append(abs(r["offset"] - 124.29))
        assert r["n_used"] >= 15 and r["stderr"] < 0.03
    assert max(errs) < 0.05


def test_calibration_timeline_for_a_knocked_tripod():
    from futsal.homography import Calibration, calibration_at, timeline_from_json
    from futsal.pipeline.detect import Detection
    from futsal.pipeline.run import to_observations
    court = Court()
    H = np.array([[0.02, 0.001, -5.0], [0.0005, 0.05, -10.0], [0.0, 0.0001, 1.0]])
    before = Calibration(H, 1.0)
    after = before.shifted(-41, -19)                                     # cam1 at 2:40: picture slid 41 px left, 19 up
    u, v = 900.0, 500.0
    p = before.to_pitch([[u, v]])[0]
    assert np.allclose(after.to_pitch([[u - 41, v - 19]])[0], p)
    tl = timeline_from_json([{"from": 160.0, **after.to_json()}, {"from": 0, **before.to_json()}])
    assert calibration_at(tl, 100.0) is tl[0][1] and calibration_at(tl, 200.0) is tl[1][1]
    dets = [Detection(0, 100.0, u, v, 0.9, None, 60.0, None), Detection(1, 200.0, u - 41, v - 19, 0.9, None, 60.0, None)]
    obs = to_observations(dets, tl, "cam1", 0.0, 10.0, court, margin=50)
    pts = [o.xy for k in sorted(obs) for o in obs[k]]
    assert len(pts) == 2 and np.allclose(pts[0], pts[1])               # same pitch spot before and after the knock


def test_calibrate_fits_lens_distortion_only_when_it_helps():
    from futsal.homography import Calibration, calibrate
    court = Court()
    pos = (-1.3, -1.3)
    cam = Camera.on_tripod(pos, 2.2, yaw_towards(pos, (20, 10)), 72)
    kp = court.keypoints()
    uv, ok = cam.project(np.array(list(kp.values())))
    inside = ok & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    names = [n for n, k in zip(kp, inside) if k]
    rng = np.random.default_rng(0)
    lens = Calibration(np.eye(3), 0, [], -0.08, (960.0, 540.0), float(np.hypot(1920, 1080) / 2))   # barrel distortion
    clean = {n: tuple(p) for n, p in zip(names, uv[inside] + rng.normal(0, 0.5, (inside.sum(), 2)))}
    bent = {n: tuple(lens.distort([p])[0]) for n, p in clean.items()}
    assert len(names) >= 8
    plain = calibrate(bent, court, 8, (1920, 1080), distortion=False)
    fitted = calibrate(bent, court, 8, (1920, 1080))
    assert abs(fitted.k1 + 0.08) < 0.02 and fitted.rms_px < 1.5 < plain.rms_px
    assert len(fitted.used) >= len(plain.used)
    # a player's foot near the frame edge lands within 10 cm with the distortion model
    foot = np.array([[3.0, 4.0]])
    pix, _ = cam.project(foot)
    seen = lens.distort(pix)
    assert np.linalg.norm(fitted.to_pitch(seen) - foot) < 0.1
    assert np.allclose(fitted.to_image(fitted.to_pitch(seen)), seen, atol=0.5)
    # undistorted taps: no distortion is invented
    assert calibrate(clean, court, 8, (1920, 1080)).k1 == 0.0
    back = Calibration.from_json(fitted.to_json())
    assert back.k1 == fitted.k1 and np.allclose(back.to_pitch(seen), fitted.to_pitch(seen))
