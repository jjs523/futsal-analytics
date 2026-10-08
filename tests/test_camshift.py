import cv2
import numpy as np

from futsal import Court, camshift
from futsal.camera import Camera, yaw_towards
from futsal.homography import Calibration, calibration_at, framing

W, H, FPS = 640, 360, 20


def _video(path, shift, seconds=24):
    """Static scene (lines, fence) with moving players, a player blocking the lens for 2 s, an exposure jump,
    and the camera knocked according to `shift(t)` -> (dx, dy) on screen."""
    rng = np.random.default_rng(0)
    canvas = cv2.add(np.full((H + 100, W + 100, 3), (40, 120, 50), np.uint8), rng.integers(0, 25, (H + 100, W + 100, 3), dtype=np.uint8))
    for _ in range(8):
        cv2.line(canvas, (int(rng.uniform(0, W + 100)), 0), (int(rng.uniform(0, W + 100)), H + 100), (230, 230, 230), 2)
    cv2.rectangle(canvas, (50, 30), (W + 50, 70), (90, 90, 90), -1)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * seconds):
        t = i / FPS
        dx, dy = shift(t)
        f = canvas[50 - dy:50 - dy + H, 50 - dx:50 - dx + W].copy()
        for p in range(5):
            x = int((100 + p * 90 + 40 * t * (1 + p % 2)) % W)
            cv2.rectangle(f, (x, 150 + 20 * p), (x + 12, 185 + 20 * p), (0, 200, 255), -1)
        if 13 <= t < 15:
            cv2.rectangle(f, (150, 50), (350, H), (20, 20, 20), -1)          # someone right in front of the lens
        if t >= 19:
            f = cv2.convertScaleAbs(f, alpha=1.25, beta=10)                   # auto exposure
        vw.write(f)
    vw.release()


def test_camera_moves_finds_knocks_and_ignores_players_and_exposure(tmp_path):
    path = str(tmp_path / "v.mp4")
    _video(path, lambda t: (0, 0) if t < 6 else ((-20, -9) if t < 17 else (-13, -9)))
    moves = camshift.camera_moves(path, min_px=2.0)["moves"]
    assert len(moves) == 2
    assert abs(moves[0]["t"] - 6) < 1 and abs(moves[0]["dx"] + 20) < 1.5 and abs(moves[0]["dy"] + 9) < 1.5
    assert abs(moves[1]["t"] - 17) < 1 and abs(moves[1]["dx"] - 7) < 1.5 and not moves[1]["retap"]


def test_no_moves_on_a_steady_camera(tmp_path):
    path = str(tmp_path / "v.mp4")
    _video(path, lambda t: (0, 0), seconds=12)
    assert camshift.camera_moves(path, min_px=2.0)["moves"] == []


def test_timeline_from_moves_keeps_pitch_positions():
    H0 = np.array([[0.02, 0.001, -5.0], [0.0005, 0.05, -10.0], [0.0, 0.0001, 1.0]])
    base = Calibration(H0, 1.0)                       # tapped at 1:00, after the first knock
    moves = [{"t": 30.0, "dx": -41, "dy": -19, "total_dx": -41, "total_dy": -19},
             {"t": 600.0, "dx": 14, "dy": 0, "total_dx": -27, "total_dy": -19}]
    tl = camshift.timeline_from_moves(base, moves, tapped_at_s=60.0)
    p = base.to_pitch([[900, 500]])[0]
    assert np.allclose(calibration_at(tl, 60).to_pitch([[900, 500]])[0], p)
    assert np.allclose(calibration_at(tl, 10).to_pitch([[941, 519]])[0], p)       # before the knock: 41 px right, 19 down
    assert np.allclose(calibration_at(tl, 700).to_pitch([[914, 500]])[0], p)


def test_framing_tells_a_tilted_down_phone_to_look_up():
    court = Court()
    pos = (-1.3, -1.3)
    good = Camera.on_tripod(pos, 2.0, yaw_towards(pos, (20, 10)), 70)
    low = Camera.on_tripod(pos, 2.0, yaw_towards(pos, (20, 10)), 70, horizon_margin_deg=-14)
    fg = framing(Calibration(good.floor_homography(), 0.0), 1920, 1080, court)
    fl = framing(Calibration(low.floor_homography(), 0.0), 1920, 1080, court)
    assert fl["visible"] < fg["visible"] - 0.2
    assert fl["advice"] == "카메라를 위로 드세요"
