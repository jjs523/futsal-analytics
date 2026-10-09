"""run.build_tracks(ids='v2') end to end on a synthetic 5 v 5 two-camera scene (no video, no GPU, no boxmot):

    Detections (box + ReID + colour) -> CamBoxes -> teams -> alignment -> tracklets -> closed-set identities
    -> RTS positions -> TrackSet

The scene: ten players (five in yellow bibs, five in white) on a 40 x 20 m pitch, two cameras on opposite long
sides with cam2's clock 2.5 s behind and its calibration off by a smooth field, two yellow players crossing (their
boxes overlap in both views, so every per-camera tracklet is cut there and only the identity stage can carry them
through), a white player passing a yellow one, and cam1's tripod knocked half way (a calibration timeline).
"""
import numpy as np
import pytest

from futsal.court import Court
from futsal.homography import Calibration
from futsal.pipeline.detect import Detection
from futsal.pipeline.run import build_tracks, guess_frame_wh, has_reid
from futsal.tracks import TrackSet

pytest.importorskip("scipy")

COURT = Court()
RATE = 10.0
N = 150                          # grid frames (15 s)
CAM2_OFFSET = 2.5                # cam1 time = cam2 time + 2.5 s
KNOCK_S = 7.0                    # cam1's own clock: the picture slides by KNOCK_PX from here on
KNOCK_PX = (12.0, -6.0)

M1 = np.array([[40.0, 8.0, 160.0], [0.0, -25.0, 950.0], [0.0, 0.004, 1.0]])            # pitch -> image, y < 0 side
M2 = M1 @ np.array([[-1.0, 0.0, 40.0], [0.0, -1.0, 20.0], [0.0, 0.0, 1.0]])            # the opposite side
CAL1 = Calibration(np.linalg.inv(M1), 1.0)
CAL1_KNOCKED = CAL1.shifted(*KNOCK_PX)
CAL2 = Calibration(np.linalg.inv(M2), 1.0)
CALS = {"cam1": [(0.0, CAL1), (KNOCK_S, CAL1_KNOCKED)], "cam2": CAL2}
SYNC = {"cam1": 0.0, "cam2": {"offset": CAM2_OFFSET, "drift": 0.0}}

# (team, start, end): players 2 and 3 (both yellow) cross at (17, 10) half way, 7 (white) passes 1 (yellow)
PLAYERS = [("A", (4, 4), (12, 5)), ("A", (26, 4), (18, 6)), ("A", (12, 7), (22, 13)), ("A", (12, 13), (22, 7)),
           ("A", (34, 16), (28, 15)),
           ("B", (30, 10), (36, 8)), ("B", (6, 16), (12, 17)), ("B", (14, 6), (24, 3)), ("B", (24, 18), (32, 18)),
           ("B", (5, 10), (9, 11))]
CROSSING = (2, 3)


def warp(xy: np.ndarray) -> np.ndarray:
    """cam2's calibration error: a smooth offset of 0.3 - 0.7 m."""
    x, y = xy[:, 0] / 40.0, xy[:, 1] / 20.0
    return np.stack([0.4 + 0.3 * x * x, -0.3 + 0.2 * x * y], 1)


def truth(k) -> np.ndarray:
    """(players, 2) positions at grid frame(s) k: straight lines at constant speed."""
    s = np.clip(np.asarray(k, float) / (N - 1), 0, 1)[..., None, None]
    a = np.array([p[1] for p in PLAYERS], float)
    b = np.array([p[2] for p in PLAYERS], float)
    return (1 - s) * a + s * b


def box_of(cal: Calibration, xy) -> tuple[float, float, float, float]:
    u, v = cal.to_image(np.asarray(xy, float).reshape(1, 2))[0]
    h = 0.16 * v
    return (u - 0.2 * h, v - h, u + 0.2 * h, v)


def kit(team: str, rng) -> list[int]:
    """Quantised upper + lower body histogram: yellow bibs (hue bins 1-2) or white."""
    up = np.full(15, 0.01)
    up[1:3 if team == "A" else 1] += 0.35
    up[14] += 0.6 if team == "B" else 0.0
    up[12] += 0.2
    up = np.abs(up + rng.normal(0, 0.02, 15))
    lo = np.abs(np.full(15, 1 / 15) + rng.normal(0, 0.01, 15))
    return [int(round(x * 255)) for x in np.concatenate([up / up.sum(), lo / lo.sum()])]


def scene(reid: bool = True) -> dict[str, list[Detection]]:
    rng = np.random.default_rng(11)
    looks = rng.normal(size=(len(PLAYERS) + 1, 512))
    looks /= np.linalg.norm(looks, axis=1, keepdims=True)
    out = {}
    for cam in ("cam1", "cam2"):
        dets = []
        for k in range(N):
            t = k / RATE - (CAM2_OFFSET if cam == "cam2" else 0.0)            # the camera's own clock
            fid = 3 * k if cam == "cam1" else 6 * k + 2
            pos = truth(k)
            if cam == "cam2":
                pos = pos + warp(pos)
            cal = CAL2 if cam == "cam2" else (CAL1_KNOCKED if t >= KNOCK_S else CAL1)
            for j, xy in enumerate(pos):
                b = box_of(cal, xy)
                e = looks[j] + rng.normal(0, 0.02, 512)           # within-player 1 - cos ~ 0.17, like OSNet
                dets.append(Detection(fid, t, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], kit(PLAYERS[j][0], rng),
                                      b, (e / np.linalg.norm(e)).astype(np.float32) if reid else None))
            if k % 10 == 0:                                                  # a spectator off the pitch
                b = box_of(cal, (20.0, -4.0))
                dets.append(Detection(fid, t, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b,
                                      looks[-1].astype(np.float32) if reid else None))
        out[cam] = dets
    return out


@pytest.fixture(scope="module")
def v2():
    info: dict = {}
    ts = build_tracks(COURT, scene(), CALS, SYNC, fps_out=RATE, ids="v2", info=info)
    return ts, info


def _match(ts: TrackSet) -> dict[int, int]:
    """Output player id -> the true player it follows (nearest at most of its frames), checked one to one."""
    T = truth(np.arange(N))
    out = {}
    for p in ts.players:
        ok = np.isfinite(p.xy[:, 0])
        d = np.linalg.norm(p.xy[ok][:, None] - T[ok], axis=2)
        out[p.id] = int(np.bincount(d.argmin(1)).argmax())
    assert sorted(out.values()) == list(range(len(PLAYERS)))
    return out


def test_v2_gives_one_identity_per_player_with_teams(v2):
    ts, info = v2
    assert info["ids"] == "v2" and "fallback" not in info
    assert ts.fps == RATE and ts.n_frames == N and len(ts.players) == len(PLAYERS)
    assert [p.id for p in ts.players] == list(range(1, len(PLAYERS) + 1))
    who = _match(ts)
    lab = {p.id: p.team for p in ts.players}
    assert set(lab.values()) == {"A", "B"}
    # A / B are cluster names: one consistent mapping onto the two kits
    pairs = {(lab[i], PLAYERS[j][0]) for i, j in who.items()}
    assert len(pairs) == 2 and len({a for a, _ in pairs}) == 2
    # extras for the caller: alignment found cam2's error, the solver ran
    al = info["alignment"]
    assert al is not None and al["pairs"] >= 200 and al["median_after_m"] < 0.1 < 0.3 < al["median_before_m"]
    assert info["tracklets"] > len(PLAYERS) and info["identities"] == len(PLAYERS)
    assert info["solver"].windows and len(info["solver"].per_window) == 1          # 15 s: a single window


def test_v2_ids_stay_on_their_player_through_the_crossing(v2):
    ts, _ = v2
    T = truth(np.arange(N))
    who = _match(ts)
    for p in ts.players:
        j = who[p.id]
        ok = np.isfinite(p.xy[:, 0])
        assert ok.mean() > 0.95                                       # present almost the whole clip
        # on its own player at every frame (a swap at the crossing would put it metres off after it). The alignment
        # meets the two calibrations half way: cam1 is exact here, so the target is half of cam2's error away
        mid = T[ok, j] + 0.5 * warp(T[ok, j])
        err = np.linalg.norm(p.xy[ok] - mid, axis=1)
        assert np.median(err) < 0.05 and err.max() < 0.3, f"id {p.id} leaves player {j}"
    for j in CROSSING:                                                # the crossing players are covered through it
        (pid,) = [i for i, q in who.items() if q == j]
        xy = ts.players[pid - 1].xy
        assert np.isfinite(xy[N // 2 - 10:N // 2 + 10, 0]).all()


def test_v2_trackset_round_trips_through_json(v2):
    ts, _ = v2
    back = TrackSet.from_json(ts.to_json())
    assert back.n_frames == N and [p.team for p in back.players] == [p.team for p in ts.players]
    assert all(np.allclose(a.xy, b.xy, atol=0.006, equal_nan=True) for a, b in zip(ts.players, back.players))
    assert all(p.stats for p in ts.players)


def test_without_reid_v2_falls_back_to_appearance():
    dets = scene(reid=False)
    assert not has_reid(dets) and has_reid(scene())
    info: dict = {}
    ts = build_tracks(COURT, dets, CALS, SYNC, fps_out=RATE, info=info)            # ids="v2" is the default
    assert info == {"fallback": "no ReID embeddings", "ids": "appearance"}
    ref = build_tracks(COURT, dets, CALS, SYNC, fps_out=RATE, ids="appearance")
    assert ts.to_json() == ref.to_json() and len(ts.players) >= len(PLAYERS) - 2
    # one camera without ReID is enough to fall back
    mixed = scene()
    mixed["cam2"] = dets["cam2"]
    info = {}
    build_tracks(COURT, mixed, CALS, SYNC, fps_out=RATE, info=info)
    assert info["ids"] == "appearance"
    with pytest.raises(ValueError):
        build_tracks(COURT, dets, CALS, SYNC, ids="v3")


def test_guess_frame_wh():
    d = lambda x2, y2: Detection(0, 0.0, 0, 0, 0.9, box=(x2 - 10, y2 - 30, x2, y2))
    assert guess_frame_wh([d(1919.5, 1079.0)]) == (1920, 1080)
    assert guess_frame_wh([d(2500.0, 1000.0)]) == (3840, 2160)
    assert guess_frame_wh([d(4000.0, 1000.0)]) == (4000, 1000)
    assert guess_frame_wh([Detection(0, 0.0, 5, 5, 0.9)]) == (1920, 1080)


def test_v2_long_clip_is_solved_in_chained_windows():
    info: dict = {}
    ts = build_tracks(COURT, scene(), CALS, SYNC, fps_out=RATE, info=info, window_s=8.0, overlap_s=3.0)
    assert len(info["solver"].windows) == 3 and info["solver"].matched == [len(PLAYERS)] * 2
    assert len(ts.players) == len(PLAYERS)
    T = truth(np.arange(N))
    for p, j in _match(ts).items():
        xy = ts.players[p - 1].xy
        ok = np.isfinite(xy[:, 0])
        assert np.linalg.norm(xy[ok] - T[ok, j] - 0.5 * warp(T[ok, j]), axis=1).max() < 0.3


def test_trackview_runs_v2_with_reid_and_caches_it(tmp_path):
    """trackview plumbing on synthetic footage (as tests/test_trackview.py): ReID per box into the .reid.npy sidecar,
    a cache key that knows about it, the v2 tracker, the video. The stand-in embedder is the bib + shorts colour
    histogram, so this checks the wiring, not identity quality."""
    import json
    import os

    import cv2

    from futsal import calibtool, trackview
    from futsal.pipeline import detect
    from futsal.sim import video
    from test_trackview import SCALE, _setup, _taps

    _, cams, paths = _setup(tmp_path)
    cals = {}
    for name, cam in cams.items():
        out = tmp_path / f"calib_{name}"
        out.mkdir()
        (out / "taps.json").write_text(json.dumps({"at": 3.0, "width": int(cam.width * SCALE),
                                                   "height": int(cam.height * SCALE), "court": [40, 20],
                                                   "taps": _taps(cam)}))
        calibtool.fit(str(out / "taps.json"))
        cals[name] = trackview.load_calibration(str(out / "calib.json"))
    calls = []

    def embed(frame, xyxy):
        calls.append(len(xyxy))
        f = np.array([detect.appearance(frame, b) for b in xyxy], np.float32).reshape(len(xyxy), -1)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)

    court = Court(40, 20)
    factory = lambda c: detect.synthetic_detector(video.background(cams[c], court, SCALE), video.VEST_TO_TEAM)
    out_dir = str(tmp_path / "tv")
    sync = {"cam1": (0.0, 0.0), "cam2": (0.0, 0.0)}
    res = trackview.run(paths, cals, sync, start=2.0, duration=4.0, out_dir=out_dir, detector_factory=factory,
                        court=court, embedder=embed)
    assert res["tracker"] == "v2" and calls and res["frames"] == 40 and 6 <= res["ids_over_half"] <= 14
    for cam in cams:
        cache = os.path.join(out_dir, f"det_{cam}.json")
        assert os.path.exists(detect.reid_path(cache)) and detect.load(cache)[1]["tag"].endswith("+reid")
        assert all(d.reid is not None and d.box is not None for d in detect.load(cache)[0])
    cap = cv2.VideoCapture(os.path.join(out_dir, "trackview.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    # cached detections (with their embeddings) are reused
    n = len(calls)
    again = trackview.run(paths, cals, sync, start=2.0, duration=4.0, out_dir=out_dir,
                          detector_factory=lambda c: (_ for _ in ()).throw(AssertionError("should use cache")),
                          court=court, embedder=embed)
    assert len(calls) == n and again["tracker"] == "v2" and again["ids"] == res["ids"]


def test_alignment_is_not_extrapolated_off_the_pitch():
    """align.apply_alignment: the cubic correction is exact on the pitch + VALID_MARGIN_M and held at the zone's edge
    further out (an off-pitch box once went to 3e8 m), so it stays bounded and continuous."""
    from futsal.pipeline import align
    from futsal.pipeline.boxes import build_cam_boxes

    far = [(5.0, 5.0), (40.0 + align.VALID_MARGIN_M, 10.0), (300.0, -200.0), (40.0 + align.VALID_MARGIN_M + 0.01, 10.0)]
    dets = [Detection(0, 0.0, *CAL1.to_image([xy])[0], 0.9, None, 80.0) for xy in far]
    cam = build_cam_boxes("cam1", dets, CAL1, 0.0, 0.0, 0.0, RATE, COURT)
    assert np.allclose(cam.xy_raw, far, atol=1e-6)
    W = np.zeros((10, 2))
    W[6] = [3.0, -2.0]                                                 # a strong x^3 term
    align.apply_alignment({"cam1": cam}, {"cam1": W.tolist()}, COURT)
    shift = lambda x: 0.5 * np.array([3.0, -2.0]) * (x / COURT.length) ** 3
    assert np.allclose(cam.xy[0], np.array(far[0]) + shift(5.0))      # on the pitch: the fitted field
    assert np.allclose(cam.xy[1], np.array(far[1]) + shift(43.0))     # at the zone edge: still the field
    edge = shift(43.0)                                                # the x^3 term only depends on x
    assert np.allclose(cam.xy[3] - cam.xy_raw[3], edge, atol=1e-9)    # just outside: no jump
    assert np.allclose(cam.xy[2] - cam.xy_raw[2], edge, atol=1e-9)    # 300 m out: held at x = 43, not 300^3
    assert not cam.in_court[1:].any()
