"""Box-level tracking foundation: per-camera boxes on the frame grid, teams, cross-camera alignment, ReID storage.
The research parity check on the cached match runs only with FUTSAL_RESEARCH_PARITY=1 (needs experiments/cache)."""
import json
import os
import sys

import numpy as np
import pytest

from futsal.court import Court
from futsal.homography import Calibration
from futsal.pipeline import detect
from futsal.pipeline.align import VALID_MARGIN_M, align_cameras, apply_alignment
from futsal.pipeline.boxes import (CamBoxes, assign_teams, build_cam_boxes, cam_boxes_from_cache, cams_from_cache,
                                   grid_frames)
from futsal.pipeline.detect import Detection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "experiments", "cache")
COURT = Court()
# a plausible side-on camera: 1920x1080 frame, whole pitch visible
CAL = Calibration(np.linalg.inv(np.array([[40.0, 8.0, 160.0], [0.0, -25.0, 950.0], [0.0, 0.004, 1.0]])), 1.0)


def _box_at(cal: Calibration, xy, h: float = 120.0) -> tuple[float, float, float, float]:
    u, v = cal.to_image([xy])[0]
    return (u - 0.2 * h, v - h, u + 0.2 * h, v)


def test_grid_frames_give_every_source_frame_its_own_k():
    # 60 fps sampled every 6th frame: times at x.x5 s round two consecutive samples onto the same 10 Hz frame
    t = 100.0 + np.arange(0, 600, 6) / 60.0 + 0.05
    fid = np.arange(100)
    k = grid_frames(fid, t, offset_s=0.0, drift=0.0, start_s=100.0, rate=10.0)
    assert len(set(k)) == 100 and np.all(np.diff(k) > 0)
    # several boxes of one source frame share its k; drift stretches the camera clock onto the reference clock
    k2 = grid_frames(np.array([0, 0, 1]), np.array([10.0, 10.0, 20.0]), offset_s=5.0, drift=0.01, start_s=0.0, rate=10.0)
    assert list(k2) == [151, 151, 252]


def test_build_cam_boxes_from_detections():
    dets, truth = [], []
    for f in range(5):
        for j, xy in enumerate([(5.0 + f, 4.0), (30.0, 15.0 - f), (60.0, 40.0)]):       # last one is far off the pitch
            b = _box_at(CAL, xy)
            dets.append(Detection(f, 200.0 + f / 10, (b[0] + b[2]) / 2, b[3], 0.9 - 0.1 * j, None, b[3] - b[1],
                                  [10] * 30 if j == 0 else None, b, np.full(512, 512 ** -0.5, np.float32) if j < 2 else None))
            truth.append(xy)
    dets.append(Detection(5, 200.5, *CAL.to_image([(20.0, 10.0)])[0], 0.8, None, 100.0))  # an old detection: no box
    cam = build_cam_boxes("cam1", dets, CAL, offset_s=3.0, drift=0.0, start_s=203.0, rate=10.0, court=COURT)
    assert isinstance(cam, CamBoxes) and len(cam) == 16
    assert np.allclose(cam.xy[:15], truth, atol=1e-6) and np.allclose(cam.xy[15], (20.0, 10.0), atol=1e-6)
    assert np.array_equal(cam.xy, cam.xy_raw)
    assert list(cam.k[::3]) == [0, 1, 2, 3, 4, 5]
    assert np.allclose(cam.w[15], 40.0) and np.allclose(cam.h[15], 100.0)               # rebuilt from foot + height
    assert cam.reid.shape == (16, 512) and np.allclose(np.linalg.norm(cam.reid[:2], axis=1), 1)
    assert not cam.reid[2].any()                                                         # missing -> zero row
    assert cam.color.shape == (16, 30) and cam.color[0].max() == pytest.approx(10 / 255) and not cam.color[1].any()
    assert list(cam.boxes_at(2)) == [6, 7]                                              # the off-pitch box is dropped
    assert list(cam.boxes_at(2, court_only=False)) == [6, 7, 8]
    assert list(cam.boxes_at(2, min_conf=0.85)) == [6]
    assert len(cam.boxes_at(99)) == 0
    assert np.all(cam.sigma > 0) and set(cam.team) == {""}


def test_detections_keep_boxes_and_reid_through_save_and_load(tmp_path):
    path = str(tmp_path / "det.json")
    e = np.random.default_rng(0).normal(size=(2, 512)).astype(np.float32)
    dets = [Detection(0, 1.0, 10, 20, 0.9, None, 30.0, [1] * 30, (5.0, -10.0, 15.0, 20.0), e[0]),
            Detection(0, 1.0, 50, 60, 0.8, None, 30.0, None, (45.0, 30.0, 55.0, 60.0), None),
            Detection(1, 1.1, 10, 20, 0.7, None, 30.0, None, None, e[1])]
    detect.save(dets, path, fps=30.0)
    raw = json.load(open(path))
    assert "reid" not in raw["detections"][0] and raw["detections"][0]["box"] == [5.0, -10.0, 15.0, 20.0]
    assert "box" not in raw["detections"][2]
    back, meta = detect.load(path)
    assert meta == {"fps": 30.0} and back == dets                           # reid is not part of equality
    assert back[0].box == (5.0, -10.0, 15.0, 20.0) and back[2].box is None
    assert back[1].reid is None and np.allclose(back[0].reid, e[0], atol=1e-2) and np.allclose(back[2].reid, e[1], atol=1e-2)
    # saving again without embeddings removes the sidecar, so stale ones are never attached
    detect.save([Detection(0, 1.0, 10, 20, 0.9)], path)
    assert not os.path.exists(path + ".reid.npy") and detect.load(path)[0][0].reid is None
    # files written before boxes were kept
    old = {"meta": {}, "detections": [{"frame": 0, "t": 1.0, "u": 1, "v": 2, "conf": 0.5, "team": None, "h_px": 3.0, "feat": None}]}
    json.dump(old, open(path, "w"))
    d, _ = detect.load(path)
    assert d[0].box is None and d[0].reid is None


def test_detect_video_embeds_every_box_once_per_frame(tmp_path):
    import cv2
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    for _ in range(30):
        vw.write(np.full((90, 160, 3), 120, np.uint8))
    vw.release()
    calls = []

    def embedder(frame, xyxy):
        calls.append(len(xyxy))
        return np.tile(np.eye(4, dtype=np.float32)[:len(xyxy)], (1, 128))

    two = lambda f: [(10, 10, 20, 40, 0.9, None), (50, 20, 60, 50, 0.8, None)]
    dets, _, _ = detect.detect_video(path, two, rate_hz=10, scale=2.0, embedder=embedder)
    assert len(dets) == 20 and calls == [2] * 10
    assert dets[1].box == (100.0, 40.0, 120.0, 100.0) and (dets[1].u, dets[1].v) == (110.0, 100.0)
    assert all(d.reid is not None and d.reid.shape == (512,) for d in dets)
    dets, _, _ = detect.detect_video(path, two, rate_hz=10)
    assert dets[0].reid is None and dets[0].box == (10.0, 10.0, 20.0, 40.0)


def _hist(yellow: float, white: float, rng) -> np.ndarray:
    """Upper + lower body histogram with a given yellow (hue bins 1-2) and white share in the upper body."""
    up = np.zeros(15)
    up[1:3] = yellow / 2
    up[14] = white
    up[12] = max(0.0, 1 - yellow - white)
    up = np.abs(up + rng.normal(0, 0.02, 15))
    return np.concatenate([up / up.sum(), np.full(15, 1 / 15)])


def test_assign_teams_splits_two_kits_and_leaves_unclear_boxes_blank():
    rng = np.random.default_rng(1)
    cams = {}
    for name in ("cam1", "cam2"):
        dets, kit = [], []
        for f in range(40):
            for j in range(10):
                yellow = j < 5
                b = _box_at(CAL, (4.0 + 3 * j, 10.0))
                dets.append(Detection(f, f / 10, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b))
                kit.append(_hist(0.6, 0.1, rng) if yellow else _hist(0.0, 0.7, rng))
        mid = _box_at(CAL, (20.0, 5.0))
        dets.append(Detection(40, 4.0, (mid[0] + mid[2]) / 2, mid[3], 0.9, None, 120.0, None, mid))
        kit.append(np.mean(kit, axis=0))                               # half way between the two kits
        tiny = (100.0, 900.0, 108.0, 920.0)
        dets.append(Detection(40, 4.0, 104.0, 920.0, 0.9, None, 20.0, None, tiny))
        kit.append(_hist(0.6, 0.1, rng))
        cam = build_cam_boxes(name, dets, CAL, 0.0, 0.0, 0.0, 10.0, COURT)
        cam.color = np.array(kit)
        cams[name] = cam
    info = assign_teams(cams)
    assert info["n_fit"] == 2 * (400 + 1)
    for cam in cams.values():
        team = cam.team[:400].reshape(40, 10)
        a = team[0, 0]
        assert a in ("A", "B") and np.all(team[:, :5] == a) and np.all((team[:, 5:] != a) & (team[:, 5:] != ""))
        assert cam.team[400] == "" and cam.team[401] == ""
    # no colour at all: nothing to fit, every box stays ''
    for cam in cams.values():
        cam.color = None
    assert assign_teams(cams) == {"n_fit": 0} and all(set(c.team) == {""} for c in cams.values())


def _two_warped_cameras(n_frames: int = 60, with_reid: bool = True) -> tuple[dict[str, CamBoxes], np.ndarray]:
    """Ten players seen by two cameras whose calibrations disagree by a smooth warp of up to ~1 m."""
    rng = np.random.default_rng(2)
    players = rng.uniform([3, 3], [37, 17], (10, 2))
    ident = np.eye(10, 512, dtype=np.float32)
    warp = lambda xy, s: xy + s * np.stack([0.5 * np.sin(xy[:, 1] / 6), 0.4 * np.cos(xy[:, 0] / 9)], 1)
    cams = {}
    for name, s in (("cam1", 1.0), ("cam2", -1.0)):
        dets = []
        for f in range(n_frames):
            pos = warp(players + 0.05 * f, s)
            for j, xy in enumerate(pos):
                b = _box_at(CAL, xy)
                dets.append(Detection(f, f / 10, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b,
                                      ident[j] if with_reid else None))
        cams[name] = build_cam_boxes(name, dets, CAL, 0.0, 0.0, 0.0, 10.0, COURT)
    return cams, players


def test_align_cameras_moves_both_cameras_to_the_midpoint():
    cams, _ = _two_warped_cameras()
    before = np.median(np.linalg.norm(cams["cam1"].xy - cams["cam2"].xy, axis=1))
    al = align_cameras(cams, COURT)
    assert al is not None and al["pairs"] >= 400 and set(al["coef"]) == {"cam1", "cam2"}
    after = np.median(np.linalg.norm(cams["cam1"].xy - cams["cam2"].xy, axis=1))
    assert before > 0.4 and after < 0.1 and al["median_after_m"] < 0.1 < 0.4 < al["median_before_m"]
    # re-applying the same coefficients starts again from xy_raw, so it does not compound
    moved = cams["cam1"].xy.copy()
    apply_alignment(cams, al["coef"], COURT)
    assert np.allclose(cams["cam1"].xy, moved)
    assert align_cameras(_two_warped_cameras(n_frames=10)[0], COURT) is None                 # 100 pairs: too few
    no_reid, _ = _two_warped_cameras(with_reid=False)
    assert align_cameras(no_reid, COURT) is None and np.array_equal(no_reid["cam1"].xy, no_reid["cam1"].xy_raw)


@pytest.mark.skipif(os.environ.get("FUTSAL_RESEARCH_PARITY") != "1" or not os.path.exists(os.path.join(CACHE, "meta.json")),
                    reason="set FUTSAL_RESEARCH_PARITY=1 (needs experiments/cache)")
def test_parity_with_the_research_harness():
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from harness import Data
    raw, aligned = Data(align=False), Data(align=True)
    cams, meta = cams_from_cache(CACHE, ROOT)
    for name, cam in cams.items():
        ref = raw.cams[name]
        assert np.array_equal(cam.k, ref.k) and np.array_equal(cam.in_court, ref.in_court)
        assert np.allclose(cam.xy, ref.xy, atol=1e-9) and np.allclose(cam.sigma, ref.sigma, atol=1e-9)
        assert set(cam.by_k) == set(ref.by_k) and all(np.array_equal(cam.by_k[k], ref.by_k[k]) for k in ref.by_k)
    assign_teams(cams)
    ours = np.concatenate([c.team for c in cams.values()])
    lab = np.concatenate([raw.cams[n].team for n in cams])
    labelled = lab != ""
    y, o = lab[labelled] == "Y", ours[labelled]
    # A/B mapped to Y/N by majority; a box we leave '' counts as a disagreement
    agree = max(np.mean(np.where(y, o == "A", o == "B")), np.mean(np.where(y, o == "B", o == "A")))
    assert agree >= 0.95
    al = align_cameras(cams, COURT, k_range=(0, raw.n))
    assert al["pairs"] == aligned.alignment["pairs"]
    for name, cam in cams.items():
        # the research applies the cubic everywhere (one off-pitch box went to 3e8 m); src holds it at the zone edge
        zone = COURT.contains(cam.xy_raw, VALID_MARGIN_M)
        assert np.allclose(cam.xy[zone], aligned.cams[name].xy[zone], atol=1e-6, rtol=0)
        assert np.abs(cam.xy - cam.xy_raw).max() < 5.0
        assert np.array_equal(cam.in_court, aligned.cams[name].in_court)
    one = cam_boxes_from_cache(os.path.join(CACHE, "cam2.npz"), "cam2", cams["cam2"].cal, meta["offsets"]["cam2"],
                               meta["start"], meta["rate"], COURT)
    assert np.array_equal(one.xy, cams["cam2"].xy_raw)
