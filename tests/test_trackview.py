"""Tracking result video end-to-end on synthetic two-camera footage (no YOLO needed)."""
import json
import os

import cv2
import numpy as np

from futsal import Court, calibtool, trackview
from futsal.pipeline import detect
from futsal.sim import layout, scenario, video

COURT = Court(40, 20)
SCALE = 0.5


def _setup(tmp_path):
    truth = scenario.synthetic_match(COURT, seconds=12, fps=10, seed=7)
    cams = layout.diagonal(COURT, 5).build(layout.HFOV_AVERAGE)
    paths = {}
    for name, cam in cams.items():
        paths[name] = str(tmp_path / f"{name}.mp4")
        video.render(cam, COURT, truth, paths[name], scale=SCALE)
    return truth, cams, paths


def _taps(cam):
    kp = COURT.keypoints()
    uv, ok = cam.project(np.array(list(kp.values())))
    w, h = cam.width * SCALE, cam.height * SCALE
    return {n: [float(u * SCALE), float(v * SCALE)] for n, (u, v), o in zip(kp, uv, ok)
            if o and 0 <= u * SCALE < w and 0 <= v * SCALE < h}


def test_calibtool_page_and_fit(tmp_path):
    _, cams, paths = _setup(tmp_path)
    page = calibtool.make_tap_page(paths["cam1"], 3.0, str(tmp_path / "calib1"), COURT, "cam1")
    html = open(page, encoding="utf-8").read()
    assert "data:image/jpeg;base64," in html and "센터 마크" in html and "__DATA__" not in html
    d = json.loads(html.split("const D = ", 1)[1].split(";\n", 1)[0])
    taps = {"video": d["video"], "at": d["at"], "camera": "cam1", "frame_path": d["frame_path"], "width": d["width"],
            "height": d["height"], "court": d["court"], "taps": _taps(cams["cam1"])}
    (tmp_path / "calib1" / "taps.json").write_text(json.dumps(taps))
    r = calibtool.fit(str(tmp_path / "calib1" / "taps.json"))
    assert r["rms_px"] < 1 and len(r["used"]) >= 6
    assert os.path.exists(tmp_path / "calib1" / "calib.json") and os.path.exists(tmp_path / "calib1" / "check.jpg")
    assert 0.3 < r["framing"]["visible"] <= 1


def test_trackview_end_to_end(tmp_path):
    _, cams, paths = _setup(tmp_path)
    cals = {}
    for name, cam in cams.items():
        out = tmp_path / f"calib_{name}"
        out.mkdir()
        (out / "taps.json").write_text(json.dumps({"at": 3.0, "width": int(cam.width * SCALE), "height": int(cam.height * SCALE),
                                                   "court": [40, 20], "taps": _taps(cam)}))
        calibtool.fit(str(out / "taps.json"))
        cals[name] = trackview.load_calibration(str(out / "calib.json"))
    out_dir = str(tmp_path / "tv")
    res = trackview.run(paths, cals, {"cam1": (0.0, 0.0), "cam2": (0.0, 0.0)}, start=2.0, duration=6.0, out_dir=out_dir,
                        detector_factory=lambda c: detect.synthetic_detector(video.background(cams[c], COURT, SCALE), video.VEST_TO_TEAM),
                        court=COURT)
    assert res["frames"] == 60
    assert 8 <= res["ids_over_half"] <= 12                      # 10 on the pitch
    from futsal.sim import evaluate
    from futsal.tracks import PlayerTrack, TrackSet
    truth = scenario.synthetic_match(COURT, seconds=12, fps=10, seed=7)
    window = TrackSet(40, 20, 10, [PlayerTrack(p.id, p.team, p.xy[20:80]) for p in truth.players])
    assert evaluate.id_scores(window, TrackSet.load(os.path.join(out_dir, "tracks.json")))["idf1"] > 0.6     # synthetic baseline 0.53-0.68 over full matches
    cap = cv2.VideoCapture(os.path.join(out_dir, "trackview.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
    tr = json.load(open(os.path.join(out_dir, "tracks.json")))
    assert tr["n_frames"] == 60 and tr["start"] == 2.0
    assert open(os.path.join(out_dir, "events.csv"), encoding="utf-8-sig").readline().startswith("시각(cam1)")
    # second run reuses the cached detections
    res2 = trackview.run(paths, cals, {"cam1": (0.0, 0.0), "cam2": (0.0, 0.0)}, start=2.0, duration=6.0, out_dir=out_dir,
                         detector_factory=lambda c: (_ for _ in ()).throw(AssertionError("should use cache")), court=COURT)
    assert res2["ids"] == res["ids"]


def test_find_events_flags_mid_pitch_breaks_only():
    from futsal.tracks import PlayerTrack, TrackSet
    n = 100
    a = np.full((n, 2), np.nan); a[:60] = [20, 10]                 # lost in the centre at frame 59
    b = np.full((n, 2), np.nan); b[30:] = [0.5, 10]                # appears on the goal line: a real entry
    c = np.tile([10.0, 10.0], (n, 1))                              # whole window
    ev = trackview.find_events(TrackSet(40, 20, 10, [PlayerTrack(1, "A", a), PlayerTrack(2, "A", b), PlayerTrack(3, "B", c)]), COURT)
    assert [(e["id"], e["kind"]) for e in ev] == [(1, "lost")]


def test_cam2_page_is_drawn_from_its_own_corner_and_flips_are_fixed(tmp_path):
    _, cams, paths = _setup(tmp_path)
    page = calibtool.make_tap_page(paths["cam2"], 3.0, str(tmp_path / "c2"), COURT, "cam2")
    d = json.loads(open(page, encoding="utf-8").read().split("const D = ", 1)[1].split(";\n", 1)[0])
    mine = next(k for k in d["keypoints"] if k["label"] == "내 폰 바로 앞 모서리")
    assert mine["name"] == "corner_tr" and mine["xy"] == [0.0, 0.0]          # drawn bottom-left, saved as the real corner
    assert not any(k["name"].startswith("sub_mark") for k in d["keypoints"])
    good = {k: v for k, v in _taps(cams["cam2"]).items() if not k.startswith("sub_mark")}
    wrong = {calibtool.rotated_name(COURT, k): v for k, v in good.items()}        # tapped as if it were cam1
    base = {"at": 3.0, "width": d["width"], "height": d["height"], "court": [40, 20], "camera": "cam2"}
    for taps, flipped in ((good, False), (wrong, True)):
        (tmp_path / "c2" / "taps.json").write_text(json.dumps(base | {"taps": taps}))
        r = calibtool.fit(str(tmp_path / "c2" / "taps.json"))
        assert r["flipped_180"] is flipped and r["rms_px"] < 1 and r["framing"]["visible"] > 0.3
