"""Box-level tracking foundation: identity blocks (pipeline.identity) and per-camera / cross-view tracklets
(pipeline.tracklets) on small synthetic CamBoxes. The research parity check on the cached match runs only with
FUTSAL_RESEARCH_PARITY=1 (it needs experiments/cache and takes ~30 s)."""
import os

import numpy as np
import pytest

from futsal import Court
from futsal.homography import Calibration
from futsal.pipeline import identity, tracklets
from futsal.pipeline.boxes import CamBoxes, TrackPoint

COURT = Court(40, 20)
CAL = Calibration(np.diag([1 / 48, 1 / 54, 1.0]), 0.0)      # 1920 x 1080 px -> 40 x 20 m
RATE = 10.0


def box(u, v, h=120.0, w=50.0):
    return (u - w / 2, v - h, u + w / 2, v)


def make_cam(name, rows):
    """rows: (k, xyxy, conf, reid, team) per box."""
    k = np.array([r[0] for r in rows], int)
    xyxy = np.array([r[1] for r in rows], float).reshape(-1, 4)
    foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)
    xy = CAL.to_pitch(foot)
    by_k = {}
    for i in np.argsort(k, kind="stable"):
        by_k.setdefault(int(k[i]), []).append(int(i))
    return CamBoxes(name=name, fid=k * 3, t=k / RATE, k=k, xyxy=xyxy, conf=np.array([r[2] for r in rows], float),
                    reid=np.array([r[3] for r in rows], np.float32), color=None, foot=foot,
                    h=xyxy[:, 3] - xyxy[:, 1], w=xyxy[:, 2] - xyxy[:, 0], xy=xy, xy_raw=xy.copy(),
                    sigma=3.0 * CAL.metres_per_pixel(foot), in_court=COURT.contains(xy, 1.0),
                    team=np.array([r[4] for r in rows], dtype="<U1"), cal=CAL, frame_wh=(1920, 1080),
                    by_k={kk: np.array(v) for kk, v in by_k.items()})


E = np.eye(8, dtype=np.float32)


def two_players(name, n=30, crossing=False):
    """Player 0 (team A, ReID e0) and player 1 (team B, ReID e1) walking along y = 11 m; box index = 2 k + player."""
    rows = []
    for k in range(n):
        u0, u1 = (700 + 10 * k, 1000 - 10 * k) if crossing else (400 + 5 * k, 1400 - 5 * k)
        rows.append((k, box(u0, 600), 0.9, E[0], "A"))
        rows.append((k, box(u1, 600), 0.9, E[1], "B"))
    return make_cam(name, rows)


def player_of(track):
    return {i % 2 for p in track.values() for _, i in p.boxes}


# ---------------------------------------------------------------------------------------------------------------
# tracklets
# ---------------------------------------------------------------------------------------------------------------

def test_conservative_tracklets_follow_two_separate_players():
    cams = {"cam1": two_players("cam1")}
    out = tracklets.conservative_tracklets(cams, "cam1", RATE)
    assert len(out) == 2 and all(len(t) == 30 for t in out)
    assert all(len(player_of(t)) == 1 for t in out)
    t = out[0]
    assert all(len(p.boxes) == 1 and np.allclose(p.xy, cams["cam1"].xy[p.boxes[0][1]]) for p in t.values())


def test_conservative_tracklets_cut_at_a_crossing_instead_of_mixing():
    cams = {"cam1": two_players("cam1", crossing=True)}
    st = tracklets.CutStats()
    out = tracklets.conservative_tracklets(cams, "cam1", RATE, stats=st)
    assert len(out) > 2 and st.overlap > 0                     # the overlapped boxes become their own tracklets
    assert all(len(player_of(t)) == 1 for t in out)            # ... and no tracklet holds both players
    used = [cb for t in out for p in t.values() for cb in p.boxes]
    assert len(used) == len(set(used)) == 60


def test_pair_views_fuses_the_two_cameras_views_of_each_player():
    cams = {"cam1": two_players("cam1"), "cam2": two_players("cam2")}
    out = tracklets.build_tracklets(cams, rate=RATE)
    assert len(out) == 2
    for t in out:
        assert len(t) == 30 and len(player_of(t)) == 1
        assert all(sorted(c for c, _ in p.boxes) == ["cam1", "cam2"] for p in t.values())


def test_pair_views_respects_teams_and_min_shared():
    cams = {"cam1": two_players("cam1"), "cam2": two_players("cam2")}
    cams["cam2"].team = np.where(cams["cam2"].team == "A", "B", "A")       # contradicting teams ...
    assert len(tracklets.build_tracklets(cams, rate=RATE, team_override_sim=None)) == 4   # ... never fused by colour
    assert len(tracklets.build_tracklets(cams, rate=RATE)) == 2     # ... unless the two views also look alike (ReID)
    cams = {"cam1": two_players("cam1"), "cam2": two_players("cam2")}
    assert len(tracklets.build_tracklets(cams, rate=RATE, min_shared=40)) == 4    # routed to pair_views
    with pytest.raises(TypeError):
        tracklets.build_tracklets(cams, rate=RATE, no_such_param=1)


def test_build_tracklets_single_camera_and_frame_window():
    cams = {"cam1": two_players("cam1")}
    assert len(tracklets.build_tracklets(cams, rate=RATE)) == 2
    out = tracklets.build_tracklets(cams, rate=RATE, frames=range(10, 20))
    assert sorted(len(t) for t in out) == [10, 10] and min(min(t) for t in out) == 10


# ---------------------------------------------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------------------------------------------

def test_team_vote_and_split_at_a_lasting_flip():
    rows = [(k, box(400 + 3 * k, 600), 0.9, E[0], "A" if k < 30 else "B") for k in range(60)]
    cams = {"cam1": make_cam("cam1", rows)}
    track = {k: TrackPoint(cams["cam1"].xy[k], [("cam1", k)]) for k in range(60)}
    assert identity.team_vote(track, cams)[0] == "U"
    pieces = identity.split_team_flips([track], cams)
    assert [sorted(p)[0] for p in pieces] == [0, 30] and sum(len(p) for p in pieces) == 60
    assert [identity.team_vote(p, cams)[0] for p in pieces] == ["A", "B"]
    blip = {k: TrackPoint(track[k].xy, [("cam1", k)]) for k in range(30)}         # one-off flips do not cut
    assert len(identity.split_team_flips([blip], cams)) == 1


def test_clean_masks_border_exemption_and_overlap():
    rows = [(0, (500, 0, 550, 120), 0.9, E[0], "A"),             # touches the top edge
            (0, (900, 300, 950, 420), 0.9, E[1], "B"),           # overlaps the next box
            (0, (910, 300, 960, 420), 0.9, E[2], "B"),
            (0, (1300, 300, 1350, 420), 0.9, E[3], "A")]         # clean
    cams = {"cam2": make_cam("cam2", rows)}
    assert identity.clean_masks(cams)["cam2"].tolist() == [False, False, False, True]
    assert identity.clean_masks(cams, border_exempt={"cam2": ("top",)})["cam2"].tolist() == [True, False, False, True]
    with pytest.raises(ValueError):
        identity.clean_masks(cams, border_exempt={"cam2": ("up",)})


def test_fuse_and_covariance():
    cams = {"cam1": two_players("cam1"), "cam2": two_players("cam2")}
    R = identity.covariance(cams, "cam1", 0)
    assert R.shape == (2, 2) and np.all(np.linalg.eigvalsh(R) >= identity.HOMOGRAPHY_FLOOR_M ** 2 - 1e-12)
    xy, cov = identity.fuse(cams, [("cam1", 0)])
    assert np.allclose(xy, cams["cam1"].xy[0]) and np.allclose(cov, R)
    xy2, cov2 = identity.fuse(cams, [("cam1", 0), ("cam2", 0)])
    assert np.allclose(xy2, xy) and np.all(np.linalg.eigvalsh(cov - cov2) > 0)
    assert np.allclose(identity.point_cov(cams, TrackPoint(xy, [])), identity.NO_BOX_VAR * np.eye(2))
    with pytest.raises(ValueError):
        identity.fuse(cams, [])


def test_reachable_and_cannot_link():
    cams = {"cam1": two_players("cam1", n=40)}
    c = cams["cam1"]
    a = {k: TrackPoint(c.xy[2 * k], [("cam1", 2 * k)]) for k in range(20)}
    b = {k: TrackPoint(c.xy[2 * k], [("cam1", 2 * k)]) for k in range(20, 40)}
    far = {k: TrackPoint(c.xy[2 * k] + [15.0, 0.0], []) for k in range(20, 25)}
    other = {k: TrackPoint(c.xy[2 * k + 1], [("cam1", 2 * k + 1)]) for k in range(20, 40)}
    assert identity.reachable(a, b, cams, RATE) and not identity.cannot_link(a, b, cams, RATE)
    assert not identity.reachable(a, far, cams, RATE) and identity.cannot_link(a, far, cams, RATE)
    assert identity.cannot_link(a, other, cams, RATE)                       # team A vs team B
    assert identity.cannot_link(a, a, cams, RATE)                           # time overlap
    assert identity.overlap(a, b) == 0


def test_track_embedding_and_app_distance():
    cams = {"cam1": two_players("cam1")}
    c = cams["cam1"]
    p0 = {k: TrackPoint(c.xy[2 * k], [("cam1", 2 * k)]) for k in range(30)}
    p1 = {k: TrackPoint(c.xy[2 * k + 1], [("cam1", 2 * k + 1)]) for k in range(30)}
    e0, e1 = identity.track_embedding(p0, cams), identity.track_embedding(p1, cams)
    assert e0["n"] == 30 and np.isclose(np.linalg.norm(e0["mean"]), 1.0)
    assert identity.app_distance(e0, e0) == pytest.approx(0.0, abs=1e-6)
    assert identity.app_distance(e0, e1) == pytest.approx(1.0, abs=1e-6)
    assert identity.app_distance(e0, None) == 0.5
    assert identity.track_embedding({0: p0[0]}, cams) is None


def test_fill_gaps_and_smooth():
    line = {k: TrackPoint(np.array([1.0 + 0.3 * k, 5.0 - 0.1 * k]), [("cam1", k)]) for k in range(40) if not 10 < k < 20}
    filled = identity.fill_gaps(line, RATE)
    assert sorted(filled) == list(range(40)) and filled[15].boxes == []
    assert np.allclose(filled[15].xy, [1.0 + 0.3 * 15, 5.0 - 0.1 * 15])
    assert len(identity.fill_gaps(line, RATE, max_gap_s=0.5)) == len(line)
    sm = identity.smooth(line)
    assert sm.keys() == line.keys() and sm[5].boxes == [("cam1", 5)]
    assert np.allclose(sm[5].xy, line[5].xy)                                # a straight line is kept


# ---------------------------------------------------------------------------------------------------------------
# research parity on the cached match (opt-in)
# ---------------------------------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(os.environ.get("FUTSAL_RESEARCH_PARITY") != "1"
                    or not os.path.exists(os.path.join(ROOT, "experiments", "cache", "meta.json")),
                    reason="set FUTSAL_RESEARCH_PARITY=1 (needs experiments/cache)")
def test_build_tracklets_matches_research_xview():
    import copy
    import sys
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import xview
    from harness import Data
    data = Data()
    cams = {}
    for name, c in data.cams.items():             # research team labels Y / N -> A / B
        a = copy.copy(c)
        a.team = np.array([{"Y": "A", "N": "B"}.get(x, "") for x in c.team], dtype="<U1")
        a.frame_wh = (1920, 1080)
        cams[name] = a
    key = lambda tr: frozenset((k, cb) for k, p in tr.items() for cb in p.boxes)
    ref = {key(t) for t in xview.xview(data, "conservative")}
    got = {key(t) for t in tracklets.build_tracklets(cams, rate=data.rate, frames=range(data.n))}
    assert ref == got
