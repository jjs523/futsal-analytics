"""Box-level tracking v2 end to end on a synthetic two-camera scene (no video, no GPU, no boxmot):

    Detections -> boxes.build_cam_boxes -> assign_teams -> align.align_cameras -> tracklets.build_tracklets
    and the identity blocks (team_vote, reachable / cannot_link, fill_gaps) on the result.

The scene: six players on a 40 x 20 m pitch filmed by two cameras from opposite long sides (simple homographies),
two team-A players crossing (their boxes overlap in both views), a few low-confidence boxes, spectator / clutter
boxes, and a camera-2 calibration that is off by a smooth known field. The research parity check runs when
experiments/cache is present and is skipped otherwise.
"""
import os
import sys

import numpy as np
import pytest

from futsal.court import Court
from futsal.homography import Calibration
from futsal.pipeline import identity, tracklets
from futsal.pipeline.align import align_cameras
from futsal.pipeline.boxes import CamBoxes, TrackPoint, assign_teams, build_cam_boxes, cams_from_cache
from futsal.pipeline.detect import FEAT_LEN, Detection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "experiments", "cache")
COURT = Court()
RATE = 10.0
START = 100.0                    # window start on the reference (cam1) clock, s
CAM2_OFFSET = 2.5                # cam2's clock runs 2.5 s behind
N = 80                           # grid frames (8 s)

# pitch -> image of a camera on the y < 0 long side; cam2 looks from the other side (pitch rotated by 180 degrees)
M1 = np.array([[40.0, 8.0, 160.0], [0.0, -25.0, 950.0], [0.0, 0.004, 1.0]])
M2 = M1 @ np.array([[-1.0, 0.0, 40.0], [0.0, -1.0, 20.0], [0.0, 0.0, 1.0]])
CALS = {"cam1": Calibration(np.linalg.inv(M1), 1.0), "cam2": Calibration(np.linalg.inv(M2), 1.0)}

# (start, end, team): 2 and 3 are team-A players crossing at (17, 10) half way through
PLAYERS = [((5, 5), (15, 5), "A"), ((30, 14), (22, 14), "B"), ((12, 8), (22, 12), "A"),
           ((12, 12), (22, 8), "A"), ((34, 4), (28, 7), "B"), ((4, 16), (12, 17), "B")]
CROSSING = {2, 3}
LOW_CONF = {("cam1", 0, k) for k in (30, 31, 32)}          # (camera, player, k) seen with conf 0.2 only


def warp(xy: np.ndarray) -> np.ndarray:
    """cam2's calibration error: a smooth, cubic offset field of 0.3 - 0.7 m."""
    x, y = xy[:, 0] / 40.0, xy[:, 1] / 20.0
    return np.stack([0.4 + 0.3 * x * x, -0.3 + 0.2 * x * y], 1)


def positions(k: int) -> np.ndarray:
    s = k / (N - 1)
    return np.array([(1 - s) * np.array(a, float) + s * np.array(b, float) for a, b, _ in PLAYERS])


def box_of(cal: Calibration, xy) -> tuple[float, float, float, float]:
    """Box standing on pitch position xy; nearer players (lower in the image) are taller."""
    u, v = cal.to_image(np.asarray(xy, float).reshape(1, 2))[0]
    h = 0.16 * v
    return (u - 0.2 * h, v - h, u + 0.2 * h, v)


def kit(team: str, rng) -> list[int]:
    """Quantised upper + lower body histogram: team A in yellow bibs (hue bins 1-2), team B in white."""
    up = np.full(15, 0.01)
    if team == "A":
        up[1:3] += 0.35
    else:
        up[14] += 0.6
    up[12] += 0.2
    up = np.abs(up + rng.normal(0, 0.02, 15))
    lo = np.abs(np.full(15, 1 / 15) + rng.normal(0, 0.01, 15))
    return [int(round(x * 255)) for x in np.concatenate([up / up.sum(), lo / lo.sum()])]


def scene(n_frames: int = N, misaligned: bool = True):
    """(cams, truth): truth[cam][i] = player of box i, -1 for clutter boxes."""
    rng = np.random.default_rng(7)
    ids = rng.normal(size=(len(PLAYERS) + 1, 512))
    ids /= np.linalg.norm(ids, axis=1, keepdims=True)
    cams, truth = {}, {}
    for cam, cal in CALS.items():
        dets, who = [], []
        for k in range(n_frames):
            t = START + k / RATE - (CAM2_OFFSET if cam == "cam2" else 0.0)
            fid = 3 * k if cam == "cam1" else 6 * k + 7
            pos = positions(k)
            if cam == "cam2" and misaligned:
                pos = pos + warp(pos)
            for j, xy in enumerate(pos):
                b = box_of(cal, xy)
                e = ids[j] + rng.normal(0, 0.01, 512)
                conf = 0.2 if (cam, j, k) in LOW_CONF else 0.9
                dets.append(Detection(fid, t, (b[0] + b[2]) / 2, b[3], conf, None, b[3] - b[1],
                                      kit(PLAYERS[j][2], rng), b, (e / np.linalg.norm(e)).astype(np.float32)))
                who.append(j)
            clutter = []
            if k in (50, 51):
                clutter.append(((38.5, 1.0), 0.15))       # low confidence: never starts a tracklet
            if k == 60:
                clutter.append(((20.0, 18.5), 0.05))      # below low_conf: ignored altogether
            if k % 10 == 0:
                clutter.append(((20.0, -4.0), 0.9))       # spectator off the pitch
            for xy, conf in clutter:
                b = box_of(cal, xy)
                dets.append(Detection(fid, t, (b[0] + b[2]) / 2, b[3], conf, None, b[3] - b[1], None, b,
                                      (ids[-1] + 0.0).astype(np.float32)))
                who.append(-1)
        offset = CAM2_OFFSET if cam == "cam2" else 0.0
        cams[cam] = build_cam_boxes(cam, dets, cal, offset, 0.0, START, RATE, COURT)
        truth[cam] = np.array(who)
    return cams, truth


@pytest.fixture(scope="module")
def built():
    """The scene after teams, alignment and tracklets, plus the per-camera stats (built once: ~1 s)."""
    cams, truth = scene()
    teams = assign_teams(cams)
    raw = {c: cams[c].xy.copy() for c in cams}
    al = align_cameras(cams, COURT)
    stats = {c: tracklets.CutStats() for c in cams}
    per = {c: tracklets.conservative_tracklets(cams, c, RATE, stats=stats[c]) for c in cams}
    xst = tracklets.XviewStats()
    out = tracklets.pair_views(per["cam1"], per["cam2"], cams, stats=xst)
    return {"cams": cams, "truth": truth, "teams": teams, "raw": raw, "align": al, "per": per, "stats": stats,
            "out": out, "xstats": xst}


def players_of(track, truth) -> set:
    return {int(truth[cam][i]) for p in track.values() for cam, i in p.boxes}


def boxes_of(tracks) -> list:
    return [cb for t in tracks for p in t.values() for cb in p.boxes]


# ---------------------------------------------------------------------------------------------------------------
# CamBoxes and the frame grid
# ---------------------------------------------------------------------------------------------------------------

def test_cam_boxes_grid_puts_both_cameras_on_the_reference_clock(built):
    cams = built["cams"]
    for cam in cams.values():
        assert isinstance(cam, CamBoxes) and sorted(cam.by_k) == list(range(N))
        assert cam.reid.dtype == np.float32 and cam.color.shape == (len(cam), FEAT_LEN)
        assert np.all((cam.color >= 0) & (cam.color <= 1))
        # one source frame per grid frame, every box of a source frame on the same k
        for k, idx in cam.by_k.items():
            assert len(set(cam.fid[idx])) == 1 and np.all(cam.k[idx] == k)
        spectators = cam.xy_raw[:, 1] < -3
        assert spectators.any() and not cam.in_court[spectators].any()
    # cam1 and cam2 boxes of the same player at the same k are the same moment
    c1, c2 = cams["cam1"], cams["cam2"]
    assert np.allclose(c1.t[c1.by_k[40]], c2.t[c2.by_k[40]] + CAM2_OFFSET)


def test_cam_boxes_grid_pushes_a_rounding_collision_to_the_next_free_frame():
    # source frames every 0.25 s at a 4 Hz grid, all exactly half way between two grid frames: round-half-to-even
    # puts frames 1 and 2 both on k = 2 (and 3, 4 on k = 4), so the later of each pair moves on by one
    dets = [Detection(f, 50.125 + 0.25 * f, 900.0, 800.0, 0.9, None, 120.0, None, (876.0, 680.0, 924.0, 800.0))
            for f in range(6) for _ in range(2)]                    # two boxes per source frame
    cam = build_cam_boxes("cam1", dets, CALS["cam1"], 1.0, 0.0, 51.0, 4.0, COURT)
    assert cam.k.tolist() == [0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert {k: v.tolist() for k, v in cam.by_k.items()} == {0: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7], 5: [8, 9],
                                                            6: [10, 11]}
    assert cam.boxes_at(3).tolist() == [4, 5] and len(cam.boxes_at(1)) == 0
    # drift stretches the camera clock: 0.1 % fast over 100 s is one more 10 Hz frame
    one = [Detection(0, 100.0, 900.0, 800.0, 0.9, None, 120.0, None, (876.0, 680.0, 924.0, 800.0))]
    assert build_cam_boxes("c", one, CALS["cam1"], 0.0, 0.0, 0.0, 10.0, COURT).k.tolist() == [1000]
    assert build_cam_boxes("c", one, CALS["cam1"], 0.0, 0.001, 0.0, 10.0, COURT).k.tolist() == [1001]


# ---------------------------------------------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------------------------------------------

def test_assign_teams_recovers_the_two_kits(built):
    cams, truth = built["cams"], built["truth"]
    assert built["teams"]["n_fit"] >= 2 * N * len(PLAYERS) - 10
    lab = np.concatenate([cams[c].team[truth[c] >= 0] for c in cams])
    true = np.concatenate([[PLAYERS[j][2] for j in truth[c][truth[c] >= 0]] for c in cams])
    assert set(lab) <= {"A", "B", ""} and (lab != "").mean() > 0.99
    assert max((lab == true).mean(), (lab == np.where(true == "A", "B", "A")).mean()) > 0.99
    # boxes without a histogram (clutter) stay unlabelled
    assert all(set(cams[c].team[truth[c] < 0]) == {""} for c in cams)


# ---------------------------------------------------------------------------------------------------------------
# Cross-camera alignment
# ---------------------------------------------------------------------------------------------------------------

def test_align_cameras_recovers_a_known_smooth_offset(built):
    cams, truth, al = built["cams"], built["truth"], built["align"]
    assert al is not None and al["pairs"] >= 200
    assert 0.3 < al["median_before_m"] < 0.8 and al["median_after_m"] < 0.02
    c1, c2 = cams["cam1"], cams["cam2"]
    m1 = c1.in_court & (truth["cam1"] >= 0)
    true = c1.xy_raw[m1]                                          # cam1's calibration is exact in this scene
    # both cameras meet half way: cam1 moves by half of cam2's error, cam2 back by the other half
    assert np.allclose(c1.xy[m1] - true, 0.5 * warp(true), atol=0.01)
    m2 = c2.in_court & (truth["cam2"] >= 0)
    P = c2.xy_raw[m2]
    assert np.allclose(c2.xy[m2], P - 0.5 * warp(P), atol=0.02)
    # the same player's two boxes now agree
    of = lambda cam, k, j: cams[cam].xy[cams[cam].by_k[k][truth[cam][cams[cam].by_k[k]] == j]]
    assert max(np.linalg.norm(of("cam1", k, j) - of("cam2", k, j)) for k in range(0, N, 7) for j in range(len(PLAYERS))) < 0.03
    assert np.array_equal(built["raw"]["cam1"], c1.xy_raw)                       # xy_raw is never touched


# ---------------------------------------------------------------------------------------------------------------
# Per-camera conservative tracklets
# ---------------------------------------------------------------------------------------------------------------

def test_conservative_tracklets_are_pure_and_cut_at_the_crossing(built):
    truth = built["truth"]
    for cam, tracks in built["per"].items():
        assert built["stats"][cam].overlap > 0                    # the crossing players' boxes overlap
        assert all(len(players_of(t, truth)) == 1 for t in tracks)
        assert all(len(p.boxes) == 1 and p.boxes[0][0] == cam for t in tracks for p in t.values())
        whole = {j for t in tracks if len(t) == N for j in players_of(t, truth)}
        assert whole == {0, 1, 4, 5}                              # separate players: one tracklet end to end
        for j in CROSSING:                                        # crossing players: cut, never carried across
            assert sum(1 for t in tracks if players_of(t, truth) == {j}) >= 2
        assert -1 not in {j for t in tracks for j in players_of(t, truth)}     # clutter never makes a tracklet
    # the low-confidence boxes of player 0 continue its tracklet (ByteTrack's second pass)
    c1 = built["cams"]["cam1"]
    low = {int(i) for i in np.flatnonzero(c1.conf < 0.3) if truth["cam1"][i] == 0}
    assert len(low) == 3 and low <= {i for _, i in boxes_of(built["per"]["cam1"])}


def _walker(jump_at: int | None) -> dict[str, CamBoxes]:
    rng = np.random.default_rng(3)
    e = rng.normal(size=512)
    dets = []
    for k in range(40):
        xy = (8.0 + 0.12 * k + (5.0 if jump_at is not None and k >= jump_at else 0.0), 6.0)
        b = box_of(CALS["cam1"], xy)
        f = e + rng.normal(0, 0.01, 512)
        dets.append(Detection(k, k / RATE, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b,
                              (f / np.linalg.norm(f)).astype(np.float32)))
    return {"cam1": build_cam_boxes("cam1", dets, CALS["cam1"], 0.0, 0.0, 0.0, RATE, COURT)}


def test_conservative_tracklets_do_not_link_across_a_big_jump():
    cams = _walker(jump_at=None)
    assert [len(t) for t in tracklets.build_tracklets(cams, RATE)] == [40]
    cams = _walker(jump_at=20)                     # same look, but 5 m in 0.1 s: two people, or a detector swap
    out = tracklets.build_tracklets(cams, RATE)
    assert [(min(t), max(t)) for t in out] == [(0, 19), (20, 39)]
    assert identity.cannot_link(out[0], out[1], cams, RATE) and not identity.reachable(out[0], out[1], cams, RATE)


# ---------------------------------------------------------------------------------------------------------------
# Cross-view pairing
# ---------------------------------------------------------------------------------------------------------------

def test_pair_views_pairs_the_two_copies_and_partitions_the_boxes(built):
    cams, truth, out = built["cams"], built["truth"], built["out"]
    used = boxes_of(out)
    assert len(used) == len(set(used))                                    # no box twice
    assert set(used) == set(boxes_of(built["per"]["cam1"]) + boxes_of(built["per"]["cam2"]))   # none lost
    expected = {(c, int(i)) for c in cams for i in np.flatnonzero(cams[c].in_court & (cams[c].conf >= 0.3))}
    expected |= {("cam1", int(i)) for i in np.flatnonzero((cams["cam1"].conf == 0.2) & (truth["cam1"] == 0))}
    assert set(used) == expected
    for t in out:
        assert sorted(t) == list(t) and len(players_of(t, truth)) == 1
        for p in t.values():                                              # at most one box per camera per point
            assert len({c for c, _ in p.boxes}) == len(p.boxes)
            if len(p.boxes) == 2:                                         # fused: between the two views
                a, b = (cams[c].xy[i] for c, i in p.boxes)
                assert np.linalg.norm(p.xy - (a + b) / 2) <= np.linalg.norm(a - b) / 2 + 1e-9
    for j in (0, 1, 4, 5):                         # each separate player: one tracklet, both cameras at every frame
        mine = [t for t in out if players_of(t, truth) == {j}]
        assert len(mine) == 1 and len(mine[0]) == N
        assert all(sorted(c for c, _ in p.boxes) == ["cam1", "cam2"] for p in mine[0].values())
    for j in CROSSING:                             # the crossing players' long pieces before / after are fused too
        long = [t for t in out if players_of(t, truth) == {j} and len(t) >= 10]
        assert len(long) == 2 and all(len(p.boxes) == 2 for t in long for p in t.values())
    assert built["xstats"].paired_samples >= 4 * N


def test_build_tracklets_runs_both_stages(built):
    out = tracklets.build_tracklets(built["cams"], RATE)
    key = lambda tracks: {frozenset((k, cb) for k, p in t.items() for cb in p.boxes) for t in tracks}
    assert key(out) == key(built["out"])


# ---------------------------------------------------------------------------------------------------------------
# Identity blocks
# ---------------------------------------------------------------------------------------------------------------

def test_team_vote_on_the_scene(built):
    cams, truth = built["cams"], built["truth"]
    lab = {}
    for t in built["out"]:
        if len(t) >= 20:
            (j,) = players_of(t, truth)
            vote, share = identity.team_vote(t, cams)
            assert vote in ("A", "B") and (share >= 0.7 if vote == "A" else share <= 0.3)
            lab.setdefault(PLAYERS[j][2], set()).add(vote)
    assert len(lab) == 2 and all(len(v) == 1 for v in lab.values()) and lab["A"] != lab["B"]


def _cam(team: list[str], h: list[float]) -> dict[str, CamBoxes]:
    n = len(team)
    xyxy = np.array([[900.0, 800.0 - hh, 900.0 + 0.4 * hh, 800.0] for hh in h])
    z = np.zeros((n, 2))
    return {"c": CamBoxes(name="c", fid=np.arange(n), t=np.arange(n) / RATE, k=np.arange(n), xyxy=xyxy,
                          conf=np.full(n, 0.9), reid=None, color=None, foot=np.stack([xyxy[:, [0, 2]].mean(1), xyxy[:, 3]], 1), h=np.array(h, float),
                          w=0.4 * np.array(h, float), xy=z, xy_raw=z.copy(), sigma=np.full(n, 0.05),
                          in_court=np.ones(n, bool), team=np.array(team, dtype="<U1"), cal=CALS["cam1"],
                          by_k={i: np.array([i]) for i in range(n)})}


def test_team_vote_weights_big_boxes_and_ignores_small_or_blank_ones():
    track = lambda n: {k: TrackPoint(np.zeros(2), [("c", k)]) for k in range(n)}
    cams = _cam(["A"] * 8 + ["B"] * 2, [100.0] * 10)
    assert identity.team_vote(track(10), cams) == ("A", pytest.approx(0.8))
    cams = _cam(["A"] * 5 + ["B"] * 5, [150.0] * 5 + [50.0] * 5)          # the big (near) boxes say A
    assert identity.team_vote(track(10), cams) == ("A", pytest.approx(0.75))
    cams = _cam(["B"] * 3 + ["A"] * 7, [100.0] * 3 + [30.0] * 7)          # A boxes too small to count
    assert identity.team_vote(track(10), cams) == ("B", 0.0)
    cams = _cam(["A"] * 5 + ["B"] * 5, [100.0] * 10)
    assert identity.team_vote(track(10), cams)[0] == "U"
    vote, share = identity.team_vote(track(3), _cam(["", "", ""], [100.0] * 3))
    assert vote == "U" and np.isnan(share)


def test_reachable_and_cannot_link_basics():
    pt = lambda x, y: TrackPoint(np.array([x, y], float), [])          # no boxes: isotropic 0.5 m default
    a = {k: pt(10 + 0.5 * k, 10) for k in range(10)}                   # ends at x = 14.5 at k = 9
    near = {k: pt(16 + 0.5 * (k - 12), 10) for k in range(12, 20)}     # 1.5 m in 0.3 s: fine
    far = {k: pt(25, 10) for k in range(12, 20)}                       # 10.5 m in 0.3 s: no
    late_far = {k: pt(25, 10) for k in range(30, 40)}                  # 10.5 m in 2.1 s: possible again
    assert identity.reachable(a, near, {}, RATE) and not identity.cannot_link(a, near, {}, RATE)
    assert not identity.reachable(a, far, {}, RATE) and identity.cannot_link(a, far, {}, RATE)
    assert identity.reachable(a, late_far, {}, RATE)
    assert not identity.reachable(a, late_far, {}, RATE, vmax=2.0)
    assert identity.reachable(a, {}, {}, RATE)
    # time overlap: more than max_overlap shared frames
    overlapping = {k: pt(14 + 0.5 * (k - 8), 10) for k in range(8, 15)}
    assert identity.overlap(a, overlapping) == 2
    assert identity.cannot_link(a, overlapping, {}, RATE) and not identity.cannot_link(a, overlapping, {}, RATE, max_overlap=2,
                                                                                         team_a="U", team_b="U")
    # teams: A vs B never links, 'U' links with both
    assert identity.cannot_link(a, near, {}, RATE, team_a="A", team_b="B")
    assert not identity.cannot_link(a, near, {}, RATE, team_a="A", team_b="A")
    assert not identity.cannot_link(a, near, {}, RATE, team_a="U", team_b="B")


def test_fill_gaps_is_exact_on_constant_velocity():
    v = np.array([1.7, -0.6])                                          # m/s
    line = lambda k: np.array([3.0, 12.0]) + v * k / RATE
    track = {k: TrackPoint(line(k), [("cam1", k)]) for k in list(range(0, 10)) + list(range(25, 35)) + [60, 61]}
    filled = identity.fill_gaps(track, RATE, max_gap_s=2.0)
    assert sorted(filled) == list(range(0, 35)) + [60, 61]            # 15-frame gap filled, 25-frame gap kept
    for k in range(10, 25):
        assert filled[k].boxes == [] and np.allclose(filled[k].xy, line(k), atol=1e-9)
    assert all(filled[k] is track[k] for k in track)                   # observed points untouched
    # a single-sample end has no velocity of its own: the chord is used, still exact on a line
    two = {0: TrackPoint(line(0), [("c", 0)]), 8: TrackPoint(line(8), [("c", 1)])}
    assert all(np.allclose(p.xy, line(k), atol=1e-9) for k, p in identity.fill_gaps(two, RATE).items())
    # faster than vmax: the end velocities are clamped, but the endpoints are still met
    fast = {k: TrackPoint(np.array([k * 1.0, 0.0]), []) for k in (0, 1, 2, 10, 11, 12)}   # 10 m/s
    f = identity.fill_gaps(fast, RATE, vmax=8.0)
    assert sorted(f) == list(range(13)) and np.all(np.diff([f[k].xy[0] for k in range(13)]) > 0)


# ---------------------------------------------------------------------------------------------------------------
# Research parity on the cached match (runs whenever experiments/cache is there)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(CACHE, "meta.json"))
                    or not os.path.exists(os.path.join(ROOT, "experiments", "xview.py")),
                    reason="needs experiments/cache and the research code")
def test_build_tracklets_partition_matches_research_xview():
    """src path (cache -> CamBoxes -> align_cameras -> build_tracklets) gives the research xview partition box for box.
    Teams come from the research yellow-bib rule (Y -> A, N -> B): assign_teams is checked against that rule in
    test_boxes, and a handful of its different labels move the pair_views team gate."""
    pytest.importorskip("scipy")
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import xview
    from harness import Data
    data = Data()
    cams, meta = cams_from_cache(CACHE, ROOT)
    al = align_cameras(cams, COURT, k_range=(0, data.n))
    assert al is not None and al["pairs"] == data.alignment["pairs"]
    for name, cam in cams.items():
        assert np.array_equal(cam.k, data.cams[name].k) and np.array_equal(cam.in_court, data.cams[name].in_court)
        assert np.allclose(cam.xy, data.cams[name].xy, atol=1e-6, rtol=0)
        cam.team = np.array([{"Y": "A", "N": "B"}.get(x, "") for x in data.cams[name].team], dtype="<U1")
    key = lambda tracks: {frozenset((k, cb) for k, p in t.items() for cb in p.boxes) for t in tracks}
    ref = xview.xview(data, "conservative")
    got = tracklets.build_tracklets(cams, rate=float(meta["rate"]), frames=range(data.n))
    assert len(got) == len(ref) and key(got) == key(ref)
    used = boxes_of(got)
    assert len(used) == len(set(used))
