"""Closed-set identity assignment (pipeline.link_closed) and RTS position refinement (pipeline.refine).

Synthetic scene: one camera over a 40 x 20 m pitch, 5 v 5 players moving smoothly around their own areas (never
closer than ~3 m), each with its own ReID look, chopped into pure tracklets of 2-8 s with short gaps plus a few
sub-second pieces, the way the conservative builder leaves them. One team-A player is substituted half way.
The research parity check (final_d on experiments/cache) is opt-in: FUTSAL_RESEARCH_PARITY=1.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from futsal.court import Court
from futsal.homography import Calibration
from futsal.pipeline import link_closed as lc
from futsal.pipeline.boxes import TrackPoint, build_cam_boxes
from futsal.pipeline.detect import Detection
from futsal.pipeline.refine import RefineParams, refine_positions, rts_smooth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "experiments", "cache")
COURT = Court()
RATE = 10.0
N = 600                                    # 60 s
M1 = np.array([[40.0, 8.0, 160.0], [0.0, -25.0, 950.0], [0.0, 0.004, 1.0]])
CAL = Calibration(np.linalg.inv(M1), 1.0)
HOMES = [(6, 5), (6, 15), (14, 10), (22, 4), (22, 16), (34, 5), (34, 15), (26, 10), (18, 4), (18, 16)]
TEAM = ["A"] * 5 + ["B"] * 5
SUB_AT = 300                               # player 4 (team A) goes off, player 10 comes on in the same area
N_PEOPLE = 11


def box_of(xy) -> tuple[float, float, float, float]:
    u, v = CAL.to_image(np.asarray(xy, float).reshape(1, 2))[0]
    h = 0.16 * v
    return (u - 0.2 * h, v - h, u + 0.2 * h, v)


def path(j: int, k: int) -> np.ndarray:
    """Smooth wandering (<= ~2.5 m/s) within 1.4 m of the player's home."""
    home = np.array(HOMES[4 if j == 10 else j], float)
    ph = 0.7 * j
    return home + 1.4 * np.array([np.sin(0.21 * k / RATE * 2 + ph), np.cos(0.17 * k / RATE * 2 + 2 * ph)])


def on_pitch(j: int, k: int) -> bool:
    return (j != 4 or k < SUB_AT) and (j != 10 or k >= SUB_AT + 20)


def scene(seed: int = 5):
    """(cams, tracklets, truth) with truth[i] = person of box i."""
    rng = np.random.default_rng(seed)
    looks = rng.normal(size=(N_PEOPLE, 512))
    looks /= np.linalg.norm(looks, axis=1, keepdims=True)
    dets, who = [], []
    for k in range(N):
        for j in range(N_PEOPLE):
            if not on_pitch(j, k):
                continue
            b = box_of(path(j, k))
            e = looks[j] + rng.normal(0, 0.02, 512)
            dets.append(Detection(k, k / RATE, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b,
                                  (e / np.linalg.norm(e)).astype(np.float32)))
            who.append(j)
    cam = build_cam_boxes("cam1", dets, CAL, 0.0, 0.0, 0.0, RATE, COURT)
    truth = np.array(who)
    cam.team = np.array([TEAM[4 if j == 10 else j] for j in truth], dtype="<U1")
    cams = {"cam1": cam}
    tracklets = []
    for j in range(N_PEOPLE):
        idx = {int(cam.k[i]): int(i) for i in np.flatnonzero(truth == j)}
        ks = sorted(idx)
        q = 0
        while q < len(ks):
            n = int(rng.integers(5, 10)) if rng.random() < 0.15 else int(rng.integers(20, 80))
            piece = ks[q:q + n]
            tracklets.append({k: TrackPoint(cam.xy[idx[k]].copy(), [("cam1", idx[k])]) for k in piece})
            q += n + int(rng.integers(2, 8))
    return cams, tracklets, truth


@pytest.fixture(scope="module")
def solved():
    cams, T, truth = scene()
    st = lc.ClosedStats()
    out = lc.assign_identities(T, cams, RATE, stats=st)
    return cams, T, truth, out, st


def people(track, truth) -> set:
    return {int(truth[i]) for p in track.values() for _, i in p.boxes}


def boxes(tracks) -> list:
    return [(k,) + cb for t in tracks for k, p in t.items() for cb in p.boxes]


def pair_f1(a: list, b: list) -> float:
    """Pairwise box co-assignment F1 of b against a (a box missing from one side is a singleton there)."""
    from collections import Counter
    la = {x: q for q, t in enumerate(a) for x in boxes([t])}
    lb = {x: q for q, t in enumerate(b) for x in boxes([t])}
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(n) for n in Counter((la[x], lb[x]) for x in la.keys() & lb.keys()).values())
    pa = sum(c2(n) for n in Counter(la.values()).values())
    pb = sum(c2(n) for n in Counter(lb.values()).values())
    p, r = tp / max(pb, 1), tp / max(pa, 1)
    return 2 * p * r / max(p + r, 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# assign_identities
# ---------------------------------------------------------------------------------------------------------------

def test_defaults_are_the_tuned_final_d_values():
    p = lc.ClosedParams()
    assert (p.coverage, p.attach_gap_s, p.slots, p.spare, p.on_court) == (0.7, 2.0, 5, 1, 5)
    cams = {"cam1": None, "cam2": None}
    assert lc.default_border_exempt(cams) == {"cam1": ("top",), "cam2": ("top",)}


def test_assign_identities_recovers_every_player_purely(solved):
    cams, T, truth, out, st = solved
    used = boxes(out)
    assert len(used) == len(set(used))                                  # never a box twice
    assert set(used) <= set(boxes(T))
    assert all(list(t) == sorted(t) for t in out)
    assert all(len(people(t, truth)) == 1 for t in out)                  # every identity is one person
    assert sorted(j for t in out for j in people(t, truth)) == list(range(N_PEOPLE))
    assert len(used) >= 0.97 * len(boxes(T))
    # each identity carries its slot's team; the substitute took team A's spare slot
    for t, tm in zip(out, st.output_team):
        (j,) = people(t, truth)
        assert tm == TEAM[4 if j == 10 else j]
    assert st.reach_violations == 0 and not st.unpinned and st.exact_5v5_filled > 0.9
    assert st.spare_samples[5] > 0 and st.spare_samples[11] == 0
    # gap fills carry no boxes and stay within the identity's span
    for t in out:
        ks = sorted(t)
        assert ks == list(range(ks[0], ks[-1] + 1))                       # all gaps here are < 3 s
        assert all(np.all(np.isfinite(p.xy)) for p in t.values())


def test_assign_identities_never_shows_six_of_a_team():
    cams, T, truth = scene(seed=11)
    st = lc.ClosedStats()
    out = lc.assign_identities(T, cams, RATE, stats=st)
    cnt = {"A": np.zeros(N, int), "B": np.zeros(N, int)}
    for t, tm in zip(out, st.output_team):
        cnt[tm][list(t)] += 1
    assert cnt["A"].max() <= 5 and cnt["B"].max() <= 5


def test_assign_identities_edge_cases(solved):
    cams = solved[0]
    assert lc.assign_identities([], cams, RATE) == []
    assert lc.assign_identities([{}, {}], cams, RATE) == []
    with pytest.raises(ValueError):
        lc.assign_identities(solved[1], cams, RATE, border_exempt={"cam1": ("middle",)})


def test_split_jumps_and_duplicates():
    cams, T, truth = scene()
    cam = cams["cam1"]

    def person(j, ks):
        idx = {int(cam.k[i]): int(i) for i in np.flatnonzero(truth == j)}
        return {k: TrackPoint(cam.xy[idx[k]].copy(), [("cam1", idx[k])]) for k in ks}
    a = person(0, range(0, 40))
    joined = {**person(0, range(0, 20)), **person(9, range(20, 40))}    # 10+ m in 0.1 s: two people
    cut = lc.split_jumps([joined, a], cams, RATE)
    assert [(min(t), max(t)) for t in cut] == [(0, 19), (20, 39), (0, 39)]
    assert [people(t, truth) for t in cut] == [{0}, {9}, {0}]
    # the same tracklet twice: a duplicate pair (same spot, same time, same team)
    dup = lc.duplicate_pairs([a, dict(a)], ["A", "A"], lc.ClosedParams(), 0.5)
    assert dup == [(0, 1, len(a))]
    assert lc.duplicate_pairs([a, dict(a)], ["A", "B"], lc.ClosedParams(), 0.5) == []


# ---------------------------------------------------------------------------------------------------------------
# assign_identities_windowed
# ---------------------------------------------------------------------------------------------------------------

def test_windowed_with_one_window_is_the_single_solve(solved):
    cams, T, truth, out, _ = solved
    got = lc.assign_identities_windowed(T, cams, RATE, window_s=120.0, overlap_s=30.0)
    key = lambda tr: [sorted(boxes([t])) for t in tr]
    assert sorted(key(got)) == sorted(key(out))


def test_windowed_chains_windows_into_the_same_identities(solved):
    cams, T, truth, single, _ = solved
    st = lc.WindowedStats()
    out = lc.assign_identities_windowed(T, cams, RATE, window_s=25.0, overlap_s=10.0, stats=st)
    assert st.windows == [(0, 249), (150, 399), (300, 549), (450, 599)]
    assert st.cuts == [200, 350, 500]
    used = boxes(out)
    assert len(used) == len(set(used)) and set(used) <= set(boxes(T))
    assert all(list(t) == sorted(t) for t in out)
    assert all(len(people(t, truth)) == 1 for t in out)
    # every player stays one identity across the junctions (the substitute too, from his first window on)
    assert sorted(j for t in out for j in people(t, truth)) == list(range(N_PEOPLE))
    assert pair_f1(single, out) > 0.99
    assert all(a > 0.9 for a in st.agreement)
    # player 4 goes off at frame 300 and 10 comes on at 320, both inside window 1: 10 starts a new identity at the
    # first junction, 4's identity has no continuation at the second
    assert st.new_ids == [1, 0, 0] and st.ended == [0, 1, 0]
    assert [st.output_team[q] for q, t in enumerate(out) if people(t, truth) == {10}] == ["A"]


def test_windowed_rejects_bad_windows(solved):
    cams, T = solved[0], solved[1]
    with pytest.raises(ValueError):
        lc.assign_identities_windowed(T, cams, RATE, window_s=10.0, overlap_s=10.0)
    assert lc.assign_identities_windowed([], cams, RATE) == []


def test_windows_absorb_a_short_tail():
    # the research cache spans grid frames 0..3601: one 360 s solve, not 360 s + a window owning the last 30 s
    assert lc._windows(0, 3601, 3600, 3000) == [(0, 3601)]
    assert lc._windows(0, 3900, 3600, 3000) == [(0, 3599), (3000, 3900)]
    assert lc._windows(0, 599, 250, 150) == [(0, 249), (150, 399), (300, 549), (450, 599)]


# ---------------------------------------------------------------------------------------------------------------
# refine_positions
# ---------------------------------------------------------------------------------------------------------------

def _noisy_lines(seed: int = 2):
    """Constant-velocity players measured with 0.25 m foot-point noise; returns (cams, tracks, truth xy per point)."""
    rng = np.random.default_rng(seed)
    starts = [((5, 5), (2.0, 0.5)), ((30, 15), (-1.5, -0.8)), ((10, 16), (3.0, -0.2))]
    dets, true = [], []
    for k in range(200):
        for j, (p0, v) in enumerate(starts):
            xy = np.array(p0, float) + np.array(v) * k / RATE
            b = box_of(xy + rng.normal(0, 0.25, 2))
            dets.append(Detection(k, k / RATE, (b[0] + b[2]) / 2, b[3], 0.9, None, b[3] - b[1], None, b, None))
            true.append((j, k, xy))
    cam = build_cam_boxes("cam1", dets, CAL, 0.0, 0.0, 0.0, RATE, COURT)
    tracks = [{} for _ in starts]
    truth = [{} for _ in starts]
    for i, (j, k, xy) in enumerate(true):
        if j == 1 and 80 <= k < 90:
            continue                                                     # a 1 s hole, filled below without boxes
        if j == 2 and 100 <= k < 150:
            continue                                                     # a 5 s hole: two filter segments
        tracks[j][k] = TrackPoint(cam.xy[i].copy(), [("cam1", i)])
        truth[j][k] = xy
    for k in range(80, 90):                                              # linear fill, as fill_gaps would
        s = (k - 79) / 11
        tracks[1][k] = TrackPoint((1 - s) * tracks[1][79].xy + s * tracks[1][90].xy, [])
        truth[1][k] = np.array(starts[1][0], float) + np.array(starts[1][1]) * k / RATE
    return {"cam1": cam}, tracks, truth


def test_refine_positions_reduces_error_and_keeps_identities():
    cams, tracks, truth = _noisy_lines()
    out = refine_positions(tracks + [{}], cams, RATE)
    assert len(out) == len(tracks)                                       # empty tracks dropped, order kept
    raw_err, ref_err = [], []
    for t, r, tr in zip(tracks, out, truth):
        assert list(r) == sorted(t)                                      # same frames, in order
        assert all(r[k].boxes == t[k].boxes for k in t)                  # same boxes: identities intact
        assert all(np.all(np.isfinite(p.xy)) for p in r.values())
        raw_err += [np.linalg.norm(t[k].xy - tr[k]) for k in t]
        ref_err += [np.linalg.norm(r[k].xy - tr[k]) for k in t]
    raw, ref = np.sqrt(np.mean(np.square(raw_err))), np.sqrt(np.mean(np.square(ref_err)))
    assert ref < 0.5 * raw
    # speeds: noisy differences of the raw points vs the smooth refined ones (true speeds 2.06 / 1.70 / 3.01 m/s)
    sp = lambda t: np.linalg.norm(np.diff(np.stack([t[k].xy for k in range(0, 80)]), axis=0), axis=1) * RATE
    assert abs(np.median(sp(out[0])) - 2.06) < 0.15 and np.median(sp(tracks[0])) > 3.0


def test_rts_smooth_edge_cases():
    cams, tracks, _ = _noisy_lines()
    assert rts_smooth({}, cams, RATE) == {}
    fills = {k: TrackPoint(np.array([1.0 + k, 2.0]), []) for k in range(5)}      # no measurements: unchanged
    got = rts_smooth(fills, cams, RATE)
    assert all(np.array_equal(got[k].xy, fills[k].xy) for k in fills)
    one = {7: tracks[0][7]}                                                      # a single measured point
    got = rts_smooth(one, cams, RATE, RefineParams())
    assert list(got) == [7] and np.all(np.isfinite(got[7].xy)) and got[7].boxes == one[7].boxes


# ---------------------------------------------------------------------------------------------------------------
# Research parity on the cached match (opt-in, ~1 min)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FUTSAL_RESEARCH_PARITY") != "1"
                    or not os.path.exists(os.path.join(CACHE, "meta.json"))
                    or not os.path.exists(os.path.join(ROOT, "experiments", "results", "final_d.json")),
                    reason="set FUTSAL_RESEARCH_PARITY=1 (needs experiments/cache and results/final_d.json)")
def test_assign_identities_reproduces_research_final_d():
    """cache -> CamBoxes -> align_cameras -> build_tracklets -> assign_identities (cam2 top exemption) gives
    experiments/results/final_d.json: same boxes per identity, positions within the file's 3-decimal rounding.
    Teams from the research yellow-bib rule (Y -> A, N -> B), as in the build_tracklets parity test."""
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from harness import Data, load_tracks
    from futsal.pipeline.align import align_cameras
    from futsal.pipeline.boxes import cams_from_cache
    from futsal.pipeline.tracklets import build_tracklets
    data = Data()
    cams, meta = cams_from_cache(CACHE, ROOT)
    align_cameras(cams, COURT, k_range=(0, data.n))
    for name, cam in cams.items():
        cam.team = np.array([{"Y": "A", "N": "B"}.get(x, "") for x in data.cams[name].team], dtype="<U1")
    # the research code had neither the appearance override of the pairing team gate nor the camera-conflict vote
    T = build_tracklets(cams, rate=float(meta["rate"]), frames=range(data.n), team_override_sim=None)
    out = lc.assign_identities(T, cams, float(meta["rate"]), lc.ClosedParams(team_camera_conflict=False),
                               border_exempt={"cam2": ("top",)})
    ref = load_tracks(os.path.join(ROOT, "experiments", "results", "final_d.json"))
    key = lambda t: frozenset((k, cb) for k, p in t.items() for cb in p.boxes)
    assert [key(t) for t in out] == [key(t) for t in ref]
    for t, r in zip(out, ref):
        assert list(t) == list(r)
        assert max(float(np.abs(t[k].xy - r[k].xy).max()) for k in t) <= 5e-4 + 1e-9
