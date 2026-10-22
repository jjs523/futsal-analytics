"""Open-set identity linking (pipeline.link_gta), its tracklet source (pipeline.sources) and run.build_tracks(ids='v2gta').

Synthetic scenes, no video, no GPU, no boxmot (the BoostTrack proposal test skips without it):
  * the closed-set test scene of test_link_closed (5 v 5 + a substitute, one camera, pure 2-8 s tracklets);
  * hand-made micro scenes for the hard constraints: a team change on one path, a chain of one-frame debris, one
    player's two camera views as separate tracklets;
  * the two-camera end-to-end scene of test_pipeline_v2 for the v2gta wiring.
The research parity check (final_b2 on experiments/cache) is opt-in: FUTSAL_RESEARCH_PARITY=1 (~1 min).
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")

from futsal.pipeline import link_gta as lg
from futsal.pipeline import sources
from futsal.pipeline.boxes import TrackPoint, build_cam_boxes
from futsal.pipeline.detect import Detection
from futsal.pipeline.identity import clean_masks, team_vote
from futsal.pipeline.run import IDS, LINKERS, build_tracks
from test_link_closed import CAL, COURT, N_PEOPLE, RATE, box_of, boxes, pair_f1, people, scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "experiments", "cache")


def check_identities(out, inputs):
    """Invariants of any linker output: no box twice, only input boxes (each at its own frame), frames in time
    order, finite positions."""
    used = boxes(out)
    assert len(used) == len(set(used))
    assert set(used) <= set(boxes(inputs))
    for t in out:
        assert list(t) == sorted(t)
        assert all(np.all(np.isfinite(p.xy)) for p in t.values())


@pytest.fixture(scope="module")
def linked():
    cams, T, truth = scene()
    st = lg.GtaStats()
    out = lg.link_tracklets(T, cams, RATE, stats=st)
    return cams, T, truth, out, st


# ---------------------------------------------------------------------------------------------------------------
# link_tracklets on the 5 v 5 scene
# ---------------------------------------------------------------------------------------------------------------

def test_defaults_are_the_final_b2_values():
    p = lg.GtaParams()
    assert p.min_len_s == 5.0 and p.merge_thr == 0.35 and p.long_gap_s == 8.0 and p.long_thr == 0.25
    assert p.vmax == 9.0 and p.slack == 1.5 and p.w_motion == 0.05 and p.linkage == "average"
    assert p.strict_debris and (p.strict_gap_s, p.strict_margin) == (0.5, 2.0)
    assert (p.debris_len_s, p.debris_gap_s, p.noapp_margin) == (1.0, 1.5, 1.0)
    assert p.team_camera_conflict and p.smooth == "both" and p.fill_gap_s == 3.0


def test_link_tracklets_recovers_every_player_purely(linked):
    cams, T, truth, out, st = linked
    check_identities(out, T)
    assert all(len(people(t, truth)) == 1 for t in out)                  # purity: one person per identity
    assert sorted(j for t in out for j in people(t, truth)) == list(range(N_PEOPLE))   # and one identity each
    assert len(boxes(out)) == len(boxes(T))                              # nothing dropped here (no debris)
    assert [min(t) for t in out] == sorted(min(t) for t in out)          # in order of first frame
    # the open set has no slots: the substitute (10) is just another identity, with his own team
    team_of = {next(iter(people(t, truth))): tm for t, tm in zip(out, st.output_team)}
    assert team_of[10] == "A" and team_of[4] == "A" and team_of[9] == "B"
    for t in out:                                                        # all gaps here are < 3 s: filled, no boxes
        ks = sorted(t)
        assert ks == list(range(ks[0], ks[-1] + 1))
    filled = {(k, j) for t in out for k, p in t.items() if not p.boxes for j in people(t, truth)}
    assert filled and not ({(k, j) for t in T for k in t for j in people(t, truth)} & filled)
    assert st.inputs == len(T) and st.outputs == len(out) == N_PEOPLE and st.merges_app == len(T) - N_PEOPLE


def test_link_tracklets_edge_cases(linked):
    cams, T = linked[0], linked[1]
    assert lg.link_tracklets([], cams, RATE) == []
    assert lg.link_tracklets([{}, {}], cams, RATE) == []
    with pytest.raises(ValueError):
        lg.link_tracklets(T, cams, RATE, border_exempt={"cam1": ("middle",)})
    with pytest.raises(ValueError):
        lg.link_tracklets(T, cams, RATE, replace(lg.GtaParams(), linkage="single"))
    # min_len_s drops short identities: one 2 s tracklet alone
    short = [t for t in T if 20 <= len(t) < 30][:1]
    st = lg.GtaStats()
    assert lg.link_tracklets(short, cams, RATE, stats=st) == [] and st.dropped_points == len(short[0])
    assert len(lg.link_tracklets(short, cams, RATE, replace(lg.GtaParams(), min_len_s=1.0))) == 1


def test_medoid_linkage_and_no_smoothing_give_the_same_identities(linked):
    cams, T, truth, out, _ = linked
    med = lg.link_tracklets(T, cams, RATE, replace(lg.GtaParams(), linkage="medoid", smooth="none"))
    key = lambda tr: sorted(sorted(boxes([t])) for t in tr)
    assert key(med) == key(out)


# ---------------------------------------------------------------------------------------------------------------
# Hard constraints and debris on micro scenes
# ---------------------------------------------------------------------------------------------------------------

def _look(rng, base=None, noise=0.02):
    v = (rng.normal(size=512) if base is None else base) + rng.normal(0, noise, 512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def micro(rows_by_cam: dict) -> dict:
    """{cam: [(k, xy, reid, conf, team)]} -> {cam: CamBoxes}, box i = row i (one calibration for all cameras)."""
    cams = {}
    for cam, rows in rows_by_cam.items():
        dets = []
        for k, xy, emb, conf, _ in rows:
            b = box_of(xy)
            dets.append(Detection(k, k / RATE, (b[0] + b[2]) / 2, b[3], conf, None, b[3] - b[1], None, b, emb))
        c = build_cam_boxes(cam, dets, CAL, 0.0, 0.0, 0.0, RATE, COURT)
        c.team = np.array([r[4] for r in rows], dtype="<U1")
        cams[cam] = c
    return cams


def piece(cams, cam: str, idx) -> dict:
    return {int(cams[cam].k[i]): TrackPoint(cams[cam].xy[i].copy(), [(cam, int(i))]) for i in idx}


def _handover(team_x: str, team_y: str):
    """X runs 3 s along a line, Y continues the same line 0.3 s later for 3 s; both look exactly alike."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=512)
    rows = [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.9, team_x) for k in range(30)]
    rows += [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.9, team_y) for k in range(33, 63)]
    cams = micro({"cam1": rows})
    return cams, [piece(cams, "cam1", range(30)), piece(cams, "cam1", range(30, 60))]


def test_team_constraint_keeps_opposite_teams_apart():
    p = replace(lg.GtaParams(), min_len_s=1.0)
    for ty, n in (("B", 2), ("A", 1), ("", 1)):            # A vs B never merge; same team or undecided do
        cams, T = _handover("A", ty)
        st = lg.GtaStats()
        out = lg.link_tracklets(T, cams, RATE, p, stats=st)
        assert len(out) == n, ty
        check_identities(out, T)
        if n == 2:
            assert st.output_team == ["A", "B"]


def _debris_scene():
    """A 6 s track, then four 1-frame pieces 0.3, 0.7, 1.1 and 1.5 s after its end on the continuation of its line
    (no clean crops: one frame, low confidence). Only the first is within 0.5 s of the track itself."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=512)
    rows = [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.9, "A") for k in range(60)]
    deb = [63, 67, 71, 75]
    rows += [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.3, "") for k in deb]
    cams = micro({"cam1": rows})
    T = [piece(cams, "cam1", range(60))] + [piece(cams, "cam1", [60 + q]) for q in range(len(deb))]
    return cams, T, deb


def test_debris_attaches_only_next_to_the_identity_never_through_other_debris():
    cams, T, deb = _debris_scene()
    st = lg.GtaStats()
    (out,) = lg.link_tracklets(T, cams, RATE, stats=st)
    assert st.debris == 4 and st.debris_attached == 1
    assert [k for k in deb if out.get(k) and out[k].boxes] == [63]           # the chain stops at the first piece
    assert st.dropped_points == 3
    check_identities([out], T)
    # the research rule (strict_debris=False) chains every piece through the one attached before it
    st = lg.GtaStats()
    (chained,) = lg.link_tracklets(T, cams, RATE, replace(lg.GtaParams(), strict_debris=False), stats=st)
    assert st.debris_attached == 4 and all(chained[k].boxes for k in deb)


def test_debris_between_two_candidates_stays_out():
    """A piece that two identities explain about equally well (z within the 2-sigma margin) is not attached."""
    rng = np.random.default_rng(3)
    rows = [(k, (8 + 0.2 * k, 10.0), _look(rng), 0.9, "A") for k in range(60)]
    rows += [(k, (8 + 0.2 * k, 11.0), _look(rng), 0.9, "A") for k in range(60)]
    rows += [(62, (8 + 0.2 * 62, 10.5), _look(rng), 0.3, "")]
    cams = micro({"cam1": rows})
    T = [piece(cams, "cam1", range(60)), piece(cams, "cam1", range(60, 120)), piece(cams, "cam1", [120])]
    st = lg.GtaStats()
    out = lg.link_tracklets(T, cams, RATE, stats=st)
    assert len(out) == 2 and st.debris == 1 and st.debris_attached == 0 and ("cam1", 120) not in {
        cb for t in out for p in t.values() for cb in p.boxes}


def test_two_camera_views_of_one_player_are_fused():
    """cam1 sees a player for frames 0-49, cam2 for 20-69 (two tracklets the cross-view pairing left apart): one
    identity, two boxes per shared frame. A second player nearby stays apart."""
    rng = np.random.default_rng(4)
    base, other = rng.normal(size=512), rng.normal(size=512)
    path = lambda k: (10 + 0.1 * k, 8.0)
    near = lambda k: (10 + 0.1 * k, 12.0)
    r1 = [(k, path(k), _look(rng, base), 0.9, "A") for k in range(50)]
    r1 += [(k, near(k), _look(rng, other), 0.9, "A") for k in range(70)]
    r2 = [(k, path(k), _look(rng, base), 0.9, "A") for k in range(20, 70)]
    cams = micro({"cam1": r1, "cam2": r2})
    T = [piece(cams, "cam1", range(50)), piece(cams, "cam1", range(50, 120)), piece(cams, "cam2", range(50))]
    out = lg.link_tracklets(T, cams, RATE, replace(lg.GtaParams(), smooth="none"))
    assert len(out) == 2
    check_identities(out, T)
    a = next(t for t in out if ("cam2", 0) in t[20].boxes)
    assert sorted(a) == list(range(70)) and all(len(a[k].boxes) == 2 for k in range(20, 50))
    assert np.allclose(a[30].xy, path(30), atol=1e-6)                       # fused: both views agree here


def test_cluster_team_is_identity_team_vote():
    """The linker keeps the team vote as additive sums; it must equal identity.team_vote on the cluster's boxes,
    with and without the camera-conflict rule (cam1 reads the shirt as A, cam2 as B)."""
    rng = np.random.default_rng(5)
    rows1, rows2 = [], []
    for k in range(60):
        rows1.append((k, (10 + 0.1 * k, 9.0), _look(rng), 0.9, "A" if rng.random() < 0.9 else "B"))
        rows2.append((k, (10 + 0.1 * k, 9.0), _look(rng), 0.9, "B" if rng.random() < 0.85 else ""))
    cams = micro({"cam1": rows1, "cam2": rows2})
    fused = {k: TrackPoint(cams["cam1"].xy[k].copy(), [("cam1", k), ("cam2", k)]) for k in range(60)}
    T = [piece(cams, "cam1", range(30)), piece(cams, "cam2", range(30)), {k: fused[k] for k in range(30, 60)},
         {k: fused[k] for k in range(30, 35)}]
    clean = clean_masks(cams)
    for conflict in (True, False):
        L = lg.Linker(T, cams, RATE, replace(lg.GtaParams(), team_camera_conflict=conflict), clean)
        for cl in L.clusters.values():
            assert cl.team == team_vote(cl.track, cams, camera_conflict=conflict)[0]
        m = L._merge(L.clusters[0], L.clusters[2])                              # cam1-only + fused
        assert m.team == team_vote(m.track, cams, camera_conflict=conflict)[0]
        assert L._merge(m, L.clusters[1]).team == ("U" if conflict else team_vote(
            {**m.track, **T[1]}, cams, camera_conflict=False)[0])


# ---------------------------------------------------------------------------------------------------------------
# Windowed linking
# ---------------------------------------------------------------------------------------------------------------

def test_windowed_with_one_window_is_the_single_run(linked):
    cams, T, truth, out, st = linked
    wst = lg.WindowedStats()
    got = lg.link_tracklets_windowed(T, cams, RATE, window_s=120.0, overlap_s=30.0, stats=wst)
    assert len(wst.windows) == 1 and wst.output_team == st.output_team
    assert len(got) == len(out)
    for a, b in zip(got, out):
        assert list(a) == list(b) and all(a[k].boxes == b[k].boxes and np.array_equal(a[k].xy, b[k].xy) for k in a)


def test_windowed_chains_windows_into_the_same_identities(linked):
    cams, T, truth, single, _ = linked
    st = lg.WindowedStats()
    out = lg.link_tracklets_windowed(T, cams, RATE, window_s=25.0, overlap_s=10.0, stats=st)
    assert st.windows == [(0, 249), (150, 399), (300, 549), (450, 599)] and st.cuts == [200, 350, 500]
    assert len(st.per_window) == 4 and all(isinstance(w, lg.GtaStats) for w in st.per_window)
    check_identities(out, T)
    assert all(len(people(t, truth)) == 1 for t in out)
    assert sorted(j for t in out for j in people(t, truth)) == list(range(N_PEOPLE))
    assert pair_f1(single, out) > 0.99 and all(a > 0.9 for a in st.agreement)
    assert st.new_ids == [1, 0, 0] and st.ended == [0, 1, 0]               # the substitution, as for link_closed
    assert len(st.output_team) == len(out)
    with pytest.raises(ValueError):
        lg.link_tracklets_windowed(T, cams, RATE, window_s=10.0, overlap_s=10.0)
    assert lg.link_tracklets_windowed([], cams, RATE) == []


# ---------------------------------------------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------------------------------------------

def test_relink_joins_a_cut_and_refuses_an_ambiguous_one():
    """Margin rule: a player's track cut for 3 frames is re-joined; with a look-alike rival starting at the same
    spot and moment the cut stays (two equally good continuations)."""
    rng = np.random.default_rng(6)
    base = rng.normal(size=512)
    rows = [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.9, "A") for k in range(20)]
    rows += [(k, (8 + 0.2 * k, 10.0), _look(rng, base), 0.9, "A") for k in range(23, 43)]
    cams = micro({"cam1": rows})
    T = [piece(cams, "cam1", range(20)), piece(cams, "cam1", range(20, 40))]
    st = sources.RelinkStats()
    out = sources.relink(T, cams, RATE, stats=st)
    assert len(out) == 1 and st.linked == 1 and st.links[0][:3] == ("margin", 19, 23)
    rows += [(k, (8 + 0.2 * k, 10.3), _look(rng, base), 0.9, "A") for k in range(23, 43)]
    cams = micro({"cam1": rows})
    T = [piece(cams, "cam1", range(20)), piece(cams, "cam1", range(20, 40)), piece(cams, "cam1", range(40, 60))]
    st = sources.RelinkStats()
    assert len(sources.relink(T, cams, RATE, stats=st)) == 3 and st.linked == 0 and st.ambiguous >= 1


def test_absorb_short_gives_a_lone_box_to_its_host():
    rng = np.random.default_rng(7)
    rows1 = [(k, (8 + 0.2 * k, 10.0), _look(rng), 0.9, "A") for k in range(30)]
    rows2 = [(15, (8 + 0.2 * 15 + 0.2, 10.1), _look(rng), 0.9, "A"), (15, (8 + 0.2 * 15, 15.0), _look(rng), 0.9, "B")]
    cams = micro({"cam1": rows1, "cam2": rows2})
    T = [piece(cams, "cam1", range(30)), piece(cams, "cam2", [0]), piece(cams, "cam2", [1])]
    moved = []
    out = sources.absorb_short(T, cams, moved=moved)
    assert moved == [1] and len(out) == 2
    assert out[0][15].boxes == [("cam1", 15), ("cam2", 0)]                  # second view of the host
    assert boxes(out) and sorted(boxes(out)) == sorted(boxes(T))


def _v2_scene():
    from test_pipeline_v2 import CALS, COURT as C2, RATE as R2, SYNC, scene as v2_scene
    return v2_scene(), CALS, SYNC, C2, R2


def test_relink_source_partitions_the_boxes():
    from futsal.pipeline.align import align_cameras
    from futsal.pipeline.boxes import assign_teams
    from futsal.pipeline.run import clock, guess_frame_wh
    dets, cals, sync, court, rate = _v2_scene()
    cams = {}
    for cam, d in dets.items():
        off, drift = clock(sync.get(cam, 0.0))
        cams[cam] = build_cam_boxes(cam, d, cals[cam], off, drift, 0.0, rate, court, guess_frame_wh(d))
    assign_teams(cams)
    align_cameras(cams, court)
    st = sources.SourceStats()
    out = sources.relink_source(cams, rate, proposals=None, stats=st)
    used = boxes(out)
    want = {(int(c.k[i]), cam, int(i)) for cam, c in cams.items() for i in np.flatnonzero(c.in_court)}
    assert len(used) == len(set(used)) and set(used) == want                # every in-court box exactly once
    assert all(len({cam for cam, _ in p.boxes}) == len(p.boxes) for t in out for p in t.values())
    assert st.proposals is None and st.tracklets == len(out) and st.paired >= len(out)
    with pytest.raises(ValueError):
        sources.relink_source(cams, rate, proposals="hybridsort")


@pytest.mark.skipif(not sources.have_boxmot(), reason="BoostTrack proposals need boxmot")
def test_relink_source_with_boosttrack_proposals():
    from futsal.pipeline.boxes import assign_teams
    from futsal.pipeline.run import clock, guess_frame_wh
    dets, cals, sync, court, rate = _v2_scene()
    cams = {}
    for cam, d in dets.items():
        off, drift = clock(sync.get(cam, 0.0))
        cams[cam] = build_cam_boxes(cam, d, cals[cam], off, drift, 0.0, rate, court, guess_frame_wh(d))
    assign_teams(cams)
    prop = sources.boosttrack_tracklets(cams, "cam1", rate)
    assert prop and all(len(p.boxes) == 1 for t in prop for p in t.values())
    used = boxes(prop)
    assert len(used) == len(set(used))
    st = sources.SourceStats()
    out = sources.relink_source(cams, rate, stats=st)                       # 'auto' picks BoostTrack here
    assert st.proposals == "boosttrack" and len(boxes(out)) == len(set(boxes(out)))


# ---------------------------------------------------------------------------------------------------------------
# run.build_tracks(ids='v2gta')
# ---------------------------------------------------------------------------------------------------------------

def test_build_tracks_v2gta_end_to_end():
    from test_pipeline_v2 import PLAYERS, _match, truth, warp
    dets, cals, sync, court, rate = _v2_scene()
    assert "v2gta" in IDS and LINKERS == {"v2": "closed", "v2gta": "gta"}
    info: dict = {}
    ts = build_tracks(court, dets, cals, sync, fps_out=rate, ids="v2gta", info=info, source_params={"proposals": None})
    assert info["ids"] == "v2gta" and "fallback" not in info and info["source"].proposals is None
    assert len(ts.players) == len(PLAYERS) and info["identities"] == len(PLAYERS)
    assert len(info["solver"].windows) == 1 and isinstance(info["solver"].per_window[0], lg.GtaStats)
    who = _match(ts)
    lab = {p.id: p.team for p in ts.players}
    assert len({(lab[i], PLAYERS[j][0]) for i, j in who.items()}) == 2       # one consistent team mapping
    T = truth(np.arange(len(ts.players[0].xy)))
    for p in ts.players:                                                    # on its own player all along (no swap)
        j = who[p.id]
        ok = np.isfinite(p.xy[:, 0])
        assert ok.mean() > 0.95
        assert np.linalg.norm(p.xy[ok] - T[ok, j] - 0.5 * warp(T[ok, j]), axis=1).max() < 0.3
    # without ReID: the same fallback as v2
    info = {}
    build_tracks(court, {c: [replace(d, reid=None) for d in ds] for c, ds in dets.items()}, cals, sync,
                 fps_out=rate, ids="v2gta", info=info)
    assert info == {"fallback": "no ReID embeddings", "ids": "appearance"}


def test_trackview_accepts_v2gta(tmp_path, monkeypatch):
    """--ids v2gta parses, asks for the ReID embedder like v2 and reaches trackview.run with ids='v2gta'."""
    from futsal import trackview
    from futsal.pipeline import reid
    seen = {}
    monkeypatch.setattr(reid, "reid_embedder", lambda: "embedder")
    monkeypatch.setattr(trackview, "load_calibration", lambda *a, **kw: CAL)
    monkeypatch.setattr(trackview, "run", lambda *a, **kw: seen.update(kw) or {
        "tracker": kw["ids"], "ids": 0, "ids_over_half": 0, "events": 0})
    trackview.main(["--video", "a.mp4", "b.mp4", "--calib", "a.json", "b.json", "--start", "0", "--ids", "v2gta",
                    "--out", str(tmp_path)])
    assert seen["ids"] == "v2gta" and seen["embedder"] == "embedder" and seen["det_tag"].endswith("/0.1")


# ---------------------------------------------------------------------------------------------------------------
# Research parity on the cached match (opt-in, ~1 min)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FUTSAL_RESEARCH_PARITY") != "1"
                    or not os.path.exists(os.path.join(CACHE, "meta.json"))
                    or not os.path.exists(os.path.join(ROOT, "experiments", "results", "final_b2.json"))
                    or not sources.have_boxmot(),
                    reason="set FUTSAL_RESEARCH_PARITY=1 (needs experiments/cache, results/final_b2.json, boxmot)")
def test_link_tracklets_reproduces_research_final_b2(monkeypatch):
    """cache -> CamBoxes -> align_cameras -> relink_source (BoostTrack proposals) -> link_tracklets (strict debris,
    cam2 top exemption) gives experiments/results/final_b2.json: same boxes per identity in the same order, same
    frames, positions within the file's 3-decimal rounding. Teams from the research yellow-bib rule (Y -> A,
    N -> B); the switches the research code did not have are off (pair_views team override, team_vote camera
    conflict)."""
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from harness import Data, load_tracks
    from futsal.pipeline.align import align_cameras
    from futsal.pipeline.boxes import cams_from_cache
    monkeypatch.setenv("FUTSAL_LEGACY_GRID", "1")              # the stored result was made with the old grid rule
    data = Data()
    cams, meta = cams_from_cache(CACHE, ROOT, legacy_grid=True)
    align_cameras(cams, COURT, k_range=(0, data.n))
    for name, cam in cams.items():
        cam.team = np.array([{"Y": "A", "N": "B"}.get(x, "") for x in data.cams[name].team], dtype="<U1")
    rate = float(meta["rate"])
    T = sources.relink_source(cams, rate, frames=range(data.n), proposals="boosttrack", team_override_sim=None,
                              team_camera_conflict=False)
    st = lg.GtaStats()
    out = lg.link_tracklets(T, cams, rate, replace(lg.GtaParams(), team_camera_conflict=False),
                            border_exempt={"cam2": ("top",)}, stats=st)
    ref = load_tracks(os.path.join(ROOT, "experiments", "results", "final_b2.json"))
    key = lambda t: frozenset((k, cb) for k, p in t.items() for cb in p.boxes)
    assert [key(t) for t in out] == [key(t) for t in ref]
    for t, r in zip(out, ref):
        assert list(t) == list(r)
        assert max(float(np.abs(t[k].xy - r[k].xy).max()) for k in t) <= 5e-4 + 1e-9
    assert (len(T), st.debris_attached, st.outputs) == (1808, 283, 16)       # fix2's log line for final_b2
