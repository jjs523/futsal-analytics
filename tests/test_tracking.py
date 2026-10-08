"""ID tracking: two same-team players meet and bounce apart (a V, not an X), so constant-velocity motion
alone predicts a swap. Different shorts colours must keep the identities."""
import numpy as np

from futsal.fusion import Fused, Observation
from futsal.pipeline import pitch_tracker
from futsal.sim import evaluate
from futsal.tracks import PlayerTrack, TrackSet

FPS = 10


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 8, 1 / FPS)
    meet = 4.0
    # both run towards (20, 10), touch, then turn back the way they came
    a = np.stack([20 - 3 * np.abs(t - meet), 10 + 0.2 * np.abs(t - meet)], 1)
    b = np.stack([20 + 3 * np.abs(t - meet), 10 - 0.2 * np.abs(t - meet)], 1)
    feat_a = np.r_[np.eye(15)[0], np.eye(15)[12]]           # same bib, black shorts
    feat_b = np.r_[np.eye(15)[0], np.eye(15)[14]]           # same bib, white shorts
    frames = []
    for i in range(len(t)):
        dets = []
        for xy, f in ((a[i], feat_a), (b[i], feat_b)):
            o = Observation(xy + rng.normal(0, 0.15, 2), 0.2, "cam1", team="A", feat=f + rng.uniform(0, 0.05, 30))
            dets.append(Fused(o.xy, o.sigma, [o]))
        if np.linalg.norm(a[i] - b[i]) < 0.6:                # merged into one box while touching
            dets = [Fused((a[i] + b[i]) / 2, 0.3, [Observation((a[i] + b[i]) / 2, 0.3, "cam1", team="A")])]
        frames.append(dets)
    truth = TrackSet(40, 20, FPS, [PlayerTrack(1, "A", a), PlayerTrack(2, "A", b)])
    return frames, truth


def test_appearance_keeps_ids_through_a_bounce():
    frames, truth = _scene()
    est = pitch_tracker.absorb_duplicates(pitch_tracker.associate(pitch_tracker.tracklets(frames, FPS, ambiguity=0.4), len(frames), FPS))
    s = evaluate.id_scores(truth, TrackSet(40, 20, FPS, est))
    assert s["tracks"] == 2 and s["idf1"] > 0.9


def test_motion_only_tracker_swaps_on_the_same_bounce():
    frames, truth = _scene()
    est = pitch_tracker.stitch(pitch_tracker.track(frames, FPS), FPS)
    s = evaluate.id_scores(truth, TrackSet(40, 20, FPS, est))
    assert s["idf1"] < 0.75


def test_feat_distance_weights_shorts_over_bib():
    same_bib_diff_shorts = pitch_tracker.feat_distance(np.r_[np.eye(15)[0], np.eye(15)[12]], np.r_[np.eye(15)[0], np.eye(15)[14]])
    diff_bib_same_shorts = pitch_tracker.feat_distance(np.r_[np.eye(15)[0], np.eye(15)[12]], np.r_[np.eye(15)[5], np.eye(15)[12]])
    assert same_bib_diff_shorts > diff_bib_same_shorts


def test_hungarian_matches_bruteforce():
    rng = np.random.default_rng(0)
    import itertools
    for _ in range(5):
        c = rng.uniform(0, 1, (4, 4))
        best = min(sum(c[i, p[i]] for i in range(4)) for p in itertools.permutations(range(4)))
        got = sum(c[i, j] for i, j in evaluate._hungarian(c))
        assert abs(best - got) < 1e-9
