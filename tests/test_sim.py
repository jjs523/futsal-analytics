import numpy as np

from futsal import Court
from futsal.sim import coverage, layout, observe, scenario

COURT = Court(40, 20)


def test_recommended_layout_has_no_blind_area_for_any_1x_phone():
    lay = layout.diagonal(COURT, 5)
    for hfov in (layout.HFOV_NARROW, layout.HFOV_WIDE):
        s = coverage.analyse(COURT, lay, hfov).summary()
        assert s["blind_m2"] == 0
        assert s["both_pct"] > 78


def test_twist_beats_pure_facing():
    facing = coverage.analyse(COURT, layout.diagonal(COURT, 0), layout.HFOV_NARROW).summary()
    twisted = coverage.analyse(COURT, layout.diagonal(COURT, 5), layout.HFOV_NARROW).summary()
    assert twisted["both_pct"] > facing["both_pct"] + 4


def test_wide_court_needs_smaller_twist():
    assert layout.safe_twist(Court(42, 25)) < layout.safe_twist(COURT)


def test_synthetic_match_is_humanly_plausible():
    ts = scenario.synthetic_match(COURT, seconds=120, seed=3)
    ts.compute_stats()
    outfield = [p.stats for p in ts.players if p.stats["distance_m"] > 120]
    assert len(outfield) == 8
    per_min = np.mean([s["distance_m"] for s in outfield]) / 2
    assert 70 < per_min < 140
    assert max(s["max_speed_ms"] for s in outfield) < 8


def test_end_to_end_geometry_accuracy():
    truth = scenario.synthetic_match(COURT, seconds=30, seed=2)
    est, cals = observe.run(truth, layout.diagonal(COURT, 5), layout.HFOV_AVERAGE, seed=2)
    e = observe.position_errors(truth, est)
    assert np.median(e) < 0.5
    assert all(c.rms_px < 4 for c in cals.values())
