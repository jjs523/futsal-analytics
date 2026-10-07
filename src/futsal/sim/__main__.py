"""python -m futsal.sim <command> ...

  figures   --court 40x20 --twist 5 --out docs/img      spec drawing, coverage map, synthetic demo
  fov       --court 40x20 --twist 5 --out docs/img      phone FOV chart, single-phone reach, 1x vs 0.5x
  coverage  --court 40x20 --twist 5 --hfov 67.3         coverage / accuracy numbers for one layout
  twist     --court 42x25                               largest blind-free twist + where the other phone appears
  search    --court 40x20 --hfov 67.3                   brute-force best two-phone placement
  accuracy  --court 40x20 --twist 5 --hfov 71.5         synthetic end-to-end position error
  tracks    --court 40x20 --seconds 300 --out x.json    synthetic tracks.json for the web viewer
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..court import Court
from . import coverage, layout, observe, scenario


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["figures", "fov", "coverage", "twist", "search", "accuracy", "tracks"])
    ap.add_argument("--court", default="40x20")
    ap.add_argument("--twist", type=float, default=5.0)
    ap.add_argument("--hfov", type=float, default=layout.HFOV_NARROW)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args(argv)
    court = Court.parse(a.court)

    if a.command == "figures":
        from .figures import all_figures
        all_figures(court, a.twist, a.out)
        print(f"wrote {a.out}/court_spec.png, coverage.png, demo.png")
    elif a.command == "fov":
        from .fov_figures import all_fov_figures
        all_fov_figures(court, a.out, a.twist)
        print(f"wrote {a.out}/fov_phones.png, fov_reach.png, fov_lens_compare.png")
    elif a.command == "coverage":
        s = coverage.analyse(court, layout.diagonal(court, a.twist), a.hfov).summary()
        print(json.dumps({k: round(v, 3) for k, v in s.items()}, indent=1))
    elif a.command == "twist":
        for hfov in (layout.HFOV_NARROW, layout.HFOV_WIDE):
            t = layout.safe_twist(court, hfov)
            x = layout.other_phone_screen_x(court, t, hfov)
            print(f"hfov {hfov:g}: max blind-free twist {t:g} deg -> other phone at {x * 100:.0f}% of the frame width")
    elif a.command == "search":
        for s1, s2, union, both in layout.search(court, a.hfov):
            print(f"{s1.xy} yaw {s1.yaw_deg:6.1f} | {s2.xy} yaw {s2.yaw_deg:6.1f} | union {union:5.1f}%  both {both:5.1f}%")
    elif a.command == "accuracy":
        truth = scenario.synthetic_match(court, a.seconds, seed=a.seed)
        est, cals = observe.run(truth, layout.diagonal(court, a.twist), a.hfov, seed=a.seed)
        e = observe.position_errors(truth, est)
        print(f"per-frame position error: median {np.median(e):.2f} m, mean {e.mean():.2f} m, p95 {np.percentile(e, 95):.2f} m")
        print("calibration rms px:", {k: round(c.rms_px, 2) for k, c in cals.items()})
    elif a.command == "tracks":
        truth = scenario.synthetic_match(court, a.seconds, seed=a.seed)
        est, _ = observe.run(truth, layout.diagonal(court, a.twist), layout.HFOV_AVERAGE, seed=a.seed)
        est.compute_stats()
        est.dump(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
