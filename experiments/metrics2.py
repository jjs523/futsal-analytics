"""Extended label-free evaluation (research_plan.md, "Label-free evaluation"), with a time split for honest tuning:
tune on part='first' (frames [0, n/2)), report on part='second' with the parameters frozen.

    python experiments/metrics2.py baseline baseline_motion          # experiments/results/<name>.json
    python experiments/metrics2.py baseline --parts all,first,second

All harness.metrics() numbers are included (computed on the clipped tracks, per-minute rates over the clipped
duration), plus count consistency per team, physics, cross-view consistency, a crossing swap score and within-track
ReID purity.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from harness import Data, Track
from blocks import box_covariances, clean_feats, clean_masks, dbscan_labels, mahalanobis2, medoid, normalise, team_vote

MID_PITCH_M = 3.0         # "mid-pitch": at least this far from every line (both phones should see the player)


def part_range(data: Data, part: str) -> tuple[int, int]:
    half = data.n // 2
    return {"all": (0, data.n), "first": (0, half), "second": (half, data.n)}[part]


def clip(tracks: list[Track], lo: int, hi: int) -> list[Track]:
    """Tracks restricted to frames [lo, hi); empty tracks dropped."""
    out = [{k: p for k, p in t.items() if lo <= k < hi} for t in tracks]
    return [t for t in out if t]


def _window(data: Data, lo: int, hi: int):
    """A Data look-alike for harness.metrics() covering frames [lo, hi) as 0 .. hi-lo-1: n, duration, and the camera
    frame indices are shifted, so its per-minute rates, window-edge rules and box coverage refer to the window."""
    cams = {cam: SimpleNamespace(conf=c.conf, in_court=c.in_court, k=c.k - lo, h=c.h, reid=c.reid, team=c.team)
            for cam, c in data.cams.items()}
    return SimpleNamespace(n=hi - lo, rate=data.rate, duration=(hi - lo) / data.rate, court=data.court, cams=cams)


def _positions(t: Track, lo: int, hi: int) -> np.ndarray:
    a = np.full((hi - lo, 2), np.nan)
    for k, p in t.items():
        a[k - lo] = p.xy
    return a


def _mid_pitch(xy: np.ndarray, court) -> bool:
    x, y = xy
    return min(x, y, court.length - x, court.width - y) >= MID_PITCH_M


# ---------------------------------------------------------------------------------------------------------------

def events_per_min(tracks: list[Track], data: Data, lo: int, hi: int, edge_m: float = 2.5, settle_s: float = 2.0) -> float:
    """harness.metrics' births/deaths in the middle of the pitch, but judged on the UNclipped tracks: a track that
    goes quiet at minute 2:30 and resumes at 3:10 has no event at the clip boundary (as in the full-window metric),
    whereas clipping first would invent a death at 2:30. Window-edge settling is relative to [lo, hi)."""
    court, rate = data.court, data.rate
    ev = 0
    for t in tracks:
        if not t:
            continue
        ks = sorted(t)
        for kind, k in (("new", ks[0]), ("lost", ks[-1])):
            if not lo <= k < hi:
                continue
            if (kind == "new" and k < lo + settle_s * rate) or (kind == "lost" and k > hi - 1 - settle_s * rate):
                continue
            x, y = t[k].xy
            if min(x, y, court.length - x, court.width - y) >= edge_m:
                ev += 1
    return round(ev / ((hi - lo) / rate / 60.0), 2)


def count_consistency(tracks: list[Track], teams: list[str], lo: int, hi: int) -> dict:
    """Per frame, how many tracks of each team are on the pitch. A 5v5 match should show exactly 5 Y and 5 N:
    more than 5 means a duplicate / fragment, fewer a coverage hole."""
    n = hi - lo
    cnt = {t: np.zeros(n, int) for t in ("Y", "N", "U")}
    for tr, team in zip(tracks, teams):
        ks = np.array(list(tr), int) - lo
        cnt[team][ks] += 1
    y, nn = cnt["Y"], cnt["N"]
    return {
        "count_5v5": round(float(((y == 5) & (nn == 5)).mean()), 3),
        "count_over5": round(float(((y > 5) | (nn > 5)).mean()), 3),
        "count_under5": round(float(((y < 5) | (nn < 5)).mean()), 3),
        "mean_Y": round(float(y.mean()), 2),
        "mean_N": round(float(nn.mean()), 2),
        "mean_U": round(float(cnt["U"].mean()), 2),
    }


def physics(tracks: list[Track], rate: float, minutes: float, vmax: float = 9.0, win: int = 5,
            jump: float = 6.0) -> dict:
    """Raw (unsmoothed) physics violations: implied speed above vmax between samples <= 3 frames apart, and speed
    jumps: |v_after - v_before| > jump m/s with v measured over `win` frames on each side of a sample (a link
    between two people shows up as an abrupt velocity change). Consecutive flagged samples count as one event."""
    fast = jumps = 0
    for t in tracks:
        ks = np.array(sorted(t), int)
        P = np.stack([t[k].xy for k in ks])
        d = np.diff(ks)
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / (d / rate)
        fast += int(((d <= 3) & (v > vmax)).sum())
        have = set(ks.tolist())
        prev_flag = False
        for k in ks:
            k = int(k)
            if k - win not in have or k + win not in have:
                prev_flag = False
                continue
            vb = (t[k].xy - t[k - win].xy) * rate / win
            va = (t[k + win].xy - t[k].xy) * rate / win
            flag = float(np.linalg.norm(va - vb)) > jump
            if flag and not prev_flag:
                jumps += 1
            prev_flag = flag
    return {"speed_over9_per_min": round(fast / minutes, 2), "speed_jumps_per_min": round(jumps / minutes, 2)}


def cross_view(tracks: list[Track], data: Data, minutes: float, gate_d2: float = 9.21) -> dict:
    """Cross-view consistency of fused points (largest box per camera): share within 1 m, share inside the B2
    Mahalanobis gate, share of mid-pitch points supported by both cameras, and how often a track's supporting camera set changes mid-pitch (flicker between fused and
    single-view; a stable person seen by both phones should not flip)."""
    cams = list(data.cams)
    R = {cam: box_covariances(data, cam) for cam in cams}
    both = near = gated = 0
    flips = mid = mid_both = 0
    for t in tracks:
        prev_k, prev_set = None, None
        for k in sorted(t):
            p = t[k]
            if not p.boxes:
                prev_k = None
                continue
            best = {}
            for cam, i in p.boxes:
                if cam not in best or data.cams[cam].h[i] > data.cams[cam].h[best[cam]]:
                    best[cam] = i
            if len(best) == 2:
                (c1, i1), (c2, i2) = best.items()
                d = data.cams[c1].xy[i1] - data.cams[c2].xy[i2]
                both += 1
                near += float(np.linalg.norm(d)) <= 1.0
                gated += mahalanobis2(d, R[c1][i1] + R[c2][i2]) < gate_d2
            s = frozenset(best)
            if _mid_pitch(p.xy, data.court):
                mid += 1
                mid_both += len(best) == 2
            if prev_k is not None and k - prev_k <= 3 and s != prev_set and _mid_pitch(p.xy, data.court):
                flips += 1
            prev_k, prev_set = k, s
    return {"xview_points": both, "xview_within_1m": round(near / max(both, 1), 3),
            "xview_in_gate": round(gated / max(both, 1), 3), "mid_pitch_both_cams": round(mid_both / max(mid, 1), 3),
            "support_flips_per_min": round(flips / minutes, 2)}


def crossing_swaps(tracks: list[Track], teams: list[str], feats: list[dict[int, np.ndarray]], lo: int, hi: int,
                   rate: float, minutes: float, dist: float = 1.2, merge_s: float = 1.0, max_len_s: float = 5.0,
                   margin: float = 0.05) -> dict:
    """Plan metric 6: for every same-team pair coming within `dist` m, compare clean ReID 1-3 s before and 1-3 s after
    the encounter. Flag a swap when cos(A_pre, B_post) + cos(B_pre, A_post) > cos(A_pre, A_post) + cos(B_pre, B_post)
    + margin. Encounters longer than max_len_s are duplicates rather than crossings and are skipped."""
    pos = [_positions(t, lo, hi) for t in tracks]

    def window_mean(f: dict[int, np.ndarray], a: int, b: int):
        rows = [f[k] for k in range(a, b) if k in f]
        return normalise(np.mean(rows, 0)) if rows else None

    enc = scored = flagged = 0
    r1, r3 = int(1 * rate), int(3 * rate)
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if teams[i] != teams[j] or teams[i] == "U":
                continue
            close = np.linalg.norm(pos[i] - pos[j], axis=1) < dist        # NaN compares False
            idx = np.flatnonzero(close)
            if not len(idx):
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > merge_s * rate) + 1)
            for run in runs:
                if (run[-1] - run[0] + 1) > max_len_s * rate:
                    continue
                enc += 1
                k0, k1 = int(run[0]) + lo, int(run[-1]) + lo
                ap, bp = window_mean(feats[i], k0 - r3, k0 - r1 + 1), window_mean(feats[j], k0 - r3, k0 - r1 + 1)
                aq, bq = window_mean(feats[i], k1 + r1, k1 + r3 + 1), window_mean(feats[j], k1 + r1, k1 + r3 + 1)
                if any(v is None for v in (ap, bp, aq, bq)):
                    continue
                scored += 1
                flagged += float(ap @ bq + bp @ aq) > float(ap @ aq + bp @ bq) + margin
    return {"crossings": enc, "crossings_per_min": round(enc / minutes, 2), "crossings_scored": scored,
            "swap_flagged_share": round(flagged / max(scored, 1), 3)}


def reid_purity(tracks: list[Track], feats: list[tuple[np.ndarray, np.ndarray]], rate: float, min_crops: int = 5,
                long_s: float = 20.0, eps: float = 0.55, min_samples: int = 5) -> dict:
    """Within-track ReID purity: length-weighted mean cosine distance of clean crops to the track medoid, and the
    share of tracks >= long_s whose clean crops form more than one DBSCAN cluster (a likely identity mix)."""
    disp, w = [], []
    long_n = multi = 0
    for t, (_, f) in zip(tracks, feats):
        if len(f) >= min_crops:
            disp.append(float(np.mean(1.0 - f @ medoid(f))))
            w.append(len(t))
        if len(t) >= long_s * rate and len(f) >= min_samples:
            long_n += 1
            lab = dbscan_labels(f, eps, min_samples)
            multi += len(np.unique(lab[lab >= 0])) > 1
    return {"reid_dispersion": round(float(np.average(disp, weights=w)), 4) if disp else float("nan"),
            "long_tracks": long_n, "multi_cluster_share": round(multi / max(long_n, 1), 3)}


def evaluate(tracks: list[Track], data: Data, part: str = "all") -> dict:
    """harness.metrics() plus the extended metrics, on the tracks clipped to `part` ('all' | 'first' | 'second')."""
    lo, hi = part_range(data, part)
    T = clip(tracks, lo, hi)
    minutes = (hi - lo) / data.rate / 60.0
    shifted = [{k - lo: p for k, p in t.items()} for t in T]
    out = dict(harness.metrics(shifted, _window(data, lo, hi)))
    out["events_per_min"] = events_per_min(tracks, data, lo, hi)       # identical to harness for part='all'
    teams = [team_vote(t, data)[0] for t in T]
    clean = clean_masks(data)
    cf = [clean_feats(t, data, clean) for t in T]
    out |= count_consistency(T, teams, lo, hi)
    out |= physics(T, data.rate, minutes)
    out |= cross_view(T, data, minutes)
    out |= crossing_swaps(T, teams, [dict(zip(ks.tolist(), f)) for ks, f in cf], lo, hi, data.rate, minutes)
    out |= reid_purity(T, cf, data.rate)
    return out


def table(results: dict[str, dict]) -> str:
    cols = list(results)
    rows = list(next(iter(results.values())))
    wk = max(len(r) for r in rows)
    wc = max(10, *(len(c) for c in cols))
    lines = [" " * wk + "  " + "  ".join(c.rjust(wc) for c in cols)]
    for r in rows:
        lines.append(r.ljust(wk) + "  " + "  ".join(str(results[c].get(r, "")).rjust(wc) for c in cols))
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    import argparse
    import time
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+", help="result names under experiments/results/")
    ap.add_argument("--parts", default="first,second")
    args = ap.parse_args(argv)
    t0 = time.time()
    data = Data()
    print(f"data loaded in {time.time() - t0:.0f} s", file=sys.stderr)
    results = {}
    for name in args.names:
        tracks = harness.load_tracks(os.path.join(harness.ROOT, "experiments", "results", f"{name}.json"))
        for part in args.parts.split(","):
            results[f"{name}/{part}"] = evaluate(tracks, data, part)
    print(table(results))


if __name__ == "__main__":
    main(sys.argv[1:])
