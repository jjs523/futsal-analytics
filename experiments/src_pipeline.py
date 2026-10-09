"""The production tracker v2 (src/futsal/pipeline/run.py, build_tracks(ids="v2")) on the research cache, scored with
the same harness as the research candidates, so src and research stay comparable end to end.

    python experiments/harness.py src_v2 src_pipeline
    python experiments/metrics2.py final_d src_v2 src_v2_link src_v2_yn --parts first,second

  src_v2       run.track_cams exactly as build_tracks(ids="v2") calls it: assign_teams (k-means A / B) ->
               align_cameras -> build_tracklets -> assign_identities_windowed (360 s windows, top edge exempt on
               both cameras) -> refine_positions (RTS). Its positions are the analytics output: the RTS smoother
               removes what the speed metrics measure, so judge link quality on src_v2_link.
  src_v2_link  the same identities with the linker's own positions (ClosedParams() re-fusion + Savitzky-Golay,
               no RTS), comparable with final_d on the physics metrics.
  src_v2_yn    src_v2_link with the research inputs: yellow-bib teams, the clean-crop exemption on cam2 only, and
               alignment pairs and tracklets from the research window's 3600 frames (the cache also has boxes at
               k = 3600 / 3601). Should be final_d box for box (the src port is parity-tested against it).

Only the tracker comes from src; the cache's boxes come in through boxes.cams_from_cache, the same arrays (and box
indices) as harness.Data.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from harness import CACHE, ROOT, Data, Track, TrackPoint, register

sys.path.insert(0, os.path.join(ROOT, "src"))
from futsal.court import Court
from futsal.pipeline import run
from futsal.pipeline.align import align_cameras
from futsal.pipeline.boxes import assign_teams, cams_from_cache
from futsal.pipeline.link_closed import ClosedParams, WindowedStats, assign_identities_windowed
from futsal.pipeline.tracklets import build_tracklets


def to_harness(tracks: list) -> list[Track]:
    """pipeline.boxes.Track -> harness Track (same (camera, box index) pairs: both index the cache arrays)."""
    return [{int(k): TrackPoint(np.asarray(p.xy, float), [(c, int(i)) for c, i in p.boxes]) for k, p in sorted(t.items())}
            for t in tracks if t]


def _cams(court: Court) -> tuple[dict, float]:
    cams, meta = cams_from_cache(CACHE, ROOT, court)
    return cams, float(meta["rate"])


def _report(name: str, info: dict, t0: float) -> None:
    st = info["solver"]
    al = info["alignment"] or {}
    print(f"{name}: alignment {al.get('pairs')} pairs {al.get('median_before_m')} -> {al.get('median_after_m')} m, "
          f"{info['tracklets']} tracklets, windows {st.windows}, identities {info['identities']}, "
          f"teams {st.output_team}, {time.time() - t0:.1f} s", file=sys.stderr)


@register("src_v2")
def src_v2(data: Data) -> list[Track]:
    t0 = time.time()
    cams, rate = _cams(data.court)
    info: dict = {}
    ids, _ = run.track_cams(cams, data.court, rate, info=info)
    _report("src_v2", info, t0)
    return to_harness(ids)


def _linker_only(data: Data, yellow: bool, border_exempt, research_window: bool = False) -> list[Track]:
    """track_cams' steps up to the linker, with its own re-fused, smoothed positions."""
    t0 = time.time()
    cams, rate = _cams(data.court)
    info = {"teams": assign_teams(cams)}
    if yellow:
        for name, cam in cams.items():
            cam.team = np.array([{"Y": "A", "N": "B"}.get(x, "") for x in data.cams[name].team], dtype="<U1")
    k_range = (0, data.n) if research_window else None
    info["alignment"] = align_cameras(cams, data.court, k_range=k_range)
    T = build_tracklets(cams, rate, frames=range(data.n) if research_window else None)
    info["tracklets"] = len(T)
    st = WindowedStats()
    ids = assign_identities_windowed(T, cams, rate, ClosedParams(), border_exempt=border_exempt, stats=st,
                                     court=data.court)
    info["solver"], info["identities"] = st, len(ids)
    _report("src_v2_yn" if yellow else "src_v2_link", info, t0)
    return to_harness(ids)


@register("src_v2_link")
def src_v2_link(data: Data) -> list[Track]:
    return _linker_only(data, yellow=False, border_exempt=None)


@register("src_v2_yn")
def src_v2_yn(data: Data) -> list[Track]:
    return _linker_only(data, yellow=True, border_exempt={"cam2": ("top",)}, research_window=True)
