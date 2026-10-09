"""Match-level assembly: per-camera pixel detections -> one TrackSet.

ids="v2" (the default whenever every camera has ReID): the box-level tracker chosen in the blind visual audit
('final_d': identity purity 0.98 vs 0.89, same-team crossing swaps 30 % vs 55 %):

    build_cam_boxes per camera -> assign_teams -> align_cameras -> build_tracklets (conservative per camera +
    pair_views) -> assign_identities_windowed (closed set, 5 + 1 slots per team) -> refine_positions (RTS)

ids="appearance" / "motion": the older foot-point fusion trackers (pitch_tracker), used whenever ReID is missing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..court import Court
from ..fusion import Observation, fuse_frame
from ..homography import Calibration, CalibrationTimeline, calibration_at
from ..tracks import PlayerTrack, TrackSet
from . import pitch_tracker
from .detect import Detection, dequantize

DETECTOR_JITTER_PX = 3.0      # foot-point noise of the detector, used to weight cameras
IDS = ("v2", "appearance", "motion")


def clock(sync) -> tuple[float, float]:
    """A camera's sync entry: a plain offset in seconds, or {"offset": s, "drift": ratio} from futsal.sync.align."""
    if isinstance(sync, dict):
        return float(sync.get("offset", 0.0)), float(sync.get("drift", 0.0))
    return float(sync or 0.0), 0.0


def to_observations(dets: list[Detection], cal: Calibration | CalibrationTimeline, camera: str, offset_s, fps_out: float,
                    court: Court, margin: float = 1.0) -> dict[int, list[Observation]]:
    """Pixels -> pitch, map the camera's clock onto the reference clock, bucket to the output frame grid.
    Points far outside the pitch (spectators, subs on the bench) are dropped. `cal` may be a timeline
    [(from_s, Calibration), ...] (camera's own seconds) when the tripod was knocked during the match."""
    offset, drift = clock(offset_s)
    if not dets:
        return {}
    uv = np.array([[d.u, d.v] for d in dets])
    xy, sig = np.zeros_like(uv), np.zeros(len(dets))
    which = [id(calibration_at(cal, d.t)) for d in dets]
    for c in {id(c): c for c in (calibration_at(cal, d.t) for d in dets)}.values():
        m = np.array([w == id(c) for w in which])
        xy[m] = c.to_pitch(uv[m])
        sig[m] = DETECTOR_JITTER_PX * c.metres_per_pixel(uv[m])
    inside = court.contains(xy, margin)
    out: dict[int, list[Observation]] = {}
    for d, p, s, ok in zip(dets, xy, sig, inside):
        if not ok:
            continue
        k = int(round((d.t + offset + drift * d.t) * fps_out))
        if k < 0:
            continue
        out.setdefault(k, []).append(Observation(p, float(s), camera, team=d.team, feat=dequantize(d.feat)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Box-level tracker (v2)
# ---------------------------------------------------------------------------------------------------------------

def has_reid(detections: dict[str, list[Detection]]) -> bool:
    """True when every camera has detections and they carry ReID embeddings (detect_video(embedder=...))."""
    return bool(detections) and all(ds and any(d.reid is not None for d in ds) for ds in detections.values())


def guess_frame_wh(dets: list[Detection]) -> tuple[int, int]:
    """Frame size when the caller does not know it: 1920x1080 (the recording standard) unless boxes fall outside
    it, then 3840x2160, else the boxes' extent. Only the clean-crop border test uses it."""
    boxes = np.array([d.box for d in dets if d.box is not None], float).reshape(-1, 4)
    x2, y2 = (float(boxes[:, 2].max()), float(boxes[:, 3].max())) if len(boxes) else (0.0, 0.0)
    for w, h in ((1920, 1080), (3840, 2160)):
        if x2 <= w + 1 and y2 <= h + 1:
            return w, h
    return int(np.ceil(x2)), int(np.ceil(y2))


def tracks_to_trackset(tracks: list, cams: dict, court: Court, rate: float, n_frames: int,
                       slot_teams: list[str] | None = None) -> TrackSet:
    """Identities (pipeline.boxes.Track) -> TrackSet over grid frames 0 .. n_frames - 1: one PlayerTrack per
    identity, NaN where it has no point. Team 'A' / 'B' by identity.team_vote; when the vote is undecided, the team
    of the slot the solver gave it (`slot_teams`, parallel to `tracks`), else '?' as in the older trackers. IDs are
    numbered team A first, then by first appearance, so they read the same from run to run."""
    from .identity import team_vote
    rows = []
    for q, tr in enumerate(tracks):
        ks = np.array([k for k in tr if 0 <= k < n_frames], int)
        if not len(ks):
            continue
        xy = np.full((n_frames, 2), np.nan)
        xy[ks] = np.stack([tr[int(k)].xy for k in ks])
        team = team_vote(tr, cams)[0]
        if team not in ("A", "B"):
            team = slot_teams[q] if slot_teams and slot_teams[q] in ("A", "B") else "?"
        rows.append((team, int(ks.min()), xy))
    rows.sort(key=lambda r: ({"A": 0, "B": 1}.get(r[0], 2), r[1]))
    players = [PlayerTrack(i + 1, team, xy) for i, (team, _, xy) in enumerate(rows)]
    return TrackSet(court.length, court.width, rate, players)


def track_cams(cams: dict, court: Court, rate: float = 10.0, closed_params=None, refine_params=None,
               window_s: float = 360.0, overlap_s: float = 60.0, border_exempt: dict[str, tuple[str, ...]] | None = None,
               info: dict | None = None) -> tuple[list, list[str]]:
    """The v2 steps after the per-camera boxes (module docstring), on CamBoxes already on the common grid, e.g.
    boxes.build_cam_boxes or, for the research cache, boxes.cams_from_cache. Writes cam.team and the aligned xy in
    place. Returns (identities as pipeline.boxes.Track with RTS-refined xy, solver team 'A' / 'B' per identity).
    `info` receives "teams", "alignment", "tracklets", "solver" and "identities" (see build_tracks_v2)."""
    from .align import align_cameras
    from .boxes import assign_teams
    from .link_closed import ClosedParams, WindowedStats, assign_identities_windowed
    from .refine import RefineParams, refine_positions
    from .tracklets import build_tracklets

    info = info if info is not None else {}
    info["teams"] = assign_teams(cams)
    info["alignment"] = align_cameras(cams, court)
    tracklets = build_tracklets(cams, rate)
    info["tracklets"] = len(tracklets)
    st = WindowedStats()
    # the RTS refinement replaces the linker's re-fused, smoothed positions, so the linker skips computing them
    params = closed_params or replace(ClosedParams(), refuse=False, smooth=False)
    ids = assign_identities_windowed(tracklets, cams, rate, params, window_s=window_s,
                                     overlap_s=overlap_s, border_exempt=border_exempt, stats=st, court=court)
    info["solver"] = st
    slot_teams = list(st.output_team)
    ids = refine_positions(ids, cams, rate, refine_params or RefineParams())
    info["identities"] = len(ids)
    return ids, (slot_teams if len(slot_teams) == len(ids) else [])


def build_tracks_v2(court: Court, detections: dict[str, list[Detection]],
                    calibrations: dict[str, Calibration | CalibrationTimeline], offsets: dict | None = None,
                    rate: float = 10.0, frame_wh: dict[str, tuple[int, int]] | None = None, closed_params=None,
                    refine_params=None, window_s: float = 360.0, overlap_s: float = 60.0,
                    border_exempt: dict[str, tuple[str, ...]] | None = None, info: dict | None = None) -> TrackSet:
    """The box-level tracker (module docstring). The grid is k = round((t + offset + drift * t) * rate) on the
    reference clock, as in to_observations, so `rate` is both the detection rate and the output fps. `frame_wh`
    {camera: (w, h)} in calibrated pixels (guessed from the boxes when missing; the clean-crop border rule needs it).
    A clip up to `window_s` is one closed-set solve, a longer one overlapping windows chained into identities.
    `closed_params` / `refine_params` / `border_exempt` default to the audited settings of link_closed / refine
    (ClosedParams without its own re-fusion and smoothing, which refine_positions replaces).
    `info` (optional dict) receives "teams" (assign_teams summary), "alignment" (align_cameras result, None when
    skipped: one camera or too few matched pairs), "tracklets" (count), "solver" (link_closed.WindowedStats) and
    "identities" (count)."""
    from .boxes import build_cam_boxes
    from .tracklets import default_frames

    offsets = offsets or {}
    frame_wh = frame_wh or {}
    info = info if info is not None else {}
    cams = {}
    for cam, dets in detections.items():
        if cam not in calibrations:
            continue
        off, drift = clock(offsets.get(cam, 0.0))
        cams[cam] = build_cam_boxes(cam, dets, calibrations[cam], off, drift, 0.0, rate, court,
                                    frame_wh.get(cam) or guess_frame_wh(dets))
    info["ids"] = "v2"
    n = len(default_frames(cams))
    ids, slot_teams = track_cams(cams, court, rate, closed_params, refine_params, window_s, overlap_s, border_exempt,
                                 info)
    ts = tracks_to_trackset(ids, cams, court, rate, n, slot_teams or None)
    ts.compute_stats()
    return ts


# ---------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------

def build_tracks(court: Court, detections: dict[str, list[Detection]], calibrations: dict[str, Calibration | CalibrationTimeline],
                 offsets: dict | None = None, fps_out: float = 10.0, gate: float = 1.0, ids: str = "v2",
                 info: dict | None = None, **v2) -> TrackSet:
    """ids="v2": the box-level tracker (build_tracks_v2; the `v2` keywords go there, e.g. frame_wh, window_s), used
    when every camera's detections carry ReID; otherwise "appearance", silently (the server and older caches have no
    ReID; `info` says why) and the `v2` keywords are ignored. Keep fps_out at the detection rate for v2 (one grid
    frame per detector frame).
    ids="appearance": short unambiguous tracklets re-linked over the whole match with appearance + motion
    (pitch_tracker.tracklets / associate). ids="motion": the older motion-only tracker + gap stitching.
    `info` (optional dict) gets "ids" (the tracker actually used), "fallback" (why v2 was not used) and, for v2, the
    extras listed in build_tracks_v2. The TrackSet JSON format is the same for all three."""
    if ids not in IDS:
        raise ValueError(f"ids must be one of {IDS}, got {ids!r}")
    info = info if info is not None else {}
    if ids == "v2":
        used = {c: d for c, d in detections.items() if c in calibrations}
        if has_reid(used):
            return build_tracks_v2(court, used, calibrations, offsets, fps_out, info=info, **v2)
        info["fallback"] = "no ReID embeddings"
        ids = "appearance"
    info["ids"] = ids
    offsets = offsets or {}
    per_cam = {cam: to_observations(d, calibrations[cam], cam, offsets.get(cam, 0.0), fps_out, court)
               for cam, d in detections.items() if cam in calibrations}
    n = 1 + max((max(o) for o in per_cam.values() if o), default=-1)
    frames = []
    for k in range(n):
        frames.append(fuse_frame({cam: obs.get(k, []) for cam, obs in per_cam.items()}, gate=gate))
    if ids == "motion":
        players = pitch_tracker.stitch(pitch_tracker.track(frames, fps_out), fps_out)
    else:
        players = pitch_tracker.associate(pitch_tracker.tracklets(frames, fps_out, ambiguity=0.4), len(frames), fps_out)
        players = pitch_tracker.absorb_duplicates(players)
    ts = TrackSet(court.length, court.width, fps_out, players)
    ts.compute_stats()
    return ts
