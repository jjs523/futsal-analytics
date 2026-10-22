"""Match-level assembly: per-camera pixel detections -> one TrackSet.

Both box-level trackers start the same way (track_cams):

    build_cam_boxes per camera -> refine_clock (second camera onto the first one's clock, matches >= 10 min) ->
    assign_teams -> align_cameras -> ...

ids="v2" (the default whenever every camera has ReID): the closed-set linker, 5 + 1 slots per team ('final_d'):

    ... -> build_tracklets (conservative per camera + pair_views) -> assign_identities_windowed -> refine_positions

ids="v2gta": the open-set linker ('final_b2' in the research code):

    ... -> sources.relink_source (conservative per camera + local re-link, pair_views, fused re-link, absorb_short)
    -> link_gta.link_tracklets_windowed (agglomerative, strict debris) -> refine_positions (RTS)

Blind visual audits (identity purity of long tracks, swap rate at same-team crossings) on two 6-minute windows of
the 10/8 test match. Window 1 (9:00-15:00, where both were tuned): open set 0.97 / 3 swaps in 31 crossings, closed
set 0.97-0.98 / 7 in 36. Window 2 (25:00-31:00, held out: a break with bibs changing hands, players sitting and
walking off): open set 0.90 / 35 %, closed set 0.83 / 55 %, because the closed set forces whoever is there into the
five slots. The open set's price is more ids than players (goalkeepers, substitutes, off-court people get their own),
and over a whole match it does not hold people together: on the 53-minute test (YOLO26x boxes) it gave 86
identities, none seen in more than 76 % of the match, where the closed set gave 32 with 5 players above 80 % (YOLO11x
boxes: 25 and 6). Per-player match statistics are the product, so the closed set is the default; v2gta is for short
clips where a wrong id costs more than a split one.

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
IDS = ("v2", "v2gta", "appearance", "motion")
LINKERS = {"v2": "closed", "v2gta": "gta"}      # box-level trackers: ids -> track_cams(linker=...)


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
               info: dict | None = None, linker: str = "closed", gta_params=None,
               source_params: dict | None = None, clock_refine: bool = True) -> tuple[list, list[str]]:
    """The v2 steps after the per-camera boxes (module docstring), on CamBoxes already on the common grid, e.g.
    boxes.build_cam_boxes or, for the research cache, boxes.cams_from_cache. Writes cam.team and the aligned xy in
    place. Returns (identities as pipeline.boxes.Track with RTS-refined xy, solver team per identity: 'A' / 'B' for
    the closed set, 'A' / 'B' / 'U' for the open set).

    linker="closed" (ids "v2"): tracklets.build_tracklets -> link_closed.assign_identities_windowed(closed_params).
    linker="gta" (ids "v2gta"): sources.relink_source(**source_params, e.g. proposals=None) ->
    link_gta.link_tracklets_windowed(gta_params). Both linkers skip their own smoothing by default: the RTS
    refinement replaces those positions (closed: also no re-fusion; gta: GtaParams(smooth='none')).
    clock_refine: first move the second camera onto the first one's clock with the players (align.refine_clock: a
    slow drift between the phones' video clocks, which the audio sync cannot see; clips shorter than ~10 minutes are
    left as they are).
    `info` receives "clock", "teams", "alignment", "tracklets", "solver" and "identities" (see build_tracks_v2),
    plus "source" (sources.SourceStats) for the open set."""
    from .align import align_cameras, refine_clock
    from .boxes import assign_teams
    from .link_closed import WindowedStats
    from .refine import RefineParams, refine_positions

    if linker not in ("closed", "gta"):
        raise ValueError(f"linker must be 'closed' or 'gta', got {linker!r}")
    info = info if info is not None else {}
    info["clock"] = refine_clock(cams, rate) if clock_refine else None
    info["teams"] = assign_teams(cams)
    info["alignment"] = align_cameras(cams, court)
    st = WindowedStats()
    if linker == "closed":
        from .link_closed import ClosedParams, assign_identities_windowed
        from .tracklets import build_tracklets
        tracklets = build_tracklets(cams, rate)
        info["tracklets"] = len(tracklets)
        params = closed_params or replace(ClosedParams(), refuse=False, smooth=False)
        ids = assign_identities_windowed(tracklets, cams, rate, params, window_s=window_s,
                                         overlap_s=overlap_s, border_exempt=border_exempt, stats=st, court=court)
    else:
        from .link_gta import GtaParams, link_tracklets_windowed
        from .sources import SourceStats, relink_source
        sst = SourceStats()
        tracklets = relink_source(cams, rate, stats=sst, **(source_params or {}))
        info["source"] = sst
        info["tracklets"] = len(tracklets)
        # identities shorter than min_len_s are dropped: never ask for more than a quarter of the clip
        ks = [k for c in cams.values() if c.by_k for k in (min(c.by_k), max(c.by_k))]
        span_s = (1 + max(ks) - min(ks)) / rate if ks else 0.0
        params = gta_params or replace(GtaParams(), smooth="none", min_len_s=min(GtaParams().min_len_s, span_s / 4))
        ids = link_tracklets_windowed(tracklets, cams, rate, params, window_s=window_s, overlap_s=overlap_s,
                                      border_exempt=border_exempt, stats=st)
    info["solver"] = st
    slot_teams = list(st.output_team)
    ids = refine_positions(ids, cams, rate, refine_params or RefineParams())
    info["identities"] = len(ids)
    return ids, (slot_teams if len(slot_teams) == len(ids) else [])


def build_tracks_v2(court: Court, detections: dict[str, list[Detection]],
                    calibrations: dict[str, Calibration | CalibrationTimeline], offsets: dict | None = None,
                    rate: float = 10.0, frame_wh: dict[str, tuple[int, int]] | None = None, closed_params=None,
                    refine_params=None, window_s: float = 360.0, overlap_s: float = 60.0,
                    border_exempt: dict[str, tuple[str, ...]] | None = None, info: dict | None = None,
                    linker: str = "closed", gta_params=None, source_params: dict | None = None) -> TrackSet:
    """The box-level tracker (module docstring). The grid is k = round((t + offset + drift * t) * rate) on the
    reference clock, as in to_observations, so `rate` is both the detection rate and the output fps. `frame_wh`
    {camera: (w, h)} in calibrated pixels (guessed from the boxes when missing; the clean-crop border rule needs it).
    A clip up to `window_s` is one linker run, a longer one overlapping windows chained into identities.
    `linker` "closed" (ids "v2") or "gta" (ids "v2gta"); `closed_params` / `gta_params` / `source_params` /
    `refine_params` / `border_exempt` default to the audited settings (see track_cams).
    `info` (optional dict) receives "ids" ("v2" / "v2gta"), "clock" (align.refine_clock result: the correction of
    the second camera's clock, "applied" False for short clips), "teams" (assign_teams summary), "alignment"
    (align_cameras result, None when skipped: one camera or too few matched pairs), "tracklets" (count), "solver"
    (link_closed.WindowedStats), "identities" (count) and, for "v2gta", "source" (sources.SourceStats)."""
    from .boxes import build_cam_boxes
    from .tracklets import default_frames

    if linker not in ("closed", "gta"):
        raise ValueError(f"linker must be 'closed' or 'gta', got {linker!r}")
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
    info["ids"] = "v2gta" if linker == "gta" else "v2"
    n = len(default_frames(cams))
    ids, slot_teams = track_cams(cams, court, rate, closed_params, refine_params, window_s, overlap_s, border_exempt,
                                 info, linker, gta_params, source_params)
    ts = tracks_to_trackset(ids, cams, court, rate, n, slot_teams or None)
    ts.compute_stats()
    return ts


# ---------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------

def build_tracks(court: Court, detections: dict[str, list[Detection]], calibrations: dict[str, Calibration | CalibrationTimeline],
                 offsets: dict | None = None, fps_out: float = 10.0, gate: float = 1.0, ids: str = "v2",
                 info: dict | None = None, **v2) -> TrackSet:
    """ids="v2" (default): the box-level tracker with the closed-set linker (build_tracks_v2; the `v2` keywords go
    there, e.g. frame_wh, window_s), used when every camera's detections carry ReID; otherwise "appearance", silently
    (the server and older caches have no ReID; `info` says why) and the `v2` keywords are ignored. Keep fps_out at
    the detection rate (one grid frame per detector frame).
    ids="v2gta": the same with the open-set linker (link_gta; same ReID requirement and fallback).
    ids="appearance": short unambiguous tracklets re-linked over the whole match with appearance + motion
    (pitch_tracker.tracklets / associate). ids="motion": the older motion-only tracker + gap stitching.
    `info` (optional dict) gets "ids" (the tracker actually used), "fallback" (why v2 / v2gta was not used) and, for
    v2 / v2gta, the extras listed in build_tracks_v2. The TrackSet JSON format is the same for all of them."""
    if ids not in IDS:
        raise ValueError(f"ids must be one of {IDS}, got {ids!r}")
    info = info if info is not None else {}
    if ids in LINKERS:
        used = {c: d for c, d in detections.items() if c in calibrations}
        if has_reid(used):
            return build_tracks_v2(court, used, calibrations, offsets, fps_out, info=info, linker=LINKERS[ids], **v2)
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
