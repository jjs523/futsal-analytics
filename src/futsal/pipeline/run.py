"""Match-level assembly: per-camera pixel detections -> one TrackSet."""
from __future__ import annotations

import numpy as np

from ..court import Court
from ..fusion import Observation, fuse_frame
from ..homography import Calibration, CalibrationTimeline, calibration_at
from ..tracks import TrackSet
from . import pitch_tracker
from .detect import Detection, dequantize

DETECTOR_JITTER_PX = 3.0      # foot-point noise of the detector, used to weight cameras


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
    # one detection sample per output frame: a 60.04 fps video sampled every 6th frame occasionally puts two
    # samples into the same 0.1 s bucket, which would show every player twice in that frame
    t_ref = np.array([d.t + offset + drift * d.t for d in dets])
    k_of = np.round(t_ref * fps_out).astype(int)
    best: dict[int, float] = {}
    for k, tr in zip(k_of, t_ref):
        if k not in best or abs(tr - k / fps_out) < abs(best[k] - k / fps_out):
            best[k] = tr
    out: dict[int, list[Observation]] = {}
    for d, p, s, ok, k, tr in zip(dets, xy, sig, inside, k_of, t_ref):
        if not ok or k < 0 or tr != best[k]:
            continue
        out.setdefault(int(k), []).append(Observation(p, float(s), camera, team=d.team, feat=dequantize(d.feat)))
    return out


def build_tracks(court: Court, detections: dict[str, list[Detection]], calibrations: dict[str, Calibration | CalibrationTimeline],
                 offsets: dict | None = None, fps_out: float = 10.0, gate: float = 1.0, ids: str = "appearance") -> TrackSet:
    """ids="appearance": short unambiguous tracklets re-linked over the whole match with appearance + motion
    (pitch_tracker.tracklets / associate). ids="motion": the older motion-only tracker + gap stitching."""
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
    players = pitch_tracker.split_jumps(players, fps_out)            # impossible jumps: glitch or two people
    ts = TrackSet(court.length, court.width, fps_out, players)
    ts.compute_stats()
    return ts
