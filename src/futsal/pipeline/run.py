"""Match-level assembly: per-camera pixel detections -> one TrackSet."""
from __future__ import annotations

import numpy as np

from ..court import Court
from ..fusion import Observation, fuse_frame
from ..homography import Calibration
from ..tracks import TrackSet
from . import pitch_tracker
from .detect import Detection, dequantize

DETECTOR_JITTER_PX = 3.0      # foot-point noise of the detector, used to weight cameras


def clock(sync) -> tuple[float, float]:
    """A camera's sync entry: a plain offset in seconds, or {"offset": s, "drift": ratio} from futsal.sync.align."""
    if isinstance(sync, dict):
        return float(sync.get("offset", 0.0)), float(sync.get("drift", 0.0))
    return float(sync or 0.0), 0.0


def to_observations(dets: list[Detection], cal: Calibration, camera: str, offset_s, fps_out: float,
                    court: Court, margin: float = 1.0) -> dict[int, list[Observation]]:
    """Pixels -> pitch, map the camera's clock onto the reference clock, bucket to the output frame grid.
    Points far outside the pitch (spectators, subs on the bench) are dropped."""
    offset, drift = clock(offset_s)
    if not dets:
        return {}
    uv = np.array([[d.u, d.v] for d in dets])
    xy = cal.to_pitch(uv)
    sig = DETECTOR_JITTER_PX * cal.metres_per_pixel(uv)
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


def build_tracks(court: Court, detections: dict[str, list[Detection]], calibrations: dict[str, Calibration],
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
    ts = TrackSet(court.length, court.width, fps_out, players)
    ts.compute_stats()
    return ts
