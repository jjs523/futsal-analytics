"""Background job runner.

Jobs:
  segment   {camera, seq}  detect players in one uploaded segment (GPU-heavy, runs as soon as the segment arrives)
  finalize  {}             sync cameras, pixels -> pitch, fuse, track, stats, write tracks.json

This in-process thread is enough for development and the demo. In production run the same `run_job`
in a separate GPU worker process (e.g. Celery/RQ) that polls the same job table.
"""
from __future__ import annotations

import logging
import threading
import traceback

from futsal import Court
from futsal.homography import Calibration
from futsal.pipeline import detect, run

from .store import Store

log = logging.getLogger("futsal.worker")

# Swappable so tests / demos can use the synthetic colour-blob detector instead of YOLO.
DETECTOR_FACTORY = {"factory": lambda: detect.yolo_detector()}
DETECT_HZ = 10.0      # detections per second, whatever the video's frame rate (30 fps -> every 3rd frame, 60 -> 6th)


def run_job(store: Store, job: dict) -> None:
    mid, kind, p = job["match_id"], job["kind"], job["payload"]
    if kind == "segment":
        cam, seq = p["camera"], p["seq"]
        seg = store.segment(mid, cam, seq)
        t0 = seg["t0"] if seg["t0"] is not None else _implicit_t0(store, mid, cam, seq)
        dets, fps, n = detect.detect_video(str(store.segment_path(mid, cam, seq)), DETECTOR_FACTORY["factory"](), rate_hz=DETECT_HZ, t0=t0)
        detect.save(dets, str(store.detections_path(mid, cam, seq)), fps=fps, frames=n, t0=t0)
        store.mark_processed(mid, cam, seq, n / fps)
    elif kind == "finalize":
        m = store.match(mid)
        court = Court.parse(m["court"])
        cals = {cam: Calibration.from_json(c) for cam, c in store.calibrations(mid).items()}
        pending = [f"{s['camera']}#{s['seq']}" for s in m["segments"] if s["complete"] and not s["processed"]]
        if pending:
            raise RuntimeError(f"segments not processed (detection failed?): {pending}")
        dets: dict[str, list] = {}
        for s in m["segments"]:
            if s["processed"]:
                d, _ = detect.load(str(store.detections_path(mid, s["camera"], s["seq"])))
                dets.setdefault(s["camera"], []).extend(d)
        missing = sorted(set(dets) - set(cals))
        if missing:
            raise RuntimeError(f"cameras without calibration: {missing}")
        offsets = m["offsets"] or _audio_offsets(store, mid, sorted(dets))
        store.set_offsets(mid, offsets)
        ts = run.build_tracks(court, dets, cals, offsets=offsets)
        ts.dump(str(store.tracks_path(mid)))
        store.set_status(mid, "done")
    else:
        raise ValueError(f"unknown job kind {kind}")


def _audio_offsets(store: Store, mid: str, cams: list[str], min_confidence: float = 6.0) -> dict:
    """Align every camera to the first one using the audio of their first segments. Falls back to 0 s
    (videos started together) when there is no usable audio; the client can always send offsets explicitly."""
    from futsal import sync
    offsets = {c: 0.0 for c in cams}
    if len(cams) < 2:
        return offsets
    try:
        ref = sync.read_audio(str(store.segment_path(mid, cams[0], 0)))
        for cam in cams[1:]:
            res = sync.align(ref, sync.read_audio(str(store.segment_path(mid, cam, 0))), 16000)
            if res["conf_start"] >= min_confidence and (res["conf_end"] is None or res["conf_end"] >= min_confidence):
                offsets[cam] = {"offset": res["offset"], "drift": res["drift"]}
            elif res["conf_start"] >= min_confidence:
                offsets[cam] = res["offset_start"]
            else:
                log.warning("audio sync for %s not confident (%.1f); assuming 0 s", cam, res["conf_start"])
    except Exception as e:   # no audio stream, ffmpeg missing, ...
        log.warning("audio sync skipped: %s", e)
    return offsets


def _implicit_t0(store: Store, mid: str, cam: str, seq: int) -> float:
    """Segments without an explicit start time follow the previous one back-to-back."""
    if seq == 0:
        return 0.0
    prev = store.segment(mid, cam, seq - 1)
    if not prev or prev["duration"] is None:
        raise RuntimeError(f"segment {seq - 1} of {cam} not processed yet; send t0 explicitly")
    prev_t0 = prev["t0"] if prev["t0"] is not None else _implicit_t0(store, mid, cam, seq - 1)
    return prev_t0 + prev["duration"]


def drain(store: Store) -> int:
    """Run queued jobs until none are left (used by tests and the single-threaded dev server)."""
    n = 0
    while (job := store.next_job()) is not None:
        _execute(store, job); n += 1
    return n


def _execute(store: Store, job: dict) -> None:
    try:
        run_job(store, job)
        store.finish_job(job["id"])
    except Exception as e:  # keep the worker alive; surface the error on the match
        log.error("job %s failed: %s", job["id"], traceback.format_exc())
        store.finish_job(job["id"], f"{type(e).__name__}: {e}")
        store.set_status(job["match_id"], "failed", f"{job['kind']}: {e}")


class Worker(threading.Thread):
    def __init__(self, store: Store, poll_s: float = 1.0):
        super().__init__(daemon=True, name="futsal-worker")
        self.store, self.poll_s = store, poll_s
        self.wake = threading.Event()
        self.stopping = False

    def run(self) -> None:
        self.store.requeue_running()
        while not self.stopping:
            job = self.store.next_job()
            if job is None:
                self.wake.wait(self.poll_s); self.wake.clear()
                continue
            _execute(self.store, job)
