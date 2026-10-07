"""Persistence: SQLite for metadata + a directory tree for videos and results.

data/
  futsal.db
  matches/<match_id>/<camera>/seg_0000.mp4        uploaded video segments (Track A: a single seg_0000)
  matches/<match_id>/<camera>/seg_0000.det.json   pixel detections of that segment
  matches/<match_id>/tracks.json                  final result for the viewer

Swap for Postgres + S3 later without touching the API: only this module knows where bytes live.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  id TEXT PRIMARY KEY, title TEXT, court TEXT NOT NULL, status TEXT NOT NULL,
  error TEXT, created REAL NOT NULL, offsets TEXT
);
CREATE TABLE IF NOT EXISTS segments (
  match_id TEXT NOT NULL, camera TEXT NOT NULL, seq INTEGER NOT NULL,
  received INTEGER NOT NULL DEFAULT 0, complete INTEGER NOT NULL DEFAULT 0,
  t0 REAL, duration REAL, processed INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (match_id, camera, seq)
);
CREATE TABLE IF NOT EXISTS calibrations (
  match_id TEXT NOT NULL, camera TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (match_id, camera)
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, created REAL NOT NULL, finished REAL
);
"""

# match status: created -> uploading -> processing -> done | failed


class Store:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "matches").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "futsal.db", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # ---- matches ----
    def create_match(self, court: str, title: str | None) -> dict:
        mid = uuid.uuid4().hex[:12]
        self.q("INSERT INTO matches (id, title, court, status, created) VALUES (?,?,?,?,?)", (mid, title, court, "created", time.time()))
        return self.match(mid)

    def match(self, mid: str) -> dict | None:
        rows = self.q("SELECT * FROM matches WHERE id=?", (mid,))
        if not rows:
            return None
        m = dict(rows[0])
        m["offsets"] = json.loads(m["offsets"]) if m["offsets"] else {}
        m["segments"] = [dict(r) for r in self.q("SELECT camera, seq, received, complete, t0, duration, processed FROM segments WHERE match_id=? ORDER BY camera, seq", (mid,))]
        m["calibrated"] = [r["camera"] for r in self.q("SELECT camera FROM calibrations WHERE match_id=?", (mid,))]
        m["has_tracks"] = self.tracks_path(mid).exists()
        return m

    def matches(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT id, title, court, status, created FROM matches ORDER BY created DESC")]

    def set_status(self, mid: str, status: str, error: str | None = None) -> None:
        self.q("UPDATE matches SET status=?, error=? WHERE id=?", (status, error, mid))

    def set_offsets(self, mid: str, offsets: dict) -> None:
        self.q("UPDATE matches SET offsets=? WHERE id=?", (json.dumps(offsets), mid))

    # ---- files ----
    def match_dir(self, mid: str) -> Path:
        return self.root / "matches" / mid

    def segment_path(self, mid: str, cam: str, seq: int) -> Path:
        p = self.match_dir(mid) / cam / f"seg_{seq:04d}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def detections_path(self, mid: str, cam: str, seq: int) -> Path:
        return self.segment_path(mid, cam, seq).with_suffix(".det.json")

    def tracks_path(self, mid: str) -> Path:
        d = self.match_dir(mid)
        d.mkdir(parents=True, exist_ok=True)
        return d / "tracks.json"

    # ---- segments (resumable upload: the client asks for the offset, then appends from there) ----
    def segment(self, mid: str, cam: str, seq: int) -> dict | None:
        rows = self.q("SELECT * FROM segments WHERE match_id=? AND camera=? AND seq=?", (mid, cam, seq))
        return dict(rows[0]) if rows else None

    def append(self, mid: str, cam: str, seq: int, offset: int, data: bytes) -> int:
        with self._lock:
            seg = self.segment(mid, cam, seq)
            if seg is None:
                self.q("INSERT INTO segments (match_id, camera, seq) VALUES (?,?,?)", (mid, cam, seq))
                seg = self.segment(mid, cam, seq)
            if seg["complete"]:
                raise ValueError("segment already complete")
            if offset != seg["received"]:
                raise ValueError(f"offset mismatch: server has {seg['received']} bytes")
            with open(self.segment_path(mid, cam, seq), "ab" if offset else "wb") as f:
                f.write(data)
            received = offset + len(data)
            self.q("UPDATE segments SET received=? WHERE match_id=? AND camera=? AND seq=?", (received, mid, cam, seq))
            return received

    def complete_segment(self, mid: str, cam: str, seq: int, t0: float | None) -> None:
        self.q("UPDATE segments SET complete=1, t0=? WHERE match_id=? AND camera=? AND seq=?", (t0, mid, cam, seq))

    def mark_processed(self, mid: str, cam: str, seq: int, duration: float) -> None:
        self.q("UPDATE segments SET processed=1, duration=? WHERE match_id=? AND camera=? AND seq=?", (duration, mid, cam, seq))

    # ---- calibration ----
    def save_calibration(self, mid: str, cam: str, data: dict) -> None:
        self.q("INSERT OR REPLACE INTO calibrations (match_id, camera, data) VALUES (?,?,?)", (mid, cam, json.dumps(data)))

    def calibrations(self, mid: str) -> dict[str, dict]:
        return {r["camera"]: json.loads(r["data"]) for r in self.q("SELECT camera, data FROM calibrations WHERE match_id=?", (mid,))}

    # ---- jobs ----
    def enqueue(self, mid: str, kind: str, payload: dict) -> int:
        self.q("INSERT INTO jobs (match_id, kind, payload, status, created) VALUES (?,?,?,?,?)", (mid, kind, json.dumps(payload), "queued", time.time()))
        return int(self.q("SELECT last_insert_rowid() AS i")[0]["i"])

    def next_job(self) -> dict | None:
        with self._lock:
            rows = self.q("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")
            if not rows:
                return None
            self.q("UPDATE jobs SET status='running' WHERE id=?", (rows[0]["id"],))
            j = dict(rows[0]); j["payload"] = json.loads(j["payload"])
            return j

    def finish_job(self, jid: int, error: str | None = None) -> None:
        self.q("UPDATE jobs SET status=?, error=?, finished=? WHERE id=?", ("failed" if error else "done", error, time.time(), jid))

    def jobs(self, mid: str) -> list[dict]:
        return [dict(r) for r in self.q("SELECT id, kind, payload, status, error FROM jobs WHERE match_id=? ORDER BY id", (mid,))]

    def requeue_running(self) -> None:
        """After a crash/restart, jobs that were mid-flight run again (all jobs are idempotent)."""
        self.q("UPDATE jobs SET status='queued' WHERE status='running'")
