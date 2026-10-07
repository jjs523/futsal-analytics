"""The result format shared by the pipeline, the server and the web viewer (tracks.json).

{
  "version": 1,
  "court": {"length": 40, "width": 20},
  "fps": 10, "start": 0.0, "n_frames": 30000,
  "players": [{"id": 1, "team": "A", "name": "...", "x": [..], "y": [..], "stats": {...}}]
}
x / y are per-frame metres; null where the player was not seen. Positions are rounded to cm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import metrics


@dataclass
class PlayerTrack:
    id: int
    team: str
    xy: np.ndarray                    # (n_frames, 2), NaN where unseen
    name: str | None = None
    stats: dict = field(default_factory=dict)


@dataclass
class TrackSet:
    length: float
    width: float
    fps: float
    players: list[PlayerTrack]
    start: float = 0.0

    @property
    def n_frames(self) -> int:
        return len(self.players[0].xy) if self.players else 0

    def compute_stats(self) -> None:
        for p in self.players:
            p.stats = metrics.summary(p.xy, 1.0 / self.fps)

    def to_json(self) -> dict:
        def col(v):
            return [None if not np.isfinite(a) else round(float(a), 2) for a in v]
        return {
            "version": 1, "court": {"length": self.length, "width": self.width},
            "fps": self.fps, "start": self.start, "n_frames": self.n_frames,
            "players": [{"id": p.id, "team": p.team, "name": p.name, "stats": p.stats,
                         "x": col(p.xy[:, 0]), "y": col(p.xy[:, 1])} for p in self.players],
        }

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, separators=(",", ":"))

    @classmethod
    def from_json(cls, d: dict) -> "TrackSet":
        players = []
        for p in d["players"]:
            xy = np.array([[np.nan if a is None else a, np.nan if b is None else b] for a, b in zip(p["x"], p["y"])], float)
            players.append(PlayerTrack(id=p["id"], team=p["team"], xy=xy.reshape(-1, 2), name=p.get("name"), stats=p.get("stats", {})))
        return cls(length=d["court"]["length"], width=d["court"]["width"], fps=d["fps"], players=players, start=d.get("start", 0.0))

    @classmethod
    def load(cls, path: str) -> "TrackSet":
        with open(path) as f:
            return cls.from_json(json.load(f))
