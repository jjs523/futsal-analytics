"""Futsal pitch geometry from the Futsal Laws of the Game, Law 1 (The Pitch).

Length (touch line) 25-42 m, international 38-42 m. Width (goal line) 15-25 m, international 18-25 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CENTRE_CIRCLE_R = 3.0
PENALTY_AREA_R = 6.0        # quarter circles, radius measured from the OUTSIDE of each post
POST_OUTSIDE = 1.58         # 3 m between posts (inside) + 8 cm posts -> 3.16 m joining line
PENALTY_MARK = 6.0
SECOND_PENALTY_MARK = 10.0
GOAL_WIDTH, GOAL_HEIGHT, GOAL_DEPTH = 3.0, 2.0, 1.0
SUB_ZONE_FROM_HALFWAY, SUB_ZONE_LENGTH, SUB_MARK_LENGTH = 5.0, 5.0, 0.8
CORNER_ARC_R = 0.25


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> np.ndarray:
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


@dataclass(frozen=True)
class Court:
    length: float = 40.0
    width: float = 20.0

    @classmethod
    def parse(cls, text: str) -> "Court":
        """'40x20' -> Court(40, 20)."""
        length, width = (float(v) for v in text.lower().split("x"))
        return cls(length, width)

    @property
    def centre(self) -> tuple[float, float]:
        return self.length / 2, self.width / 2

    @property
    def corners(self) -> np.ndarray:
        L, W = self.length, self.width
        return np.array([[0, 0], [L, 0], [L, W], [0, W]], float)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return ((xy[:, 0] >= -margin) & (xy[:, 0] <= self.length + margin)
                & (xy[:, 1] >= -margin) & (xy[:, 1] <= self.width + margin))

    def markings(self) -> list[np.ndarray]:
        """Every painted line as a 2D polyline (N x 2)."""
        L, W = self.length, self.width
        cy = W / 2
        lines = [self.corners[[0, 1, 2, 3, 0]],
                 np.array([[L / 2, 0], [L / 2, W]]),
                 _arc(L / 2, cy, CENTRE_CIRCLE_R, 0, 360, 80)]
        for gx, s in ((0.0, 1), (L, -1)):                      # s points into the pitch
            if s > 0:
                top, bot = _arc(gx, cy + POST_OUTSIDE, PENALTY_AREA_R, 90, 0), _arc(gx, cy - POST_OUTSIDE, PENALTY_AREA_R, 0, -90)
            else:
                top, bot = _arc(gx, cy + POST_OUTSIDE, PENALTY_AREA_R, 90, 180), _arc(gx, cy - POST_OUTSIDE, PENALTY_AREA_R, 180, 270)
            joining = np.array([[gx + s * PENALTY_AREA_R, cy + POST_OUTSIDE], [gx + s * PENALTY_AREA_R, cy - POST_OUTSIDE]])
            lines += [top, joining, bot]
        for (cx, cyy), a0 in (((0, 0), 0), ((L, 0), 90), ((L, W), 180), ((0, W), 270)):
            lines.append(_arc(cx, cyy, CORNER_ARC_R, a0, a0 + 90, 10))
        for x in self.substitution_marks_x():
            lines.append(np.array([[x, -SUB_MARK_LENGTH / 2], [x, SUB_MARK_LENGTH / 2]]))
        return lines

    def substitution_marks_x(self) -> list[float]:
        h = self.length / 2
        a, b = SUB_ZONE_FROM_HALFWAY, SUB_ZONE_FROM_HALFWAY + SUB_ZONE_LENGTH
        return [h - b, h - a, h + a, h + b]

    def spots(self) -> np.ndarray:
        """Centre mark, penalty marks (6 m) and second penalty marks (10 m)."""
        L, cy = self.length, self.width / 2
        return np.array([[L / 2, cy], [PENALTY_MARK, cy], [SECOND_PENALTY_MARK, cy],
                         [L - PENALTY_MARK, cy], [L - SECOND_PENALTY_MARK, cy]])

    def keypoints(self) -> dict[str, tuple[float, float]]:
        """Named line intersections / marks a user can tap on a video frame to calibrate a camera."""
        L, W = self.length, self.width
        cy = W / 2
        k: dict[str, tuple[float, float]] = {
            "corner_bl": (0, 0), "corner_br": (L, 0), "corner_tr": (L, W), "corner_tl": (0, W),
            "halfway_bottom": (L / 2, 0), "halfway_top": (L / 2, W),
            "centre": (L / 2, cy), "circle_bottom": (L / 2, cy - CENTRE_CIRCLE_R), "circle_top": (L / 2, cy + CENTRE_CIRCLE_R),
        }
        for side, gx, s in (("left", 0.0, 1), ("right", L, -1)):
            k[f"penalty_{side}"] = (gx + s * PENALTY_MARK, cy)
            k[f"second_penalty_{side}"] = (gx + s * SECOND_PENALTY_MARK, cy)
            k[f"area_goalline_top_{side}"] = (gx, cy + POST_OUTSIDE + PENALTY_AREA_R)
            k[f"area_goalline_bottom_{side}"] = (gx, cy - POST_OUTSIDE - PENALTY_AREA_R)
            k[f"area_line_top_{side}"] = (gx + s * PENALTY_AREA_R, cy + POST_OUTSIDE)
            k[f"area_line_bottom_{side}"] = (gx + s * PENALTY_AREA_R, cy - POST_OUTSIDE)
            k[f"post_top_{side}"] = (gx, cy + GOAL_WIDTH / 2)
            k[f"post_bottom_{side}"] = (gx, cy - GOAL_WIDTH / 2)
        for i, x in enumerate(self.substitution_marks_x()):
            k[f"sub_mark_{i}"] = (x, 0.0)
        return {name: (float(x), float(y)) for name, (x, y) in k.items()}

    def goal_frames(self) -> list[np.ndarray]:
        """3D polylines (N x 3) of both goals: posts + crossbar and a simple net frame."""
        out, cy = [], self.width / 2
        y0, y1, h = cy - GOAL_WIDTH / 2, cy + GOAL_WIDTH / 2, GOAL_HEIGHT
        for x0, dx in ((0.0, -GOAL_DEPTH), (self.length, GOAL_DEPTH)):
            out += [np.array([(x0, y0, 0), (x0, y0, h), (x0, y1, h), (x0, y1, 0)], float),
                    np.array([(x0, y0, h), (x0 + dx, y0, h * .6), (x0 + dx, y0, 0), (x0, y0, 0)], float),
                    np.array([(x0, y1, h), (x0 + dx, y1, h * .6), (x0 + dx, y1, 0), (x0, y1, 0)], float),
                    np.array([(x0 + dx, y0, 0), (x0 + dx, y1, 0)], float),
                    np.array([(x0 + dx, y0, h * .6), (x0 + dx, y1, h * .6)], float)]
        return out

    def to_json(self) -> dict:
        """Everything a client needs to draw the pitch."""
        return {
            "length": self.length, "width": self.width,
            "markings": [m.round(3).tolist() for m in self.markings()],
            "spots": self.spots().round(3).tolist(),
            "goals": [{"x": 0.0, "y0": self.width / 2 - GOAL_WIDTH / 2, "y1": self.width / 2 + GOAL_WIDTH / 2, "depth": -GOAL_DEPTH},
                      {"x": self.length, "y0": self.width / 2 - GOAL_WIDTH / 2, "y1": self.width / 2 + GOAL_WIDTH / 2, "depth": GOAL_DEPTH}],
            "keypoints": self.keypoints(),
        }
