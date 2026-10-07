"""Pinhole camera model for a phone on a tripod. Used by the simulator and for sanity checks of real calibrations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLAYER_HEIGHT = 1.75


@dataclass
class Camera:
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 world -> camera rotation (rows: right, down, forward)
    C: np.ndarray            # camera centre in world coordinates (x, y, z)
    width: int = 1920
    height: int = 1080

    @classmethod
    def on_tripod(cls, position_xy, height: float, yaw_deg: float, hfov_deg: float,
                  width: int = 1920, height_px: int = 1080, horizon_margin_deg: float = 2.0) -> "Camera":
        """Landscape phone at `position_xy`, `height` m above the floor, looking along `yaw_deg`
        (0 = +x, counter-clockwise). It is tilted down so the top edge of the frame sits
        `horizon_margin_deg` above the horizon, which keeps the heads of far players in frame."""
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height_px / 2], [0, 0, 1.0]])
        vfov = 2 * np.arctan((height_px / 2) / f)
        pitch = vfov / 2 - np.radians(horizon_margin_deg)
        yaw = np.radians(yaw_deg)
        fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        down = np.cross(fwd, right)
        C = np.array([position_xy[0], position_xy[1], height], float)
        return cls(K=K, R=np.stack([right, down, fwd]), C=C, width=width, height=height_px)

    @property
    def yaw_deg(self) -> float:
        fwd = self.R[2]
        return float(np.degrees(np.arctan2(fwd[1], fwd[0])))

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan((self.width / 2) / self.K[0, 0])))

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """World points (N x 2 on the floor, or N x 3) -> pixel coords (N x 2) and an in-frame mask."""
        P = np.atleast_2d(np.asarray(points, float))
        if P.shape[1] == 2:
            P = np.hstack([P, np.zeros((len(P), 1))])
        Xc = (self.R @ (P - self.C).T).T
        z = Xc[:, 2]
        uvw = (self.K @ Xc.T).T
        uv = uvw[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv, ok

    def sees_player(self, xy) -> np.ndarray:
        """True where both the feet and the head of a standing player are inside the frame."""
        xy = np.atleast_2d(np.asarray(xy, float))
        _, feet = self.project(xy)
        _, head = self.project(np.hstack([xy, np.full((len(xy), 1), PLAYER_HEIGHT)]))
        return feet & head

    def floor_homography(self) -> np.ndarray:
        """Exact image -> floor (z = 0) homography implied by the model."""
        t = -self.R @ self.C
        floor_to_image = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], t])
        H = np.linalg.inv(floor_to_image)
        return H / H[2, 2]


def yaw_towards(src_xy, dst_xy, twist_deg: float = 0.0) -> float:
    """Yaw (deg) pointing from src to dst, rotated counter-clockwise by `twist_deg`."""
    d = np.asarray(dst_xy, float) - np.asarray(src_xy, float)
    return float(np.degrees(np.arctan2(d[1], d[0])) + twist_deg)
