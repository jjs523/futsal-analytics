"""Image <-> pitch mapping for a fixed camera, fitted from tapped pitch keypoints."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .court import Court


@dataclass
class Calibration:
    H: np.ndarray                         # image (u, v) -> pitch (x, y) in metres
    rms_px: float                         # reprojection RMS of the used keypoints, in pixels
    used: list[str] = field(default_factory=list)

    @property
    def H_inv(self) -> np.ndarray:
        Hi = np.linalg.inv(self.H)
        return Hi / Hi[2, 2]

    def to_pitch(self, uv) -> np.ndarray:
        return image_to_pitch(self.H, uv)

    def to_image(self, xy) -> np.ndarray:
        return image_to_pitch(self.H_inv, xy)

    def metres_per_pixel(self, uv) -> np.ndarray:
        return metres_per_pixel(self.H, uv)

    def shifted(self, dx: float, dy: float) -> "Calibration":
        """The same camera after it was nudged and the picture slid by (dx, dy) pixels (what was at u is now at
        u + dx). Good for small knocks of the tripod; re-tap the keypoints if the view also rotated or zoomed."""
        T = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])
        return Calibration(self.H @ T, self.rms_px, list(self.used))

    def to_json(self) -> dict:
        return {"H": self.H.tolist(), "rms_px": self.rms_px, "used": self.used}

    @classmethod
    def from_json(cls, d: dict) -> "Calibration":
        return cls(H=np.asarray(d["H"], float), rms_px=float(d["rms_px"]), used=list(d.get("used", [])))


def image_to_pitch(H: np.ndarray, uv) -> np.ndarray:
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(uv, H).reshape(-1, 2)


def metres_per_pixel(H: np.ndarray, uv) -> np.ndarray:
    """Worst-case ground displacement (m) caused by a 1 px error at each image point: the largest singular
    value of the local image->pitch Jacobian. Near the horizon this is the depth direction and gets large."""
    uv = np.atleast_2d(np.asarray(uv, float))
    g0 = image_to_pitch(H, uv)
    J = np.stack([image_to_pitch(H, uv + [1.0, 0.0]) - g0, image_to_pitch(H, uv + [0.0, 1.0]) - g0], axis=2)
    return np.linalg.svd(J, compute_uv=False)[:, 0]


def calibrate(taps: dict[str, tuple[float, float]], court: Court, ransac_px: float | None = None) -> Calibration:
    """Fit a homography from {keypoint name: (u, v) pixel} taps. Needs >= 4 keypoints, ideally spread out;
    with >= 6 taps pass `ransac_px` (e.g. 8) to reject a mis-tapped point."""
    kp = court.keypoints()
    unknown = sorted(set(taps) - set(kp))
    if unknown:
        raise ValueError(f"unknown keypoints: {unknown}")
    names = [n for n in taps if n in kp]
    if len(names) < 4:
        raise ValueError(f"need at least 4 keypoints, got {len(names)}")
    img = np.array([taps[n] for n in names], np.float64)
    world = np.array([kp[n] for n in names], np.float64)
    method = cv2.RANSAC if (ransac_px and len(names) >= 6) else 0
    # Fit pitch -> image so the residual is measured in pixels, where the tap noise lives; fitting
    # image -> pitch would let far keypoints (metres per pixel is large there) dominate the fit.
    H_wi, mask = cv2.findHomography(world, img, method, ransac_px or 3.0)
    if H_wi is None:
        raise ValueError("homography fit failed (keypoints collinear?)")
    H = np.linalg.inv(H_wi)
    H = H / H[2, 2]
    keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(names), bool)
    back = image_to_pitch(H_wi, world[keep])
    rms = float(np.sqrt(np.mean(np.sum((back - img[keep]) ** 2, 1))))
    return Calibration(H=H, rms_px=rms, used=[n for n, k in zip(names, keep) if k])


# A camera that was moved during the match: [(from_s, Calibration), ...] sorted by start time on that camera's own
# clock (video seconds), e.g. [(0, before), (160.0, after)] for a tripod knocked at 2:40.
CalibrationTimeline = list[tuple[float, Calibration]]


def calibration_at(cal: "Calibration | CalibrationTimeline", t: float) -> Calibration:
    if isinstance(cal, Calibration):
        return cal
    current = cal[0][1]
    for start, c in cal:
        if t >= start:
            current = c
    return current


def timeline_from_json(d) -> "Calibration | CalibrationTimeline":
    """{"H": ...} for one calibration, or [{"from": seconds, "H": ...}, ...] when the camera moved."""
    if isinstance(d, list):
        return sorted(((float(x.get("from", 0.0)), Calibration.from_json(x)) for x in d), key=lambda p: p[0])
    return Calibration.from_json(d)


def framing(cal: Calibration, width: int, height: int, court: Court, step: float = 1.0) -> dict:
    """How much of the court this camera sees, and which way to turn it to see more. Pitch points on a `step` m
    grid are projected into the image; those outside tell the direction (e.g. mostly above the top edge -> tilt up).
    Used right after the keypoint taps so that a badly aimed phone is caught before the match, not after."""
    xs = np.arange(step / 2, court.length, step)
    ys = np.arange(step / 2, court.width, step)
    grid = np.array([(x, y) for x in xs for y in ys])
    uv = cal.to_image(grid)
    Hi = cal.H_inv
    w = grid @ Hi[2, :2] + Hi[2, 2]                       # > 0: in front of the camera
    front = w > 0
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    out = ~inside
    counts = {"up": int(np.sum(out & front & (uv[:, 1] < 0))), "down": int(np.sum(out & front & (uv[:, 1] >= height))),
              "left": int(np.sum(out & front & (uv[:, 0] < 0))), "right": int(np.sum(out & front & (uv[:, 0] >= width))),
              "behind": int(np.sum(~front))}
    advice = None
    if out.mean() > 0.05:
        side = max(("up", "down", "left", "right"), key=lambda k: counts[k])
        advice = {"up": "카메라를 위로 드세요", "down": "카메라를 아래로 숙이세요",
                  "left": "카메라를 왼쪽으로 돌리세요", "right": "카메라를 오른쪽으로 돌리세요"}[side]
    return {"visible": float(inside.mean()), "outside": counts, "advice": advice}
