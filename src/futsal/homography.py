"""Image <-> pitch mapping for a fixed camera, fitted from tapped pitch keypoints."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .court import Court


@dataclass
class Calibration:
    H: np.ndarray                         # undistorted image (u, v) -> pitch (x, y) in metres
    rms_px: float                         # reprojection RMS of the used keypoints, in pixels
    used: list[str] = field(default_factory=list)
    k1: float = 0.0                       # radial lens distortion (0 = none), see distort()
    centre: tuple[float, float] = (0.0, 0.0)   # distortion centre (image centre), pixels
    norm: float = 1.0                     # radius normaliser (half the image diagonal), pixels

    @property
    def H_inv(self) -> np.ndarray:
        Hi = np.linalg.inv(self.H)
        return Hi / Hi[2, 2]

    def distort(self, uv) -> np.ndarray:
        """Ideal pinhole pixel -> where the lens actually puts it: p_d = c + (p - c)(1 + k1 r^2), r = |p - c| / norm.
        Phone main cameras are mostly corrected in software, but a few tens of pixels remain at the edges."""
        uv = np.atleast_2d(np.asarray(uv, float))
        if not self.k1:
            return uv
        d = uv - self.centre
        r2 = np.sum(d * d, 1, keepdims=True) / self.norm ** 2
        return self.centre + d * (1 + self.k1 * r2)

    def undistort(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        if not self.k1:
            return uv
        d = uv - self.centre
        p = d.copy()
        for _ in range(20):                       # fixed point: p = d / (1 + k1 |p|^2)
            p = d / (1 + self.k1 * np.sum(p * p, 1, keepdims=True) / self.norm ** 2)
        return self.centre + p

    def to_pitch(self, uv) -> np.ndarray:
        return image_to_pitch(self.H, self.undistort(uv))

    def to_image(self, xy) -> np.ndarray:
        return self.distort(image_to_pitch(self.H_inv, xy))

    def metres_per_pixel(self, uv) -> np.ndarray:
        if not self.k1:
            return metres_per_pixel(self.H, uv)
        uv = np.atleast_2d(np.asarray(uv, float))
        g0 = self.to_pitch(uv)
        J = np.stack([self.to_pitch(uv + [1.0, 0.0]) - g0, self.to_pitch(uv + [0.0, 1.0]) - g0], axis=2)
        return np.linalg.svd(J, compute_uv=False)[:, 0]

    def shifted(self, dx: float, dy: float) -> "Calibration":
        """The same camera after it was nudged and the picture slid by (dx, dy) pixels (what was at u is now at
        u + dx). Good for small knocks of the tripod; re-tap the keypoints if the view also rotated or zoomed."""
        T = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])
        return Calibration(self.H @ T, self.rms_px, list(self.used), self.k1,
                           (self.centre[0] + dx, self.centre[1] + dy), self.norm)

    def to_json(self) -> dict:
        d = {"H": self.H.tolist(), "rms_px": self.rms_px, "used": self.used}
        if self.k1:
            d |= {"k1": self.k1, "centre": list(self.centre), "norm": self.norm}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Calibration":
        return cls(H=np.asarray(d["H"], float), rms_px=float(d["rms_px"]), used=list(d.get("used", [])),
                   k1=float(d.get("k1", 0.0)), centre=tuple(d.get("centre", (0.0, 0.0))), norm=float(d.get("norm", 1.0)))


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


def calibrate(taps: dict[str, tuple[float, float]], court: Court, ransac_px: float | None = None,
              image_size: tuple[int, int] | None = None, distortion: bool | None = None) -> Calibration:
    """Fit a homography from {keypoint name: (u, v) pixel} taps. Needs >= 4 keypoints, ideally spread out;
    with >= 6 taps pass `ransac_px` (e.g. 8) to reject a mis-tapped point.
    With `image_size` and >= 8 taps, a radial lens distortion k1 is also fitted (distortion=None: kept only if it
    clearly explains the taps better, e.g. near-side lines bending at the frame edge; True / False to force)."""
    kp = court.keypoints()
    unknown = sorted(set(taps) - set(kp))
    if unknown:
        raise ValueError(f"unknown keypoints: {unknown}")
    names = [n for n in taps if n in kp]
    if len(names) < 4:
        raise ValueError(f"need at least 4 keypoints, got {len(names)}")
    img = np.array([taps[n] for n in names], np.float64)
    world = np.array([kp[n] for n in names], np.float64)

    def fit(k1: float, centre, norm):
        cal0 = Calibration(np.eye(3), 0.0, [], k1, centre, norm)
        und = cal0.undistort(img)
        method = cv2.RANSAC if (ransac_px and len(names) >= 6) else 0
        # Fit pitch -> image so the residual is measured in pixels, where the tap noise lives; fitting
        # image -> pitch would let far keypoints (metres per pixel is large there) dominate the fit.
        H_wi, mask = cv2.findHomography(world, und, method, ransac_px or 3.0)
        if H_wi is None:
            return None
        H = np.linalg.inv(H_wi)
        cal = Calibration(H / H[2, 2], 0.0, [], k1, centre, norm)
        err = np.linalg.norm(cal.to_image(world) - img, axis=1)          # in real (distorted) pixels
        keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(names), bool)
        if ransac_px and len(names) >= 6:
            keep = err <= ransac_px
            if keep.sum() >= 4:                                          # refit on the inliers only
                H_wi, _ = cv2.findHomography(world[keep], und[keep], 0)
                H = np.linalg.inv(H_wi)
                cal = Calibration(H / H[2, 2], 0.0, [], k1, centre, norm)
                err = np.linalg.norm(cal.to_image(world) - img, axis=1)
                keep = err <= ransac_px
        cap = (ransac_px or 8.0) ** 2
        cost = float(np.sum(np.minimum(err ** 2, cap)))                  # robust: a bad tap costs at most cap
        cal.rms_px = float(np.sqrt(np.mean(err[keep] ** 2))) if keep.any() else float("inf")
        cal.used = [n for n, k in zip(names, keep) if k]
        return cost, cal

    base = fit(0.0, (0.0, 0.0), 1.0)
    if base is None:
        raise ValueError("homography fit failed (keypoints collinear?)")
    best = base
    if image_size and distortion is not False and len(names) >= 8:
        w, h = image_size
        centre, norm = (w / 2, h / 2), float(np.hypot(w, h) / 2)
        for k1 in np.arange(-0.30, 0.301, 0.01):
            r = fit(float(k1), centre, norm)
            if r and r[0] < best[0]:
                best = r
        for k1 in np.linspace(best[1].k1 - 0.01, best[1].k1 + 0.01, 21):   # refine around the best
            r = fit(float(k1), centre, norm)
            if r and r[0] < best[0]:
                best = r
        if distortion is None and best is not base and best[0] > 0.6 * base[0]:
            best = base                                                  # not clearly better: keep the plain fit
    return best[1]


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
