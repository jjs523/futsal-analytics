"""Can player foot correspondences align the two cameras better than the tapped keypoints?
Fit maps cam2-pitch -> cam1-pitch (affine, homography, quadratic) on confident ReID-consistent pairs, with a time split
(first half fit, second half test) so the residual is honest."""
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Data

data = Data()
a, b = data.cams["cam1"], data.cams["cam2"]
P, Q, K = [], [], []
for k in range(data.n):
    ia, ib = a.boxes_at(k, 0.5), b.boxes_at(k, 0.5)
    if not len(ia) or not len(ib):
        continue
    D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
    S = a.reid[ia] @ b.reid[ib].T
    r, c = linear_sum_assignment(D - 0.5 * S)
    for i, j in zip(r, c):
        # mutual nearest + appearance agreement keeps mostly true pairs
        if D[i, j] < 2.5 and S[i, j] > 0.7 and D[i].argmin() == j and D[:, j].argmin() == i:
            P.append(b.xy[ib[j]]); Q.append(a.xy[ia[i]]); K.append(k)
P, Q, K = np.array(P), np.array(Q), np.array(K)
fit, test = K < data.n / 2, K >= data.n / 2
print(f"pairs {len(P)} (fit {fit.sum()}, test {test.sum()}); raw |d| test median {np.median(np.linalg.norm(P[test] - Q[test], axis=1)):.3f} m")


def quad_feats(X):
    x, y = X[:, 0] / 40, X[:, 1] / 20
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], 1)


def report(name, f):
    for nm, m in (("fit", fit), ("test", test)):
        r = np.linalg.norm(f(P[m]) - Q[m], axis=1)
        print(f"  {name:10s} {nm}: median {np.median(r):.3f} m, p90 {np.percentile(r, 90):.3f} m")


A, inl = cv2.estimateAffine2D(P[fit], Q[fit], ransacReprojThreshold=0.6)
report("affine", lambda X: X @ A[:, :2].T + A[:, 2])
H, _ = cv2.findHomography(P[fit], Q[fit], cv2.RANSAC, 0.6)
report("homography", lambda X: cv2.perspectiveTransform(X.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2))
F = quad_feats(P[fit])
W = np.linalg.lstsq(F, Q[fit] - P[fit], rcond=None)[0]
for _ in range(3):                                   # trim outliers and refit
    r = np.linalg.norm(P[fit] + F @ W - Q[fit], axis=1)
    keep = r < np.percentile(r, 85)
    W = np.linalg.lstsq(F[keep], (Q[fit] - P[fit])[keep], rcond=None)[0]
report("cubic", lambda X: X + quad_feats(X) @ W)
