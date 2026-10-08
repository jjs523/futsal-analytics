"""How far apart do the two cameras place the same player? Match cam1/cam2 boxes per frame (Hungarian, <= 3 m,
confident boxes) and look at the position difference as a function of where on the pitch it happens."""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Data

data = Data()
a, b = data.cams["cam1"], data.cams["cam2"]
diffs, where, sims = [], [], []
for k in range(data.n):
    ia, ib = a.boxes_at(k, 0.5), b.boxes_at(k, 0.5)
    if not len(ia) or not len(ib):
        continue
    D = np.linalg.norm(a.xy[ia][:, None] - b.xy[ib][None], axis=2)
    r, c = linear_sum_assignment(D)
    for i, j in zip(r, c):
        if D[i, j] < 3.0:
            diffs.append(b.xy[ib[j]] - a.xy[ia[i]])
            where.append((a.xy[ia[i]] + b.xy[ib[j]]) / 2)
            sims.append(float(a.reid[ia[i]] @ b.reid[ib[j]]))
diffs, where, sims = np.array(diffs), np.array(where), np.array(sims)
d = np.linalg.norm(diffs, axis=1)
print(f"matched pairs: {len(d)}; |cam2 - cam1| median {np.median(d):.2f} m, p75 {np.percentile(d, 75):.2f}, p90 {np.percentile(d, 90):.2f}")
print(f"mean offset (dx, dy) = {diffs.mean(0).round(2)}; ReID cos sim of matched pairs median {np.median(sims):.2f}")
for x0 in range(0, 40, 8):
    for y0 in range(0, 20, 10):
        m = (where[:, 0] >= x0) & (where[:, 0] < x0 + 8) & (where[:, 1] >= y0) & (where[:, 1] < y0 + 10)
        if m.sum() > 30:
            print(f"  x {x0:2d}-{x0 + 8:2d}, y {y0:2d}-{y0 + 10:2d}: n={m.sum():5d}  median |d| {np.median(d[m]):.2f} m, "
                  f"mean (dx,dy) {diffs[m].mean(0).round(2)}")
# within-camera: how much does a single camera's position jump between consecutive frames for the same player?
for cam in (a, b):
    j = []
    for k in range(data.n - 1):
        i0, i1 = cam.boxes_at(k, 0.5), cam.boxes_at(k + 1, 0.5)
        if len(i0) and len(i1):
            D = np.linalg.norm(cam.xy[i0][:, None] - cam.xy[i1][None], axis=2)
            r, c = linear_sum_assignment(D)
            j += [D[x, y] for x, y in zip(r, c) if D[x, y] < 2.0]
    j = np.array(j)
    print(f"{cam.name}: frame-to-frame step median {np.median(j):.2f} m, p90 {np.percentile(j, 90):.2f} m")
