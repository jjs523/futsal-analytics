"""Detection cache for tracking experiments: every camera, 10 Hz, YOLO boxes (conf >= 0.1) + OSNet ReID + colour histograms.

python experiments/build_cache.py --start 540 --duration 360 --out experiments/cache
Writes <out>/<cam>.npz with per-box arrays:
  fid (sample index), t (camera clock, s), xyxy (N,4), conf, reid (N,512 float16, L2-normalised), color (N,30 uint8)
and <out>/meta.json (videos, offsets, window, calibration paths).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "src")
from futsal.pipeline.detect import appearance, quantize

V = {"cam1": r"C:\Users\user\dev\futsal-videos\20261008_170919.mp4", "cam2": r"C:\Users\user\dev\futsal-videos\20261008_171122.mp4"}
OFF = {"cam1": 0.0, "cam2": 124.27}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=540.0, help="cam1 clock")
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--reid", default=r"C:\Users\user\dev\models\osnet_ain_x1_0_msmt17.pt")
    ap.add_argument("--out", default="experiments/cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from ultralytics import YOLO
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    net = YOLO(a.model)
    reid = ReidAutoBackend(weights=Path(a.reid), device=torch.device("cuda:0"), half=True).model
    for cam, path in V.items():
        s = a.start - OFF[cam]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, int(round(fps / a.rate)))
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        rows = {k: [] for k in ("fid", "t", "xyxy", "conf", "reid", "color")}
        frame_t = []
        i, k, t0 = 0, 0, time.time()
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t > s + a.duration:
                break
            if i % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                frame_t.append(t)
                r = net.predict(img, classes=[0], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.conf.cpu().numpy()
                if len(b):
                    f = reid.get_features(b, img)
                    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
                    rows["fid"].append(np.full(len(b), k)); rows["t"].append(np.full(len(b), t))
                    rows["xyxy"].append(b.astype(np.float32)); rows["conf"].append(c.astype(np.float32))
                    rows["reid"].append(f.astype(np.float16))
                    rows["color"].append(np.array([quantize(appearance(img, bb)) for bb in b], np.uint8))
                k += 1
                if k % 300 == 0:
                    print(f"  {cam}: {k / a.rate:.0f}/{a.duration:.0f}s ({time.time() - t0:.0f}s)", flush=True)
            i += 1
        cap.release()
        np.savez_compressed(os.path.join(a.out, f"{cam}.npz"), frame_t=np.array(frame_t),
                            **{n: np.concatenate(v) if v else np.zeros(0) for n, v in rows.items()})
        print(f"{cam}: {k} samples, {sum(len(x) for x in rows['fid'])} boxes, {time.time() - t0:.0f}s", flush=True)
    json.dump({"videos": V, "offsets": OFF, "start": a.start, "duration": a.duration, "rate": a.rate, "model": a.model,
               "imgsz": a.imgsz, "conf": a.conf, "reid": os.path.basename(a.reid),
               "calib": {c: f"calib/{c}/calib.json" for c in V}}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
