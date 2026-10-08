"""Detector comparison on real frames: how many people inside the court each detector finds (recall proxy), per camera.

python experiments/det_bench.py
"""
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")
from futsal.court import Court
from futsal.homography import timeline_from_json
from futsal.syncview import frames_at

V = {"cam1": r"C:\Users\user\dev\futsal-videos\20261008_170919.mp4", "cam2": r"C:\Users\user\dev\futsal-videos\20261008_171122.mp4"}
OFF = {"cam1": 0.0, "cam2": 124.27}
court = Court(40, 20)
cals = {c: timeline_from_json(json.load(open(f"calib/{c}/calib.json", encoding="utf-8"))) for c in V}
T = [600 + 0.5 * k for k in range(60)]          # cam1 clock


def frames(cam):
    return [f for _, f in frames_at(V[cam], [t - OFF[cam] for t in T])]


def run(model, imgsz, conf, band=None):
    from ultralytics import YOLO
    net = YOLO(model)

    def det(img):
        r = net.predict(img, classes=[0], conf=conf, imgsz=imgsz, verbose=False, half=True)[0]
        b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        if band:                                 # extra pass on the top band, upscaled 2x
            y0, y1 = band
            crop = cv2.resize(img[y0:y1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            r2 = net.predict(crop, classes=[0], conf=conf, imgsz=imgsz, verbose=False, half=True)[0]
            b2 = r2.boxes.xyxy.cpu().numpy() / 2 + [0, y0, 0, y0]
            b, c = np.vstack([b, b2]), np.concatenate([c, r2.boxes.conf.cpu().numpy()])
            if len(b):
                keep = np.array(cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b],
                                                 c.tolist(), conf, 0.5)).reshape(-1)
                b, c = b[keep], c[keep]
        return b, c
    return det


VARIANTS = [
    ("yolo11s.pt", 1280, 0.3, None),
    ("yolo11s.pt", 1280, 0.1, None),
    ("yolo11x.pt", 1280, 0.1, None),
    ("yolo11x.pt", 1920, 0.1, None),
    ("yolo26x.pt", 1280, 0.1, None),
    ("yolo11x.pt", 1280, 0.1, (0, 420)),
]

if __name__ == "__main__":
    imgs = {c: frames(c) for c in V}
    print("frames loaded", {c: len(v) for c, v in imgs.items()}, flush=True)
    for model, imgsz, conf, band in VARIANTS:
        det = run(model, imgsz, conf, band)
        det(imgs["cam1"][0])                      # warm-up
        row = []
        for cam in V:
            ins, far, hi, tt = [], [], [], 0.0
            for img, t in zip(imgs[cam], T):
                t0 = time.time()
                b, c = det(img)
                tt += time.time() - t0
                if not len(b):
                    ins.append(0); far.append(0); hi.append(0); continue
                foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
                xy = cals[cam].to_pitch(foot)
                m = court.contains(xy, 0.5)
                ins.append(int(m.sum()))
                hi.append(int((m & (c >= 0.5)).sum()))
                farside = (xy[:, 0] > 20) if cam == "cam1" else (xy[:, 0] < 20)
                far.append(int((m & farside).sum()))
            row.append(f"{cam}: in-court {np.mean(ins):4.1f} (conf>=.5 {np.mean(hi):4.1f}, far half {np.mean(far):4.1f}) {1000 * tt / len(T):4.0f}ms")
        print(f"{model:11s} {imgsz} conf{conf} band{band}: " + " | ".join(row), flush=True)
