"""추적 결과 영상: 실제 영상 두 개 → 선수 검출 → 두 카메라 병합·추적 → 확인용 영상

  python -m futsal.trackview --video cam1.mp4 cam2.mp4 --calib calib/cam1/calib.json calib/cam2/calib.json \
      --offset 0 124.27 --start 600 --duration 180 --out trackview

결과 (--out 폴더):
  trackview.mp4  위: 두 카메라 화면에 선수 ID, 아래: 2D 지도 + ID 사건 목록
  events.csv     ID가 경기장 한가운데서 새로 생기거나 끊긴 순간 (= ID가 바뀌었을 가능성이 큰 곳) 목록
  tracks.json    선수별 위치 (web/viewer 로 재생 가능)
  det_cam1.json  검출 결과 캐시 (같은 설정으로 다시 돌리면 검출을 건너뜀; ReID 특징은 det_cam1.json.reid.npy)

--start 는 cam1 영상 기준 초, --offset 은 cam1 시각 = 그 카메라 시각 + offset (python -m futsal.sync 결과).
검출은 YOLO(사전학습 person)를 씁니다: pip install -e ".[vision]". GPU가 없으면 3분 구간에 수십 분 걸릴 수 있습니다.

추적 방식 (--ids):
  v2          (기본) 상자 단위 추적: 카메라별 조각 → 두 카메라 짝짓기 → 5대5 정원 배정. 사람 재식별(ReID)
              특징이 필요합니다 (설치: futsal/pipeline/reid.py 설명). 검출 신뢰도 기본 0.1.
  appearance  예전 방식 (발 위치 + 색 특징). ReID를 쓸 수 없으면 v2 대신 자동으로 이 방식을 씁니다.
  motion      움직임만 보는 가장 오래된 방식.
--no-reid 는 ReID 계산을 건너뛰고 예전 방식(appearance)으로 추적합니다.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os

import cv2
import numpy as np

from .court import Court
from .homography import Calibration, CalibrationTimeline, calibration_at, timeline_from_json
from .pipeline import detect
from .pipeline.run import IDS, build_tracks
from .tracks import TrackSet

PANEL_W, PANEL_H = 960, 540


def color_of(pid: int) -> tuple[int, int, int]:
    h = int((pid * 0.618034 % 1.0) * 180)
    c = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def load_calibration(path: str, video: str | None = None, auto_moves: bool = False, cache_dir: str | None = None
                     ) -> Calibration | CalibrationTimeline:
    """calib.json (calibtool fit) or a timeline list. With `auto_moves`, camera knocks found by futsal.camshift
    over the whole video turn a single calibration into a timeline (cached as moves_<name>.json)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cal = timeline_from_json(d)
    if auto_moves and isinstance(cal, Calibration) and video:
        from . import camshift
        cache = os.path.join(cache_dir or ".", f"moves_{os.path.splitext(os.path.basename(video))[0]}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                moves = json.load(f)
        else:
            print(f"  {os.path.basename(video)}: 카메라가 움직였는지 훑는 중 (영상 전체, 몇 분)...")
            moves = camshift.camera_moves(video)["moves"]
            with open(cache, "w") as f:
                json.dump(moves, f)
        if moves:
            for line in camshift.describe({"moves": moves}):
                print("   ", line)
            cal = camshift.timeline_from_moves(cal, moves, float(d.get("at", 0.0)))
    return cal


class FrameReader:
    """Frames of one video by time, read forward (seeks only when going back or jumping far ahead)."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t, self.frame = -1e9, None

    def _read(self) -> bool:
        ok, f = self.cap.read()
        if ok:
            self.frame, self.t = f, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        return ok

    def at(self, t: float) -> np.ndarray | None:
        if t < self.t - 0.5 or t > self.t + 5:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(t - 0.5, 0) * 1000)
            self.t = -1e9
        half = 0.5 / self.fps
        while self.t < t - half:
            if not self._read():
                break
        return self.frame

    def close(self):
        self.cap.release()


def find_events(ts: TrackSet, court: Court, edge_m: float = 2.5, settle_s: float = 2.0) -> list[dict]:
    """IDs that start or end in the middle of the pitch (not at the window edges, not at the touchlines where
    players really come and go): the likely places where an identity was lost or swapped."""
    ev = []
    n, fps = ts.n_frames, ts.fps
    for p in ts.players:
        seen = np.where(np.all(np.isfinite(p.xy), axis=1))[0]
        if len(seen) == 0:
            continue
        for kind, k in (("new", int(seen[0])), ("lost", int(seen[-1]))):
            if (kind == "new" and k < settle_s * fps) or (kind == "lost" and k > n - 1 - settle_s * fps):
                continue
            x, y = p.xy[k]
            if min(x, y, court.length - x, court.width - y) < edge_m:
                continue
            ev.append({"t": k / fps, "frame": k, "id": p.id, "kind": kind, "x": round(float(x), 1), "y": round(float(y), 1)})
    return sorted(ev, key=lambda e: e["t"])


def _court_map(court: Court, w: int, h: int, margin: int = 30):
    s = min((w - 2 * margin) / court.length, (h - 2 * margin) / court.width)
    ox, oy = (w - court.length * s) / 2, (h - court.width * s) / 2

    def P(xy):
        xy = np.atleast_2d(xy)
        return np.stack([ox + xy[:, 0] * s, oy + (court.width - xy[:, 1]) * s], 1)
    base = np.full((h, w, 3), (38, 92, 52), np.uint8)
    for m in court.markings():
        cv2.polylines(base, [np.round(P(m)).astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
    for gx in (0.0, court.length):
        a, b = P([[gx, court.width / 2 - 1.5], [gx, court.width / 2 + 1.5]])
        cv2.line(base, tuple(np.int32(a)), tuple(np.int32(b)), (255, 255, 255), 5)
    return base, P


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _rect(d) -> tuple[float, float, float, float]:
    """A detection's box in calibrated pixels: the stored box, or one rebuilt from the foot point and height for
    caches written before boxes were kept."""
    if d.box is not None:
        return d.box
    return d.u - 0.22 * d.h_px, d.v - d.h_px, d.u + 0.22 * d.h_px, d.v


def frame_size(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return wh


def render(out_path: str, videos: dict[str, str], cals: dict, offsets: dict, ts: TrackSet, dets: dict[str, list],
           court: Court, start: float, events: list[dict], fps_out: float | None = None, tail_s: float = 2.0) -> int:
    """One output frame per tracking step: camera panels on top, 2D map + event list below. Returns frames written."""
    fps_out = fps_out or ts.fps
    readers = {c: FrameReader(v) for c, v in videos.items()}
    det_times = {c: sorted({d.t for d in ds}) for c, ds in dets.items()}
    det_by_t = {c: {} for c in dets}
    for c, ds in dets.items():
        for d in ds:
            det_by_t[c].setdefault(d.t, []).append(d)
    map_w, map_h = 1180, PANEL_H
    base_map, P = _court_map(court, map_w, map_h)
    W, H = PANEL_W * 2, PANEL_H * 2
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, H))
    cams = list(videos)[:2]
    n_out = int(ts.n_frames / ts.fps * fps_out)
    for j in range(n_out):
        t_rel = j / fps_out
        k = min(int(round(t_rel * ts.fps)), ts.n_frames - 1)
        canvas = np.zeros((H, W, 3), np.uint8)
        pos = {p.id: p.xy[k] for p in ts.players if np.all(np.isfinite(p.xy[k]))}
        for ci, cam in enumerate(cams):
            off, drift = offsets.get(cam, (0.0, 0.0))
            t_cam = (start + t_rel - off) / (1 + drift)
            frame = readers[cam].at(t_cam)
            panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            if frame is not None:
                fh, fw = frame.shape[:2]
                sc = min(PANEL_W / fw, PANEL_H / fh)
                small = cv2.resize(frame, (int(fw * sc), int(fh * sc)))
                panel[:small.shape[0], :small.shape[1]] = small
                cal = calibration_at(cals[cam], t_cam)
                i = bisect.bisect_left(det_times[cam], t_cam)
                near = [det_times[cam][q] for q in (i - 1, i) if 0 <= q < len(det_times[cam])]
                boxes = det_by_t[cam].get(min(near, key=lambda q: abs(q - t_cam)), []) if near else []
                feet = np.array([(d.u, d.v) for d in boxes]).reshape(-1, 2)
                taken = set()
                if pos:
                    ids = list(pos)
                    uv = cal.to_image(np.array([pos[i_] for i_ in ids]))
                    for pid, (u, v) in zip(ids, uv):
                        if not (np.isfinite(u) and -50 < u < fw + 50 and -50 < v < fh + 50):
                            continue
                        col = color_of(pid)
                        b = None
                        if len(feet):
                            dist = np.hypot(feet[:, 0] - u, feet[:, 1] - v)
                            q = int(np.argmin(dist))
                            if dist[q] < max(25.0, 0.4 * boxes[q].h_px) and q not in taken:
                                b, _ = boxes[q], taken.add(q)
                        if b is not None:
                            x0, y0, x1, y1 = (int(c * sc) for c in _rect(b))
                            cv2.rectangle(panel, (x0, y0), (x1, y1), col, 2)
                            tx, ty = x0, y0 - 4
                        else:                       # tracked here from the other camera / smoothing, no box
                            cv2.circle(panel, (int(u * sc), int(v * sc)), 5, col, 2)
                            tx, ty = int(u * sc) + 6, int(v * sc) - 6
                        cv2.putText(panel, str(pid), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(panel, str(pid), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                for q, b in enumerate(boxes):           # detections no track claimed: thin grey
                    if q not in taken:
                        x0, y0, x1, y1 = (int(c * sc) for c in _rect(b))
                        cv2.rectangle(panel, (x0, y0), (x1, y1), (160, 160, 160), 1)
            label = f"{cam}  {_fmt(t_cam)}"
            cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            canvas[:PANEL_H, ci * PANEL_W:(ci + 1) * PANEL_W] = panel
        m = base_map.copy()
        k0 = max(0, k - int(tail_s * ts.fps))
        for p in ts.players:                      # trail: only connect consecutive seen frames, never across gaps
            seg = p.xy[k0:k + 1]
            ok = np.all(np.isfinite(seg), axis=1)
            runs = np.split(np.arange(len(seg)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1)
            for r in runs:
                if len(r) >= 2 and ok[r[0]]:
                    cv2.polylines(m, [np.round(P(seg[r])).astype(np.int32)], False, color_of(p.id), 2, cv2.LINE_AA)
        for pid, xy in pos.items():
            (x, y), = P(xy)
            cv2.circle(m, (int(x), int(y)), 8, color_of(pid), -1, cv2.LINE_AA)
            cv2.putText(m, str(pid), (int(x) + 9, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        for e in events:
            if 0 <= t_rel - e["t"] < 2.5:            # flash the event for 2.5 s
                (x, y), = P([e["x"], e["y"]])
                cv2.circle(m, (int(x), int(y)), 22, (0, 0, 255) if e["kind"] == "lost" else (0, 220, 255), 3, cv2.LINE_AA)
        canvas[PANEL_H:, :map_w] = m
        side = canvas[PANEL_H:, map_w:]
        side[:] = (28, 28, 28)
        cv2.putText(side, f"{_fmt(start + t_rel)}  ids:{len(pos)}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side, "ID events (new / lost mid-pitch)", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        past = [e for e in events if e["t"] <= t_rel][-14:]
        for r, e in enumerate(reversed(past)):
            fresh = t_rel - e["t"] < 2.5
            col = ((0, 0, 255) if e["kind"] == "lost" else (0, 220, 255)) if fresh else (200, 200, 200)
            txt = f"{_fmt(start + e['t'])}  #{e['id']} {e['kind']}  ({e['x']:.0f},{e['y']:.0f})"
            cv2.putText(side, txt, (14, 102 + 30 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2 if fresh else 1, cv2.LINE_AA)
        vw.write(canvas)
        if j and j % 300 == 0:
            print(f"    영상 {j}/{n_out}")
    vw.release()
    for r in readers.values():
        r.close()
    return n_out


def run(videos: dict[str, str], cals: dict, offsets: dict[str, tuple[float, float]], start: float, duration: float,
        out_dir: str, detector_factory, court: Court, rate_hz: float = 10.0, fps_out: float = 10.0,
        det_tag: str = "", embedder=None, ids: str = "v2") -> dict:
    """Detect (cached per camera), track with build_tracks(ids=...), write tracks.json / events.csv / trackview.mp4.
    `embedder` (pipeline.reid.reid_embedder) adds a ReID embedding to every box, which ids="v2" needs; without it
    v2 falls back to "appearance". The detection cache key is `det_tag` plus whether ReID was computed."""
    os.makedirs(out_dir, exist_ok=True)
    tag = det_tag + ("+reid" if embedder is not None else "")
    dets = {}
    for cam, path in videos.items():
        off, drift = offsets.get(cam, (0.0, 0.0))
        s_cam = (start - off) / (1 + drift)
        cache = os.path.join(out_dir, f"det_{cam}.json")
        want = {"video": os.path.basename(path), "start": round(s_cam, 3), "duration": duration, "rate": rate_hz, "tag": tag}
        if os.path.exists(cache):
            d, meta = detect.load(cache)
            if all(meta.get(k) == v for k, v in want.items()):
                dets[cam] = d
                print(f"  {cam}: 저장된 검출 결과 사용 ({len(d)}개)")
                continue
        print(f"  {cam}: 선수 검출 중 ({_fmt(s_cam)}부터 {duration:.0f}초)...")
        d, fps, _ = detect.detect_video(path, detector_factory(cam), rate_hz=rate_hz, start_s=max(s_cam, 0.0),
                                        end_s=s_cam + duration, progress=True, embedder=embedder)
        detect.save(d, cache, **want)
        dets[cam] = d
    # reference clock for tracking: 0 = window start on cam1's clock
    sync = {cam: {"offset": offsets.get(cam, (0.0, 0.0))[0] - start, "drift": offsets.get(cam, (0.0, 0.0))[1]} for cam in videos}
    info: dict = {}
    extra = {"frame_wh": {cam: frame_size(p) for cam, p in videos.items()}} if ids == "v2" else {}
    ts = build_tracks(court, dets, cals, sync, fps_out=rate_hz, ids=ids, info=info, **extra)
    if info.get("fallback"):
        print("  ReID 특징이 없어 예전 방식(appearance)으로 추적합니다")
    if info.get("ids") == "v2":
        al = info.get("alignment")
        print(f"  추적 v2: 조각 {info['tracklets']}개 → 선수 {info['identities']}명"
              + (f", 두 카메라 위치 차 {al['median_before_m']} → {al['median_after_m']} m" if al else ""))
    n = int(round(duration * rate_hz))
    for p in ts.players:                                   # clip to the window
        xy = np.full((n, 2), np.nan)
        m = min(n, len(p.xy))
        xy[:m] = p.xy[:m]
        p.xy = xy
    ts.players = [p for p in ts.players if np.any(np.isfinite(p.xy[:, 0]))]
    ts.start = start
    ts.compute_stats()
    ts.dump(os.path.join(out_dir, "tracks.json"))
    events = find_events(ts, court)
    with open(os.path.join(out_dir, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["시각(cam1)", "ID", "종류", "x(m)", "y(m)"])
        for e in events:
            w.writerow([_fmt(start + e["t"]), e["id"], "새로 생김" if e["kind"] == "new" else "끊김", e["x"], e["y"]])
    print(f"  영상 만드는 중...")
    frames = render(os.path.join(out_dir, "trackview.mp4"), videos, cals, offsets, ts, dets, court, start, events, fps_out)
    long_ids = [p for p in ts.players if np.mean(np.isfinite(p.xy[:, 0])) > 0.5]
    return {"ids": len(ts.players), "ids_over_half": len(long_ids), "events": len(events), "frames": frames,
            "detections": {c: len(d) for c, d in dets.items()}, "tracker": info.get("ids", ids)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.trackview", description="추적 결과 확인 영상 만들기")
    ap.add_argument("--video", nargs="+", required=True, help="cam1 영상 [cam2 영상]")
    ap.add_argument("--calib", nargs="+", required=True, help="영상마다 calib.json (python -m futsal.calibtool)")
    ap.add_argument("--offset", nargs="+", type=float, default=None, help="영상마다 시간 차 (cam1은 0, cam2는 futsal.sync 결과)")
    ap.add_argument("--drift", nargs="+", type=float, default=None)
    ap.add_argument("--start", type=float, required=True, help="cam1 영상 기준 시작 초")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--court", default="40x20")
    ap.add_argument("--model", default="yolo11s.pt"); ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 하한 (기본: --ids v2 는 0.1, 그 외 0.3)")
    ap.add_argument("--rate", type=float, default=10.0, help="초당 검출 횟수")
    ap.add_argument("--ids", choices=IDS, default="v2", help="추적 방식: v2 (기본, ReID 필요) / appearance (예전 방식) / motion")
    ap.add_argument("--no-reid", action="store_true", help="ReID 계산을 건너뛰고 예전 방식(appearance)으로 추적")
    ap.add_argument("--auto-moves", action="store_true", help="카메라가 움직인 순간을 자동으로 찾아 보정 (futsal.camshift)")
    ap.add_argument("--out", default="trackview")
    a = ap.parse_args(argv)
    if len(a.calib) != len(a.video):
        ap.error("--calib 는 --video 와 개수가 같아야 합니다")
    names = [f"cam{i + 1}" for i in range(len(a.video))]
    offs = a.offset or [0.0] * len(names)
    drifts = a.drift or [0.0] * len(names)
    court = Court.parse(a.court)
    os.makedirs(a.out, exist_ok=True)
    videos = dict(zip(names, a.video))
    cals = {n: load_calibration(c, v, a.auto_moves, a.out) for n, c, v in zip(names, a.calib, a.video)}
    offsets = {n: (o, d) for n, o, d in zip(names, offs, drifts)}
    ids = "appearance" if a.no_reid and a.ids == "v2" else a.ids
    embedder = None
    if ids == "v2":
        from .pipeline.reid import reid_embedder
        try:
            embedder = reid_embedder()
        except ImportError as e:
            print(f"ReID를 쓸 수 없어 예전 방식(appearance)으로 추적합니다: {e}")
            ids = "appearance"
    conf = a.conf if a.conf is not None else (0.1 if ids == "v2" else 0.3)
    res = run(videos, cals, offsets, a.start, a.duration, a.out,
              lambda cam: detect.yolo_detector(a.model, conf=conf, imgsz=a.imgsz), court, a.rate,
              det_tag=f"{a.model}@{a.imgsz}/{conf}", embedder=embedder, ids=ids)
    print(f"\n추적 방식 {res['tracker']}: ID {res['ids']}개 (구간 절반 이상 보인 ID {res['ids_over_half']}개), "
          f"경기장 한가운데서 생기거나 끊긴 ID {res['events']}건")
    print(f"→ {os.path.join(a.out, 'trackview.mp4')}, {os.path.join(a.out, 'events.csv')}, {os.path.join(a.out, 'tracks.json')}")


if __name__ == "__main__":
    main()
