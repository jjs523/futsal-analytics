"""촬영 영상 점검: python -m futsal.check cam1.mp4 cam2.mp4 --out check_report

영상마다 해상도·프레임 수·길이·비트레이트, 프레임 간격이 고른지(끊김), 밝기·선명도를 보고
9칸 미리보기 이미지를 만듭니다. 두 영상이면 소리로 시간 차이도 계산합니다.
결과 폴더의 report.txt 내용과 미리보기 이미지(*_sheet.jpg)를 공유하면 원본 없이도 상태를 확인할 수 있습니다.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from . import sync


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"file": os.path.basename(path), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": round(fps, 3), "frames": n,
            "duration_s": round(n / fps, 1) if fps else None, "size_gb": round(os.path.getsize(path) / 1e9, 2)}
    if info["duration_s"]:
        info["bitrate_mbps"] = round(os.path.getsize(path) * 8 / info["duration_s"] / 1e6, 1)
    st = sync.filename_time(path)
    info["start_from_name"] = st.isoformat(sep=" ") if st else None
    info["creation_time"] = sync.creation_time(path)
    cap.release()
    return info


def frame_timing(path: str, at_s: float, seconds: float = 20.0) -> dict:
    """`at_s`부터 `seconds`초 동안 프레임 시각(PTS)을 읽어 간격이 고른지 봅니다."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ts = []
    while len(ts) < seconds * fps * 1.5:
        if not cap.grab():
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts and ms / 1000 - ts[0] > seconds:
            break
        ts.append(ms / 1000)
    cap.release()
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) < 10:
        return {"at_s": at_s, "error": "프레임 시각을 읽지 못함"}
    nominal = 1 / fps
    return {"at_s": round(at_s, 1), "measured_fps": round(1 / float(np.median(dt)), 2),
            "interval_ms_median": round(float(np.median(dt)) * 1000, 2), "interval_ms_max": round(float(dt.max()) * 1000, 1),
            "irregular_pct": round(float(np.mean(np.abs(dt - nominal) > 0.25 * nominal)) * 100, 1),
            "dropped": int(np.sum(np.round(dt / nominal) - 1).clip(0))}


def contact_sheet(path: str, out_jpg: str, n: int = 9, width: int = 640) -> list[dict]:
    """영상 전체에서 고르게 n장을 뽑아 3열 미리보기 + 밝기·선명도. 선명도(라플라시안 분산)가 낮으면 흐림."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    tiles, stats = [], []
    for k in range(n):
        t = dur * (k + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats.append({"t": f"{int(t // 60)}:{int(t % 60):02d}", "brightness": round(float(gray.mean()), 1),
                      "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)})
        small = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
        cv2.putText(small, stats[-1]["t"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(small, stats[-1]["t"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(small)
    cap.release()
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite(out_jpg, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return stats


def check(paths: list[str], out_dir: str, sync_window: float = 120.0, max_lag: float = 30.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    report = {"videos": []}
    for p in paths:
        info = probe(p)
        dur = info["duration_s"] or 0
        info["timing"] = [frame_timing(p, t) for t in (0.0, max(dur / 2 - 10, 0), max(dur - 25, 0))]
        stem = os.path.splitext(os.path.basename(p))[0]
        info["samples"] = contact_sheet(p, os.path.join(out_dir, f"{stem}_sheet.jpg"))
        report["videos"].append(info)
        print(f"  {info['file']} 확인 끝")
    if len(paths) == 2:
        hint = sync.start_hint(paths[0], paths[1]) or 0.0
        try:
            sr = 16000
            r = sync.align(sync.read_audio(paths[0], sr), sync.read_audio(paths[1], sr), sr, sync_window, max_lag, hint)
            r["hint"] = hint
            report["sync"] = r
        except Exception as e:          # ffmpeg 없음, 소리 없음 등
            report["sync"] = {"error": str(e), "hint": hint}
    report["warnings"] = warnings(report)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    text = render(report)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return report


def warnings(report: dict) -> list[str]:
    w = []
    vids = report["videos"]
    for v in vids:
        if v["height"] < 1080 and v["width"] < 1920:
            w.append(f"{v['file']}: 해상도가 1080p보다 낮음 ({v['width']}x{v['height']})")
        if v["width"] < v["height"]:
            w.append(f"{v['file']}: 세로 영상 (가로로 찍어야 함)")
        for t in v["timing"]:
            if t.get("irregular_pct", 0) > 5:
                w.append(f"{v['file']}: {t['at_s']}초 부근 프레임 간격이 불규칙함 ({t['irregular_pct']}%) → 가변 프레임, 시각 기준 처리 필요")
            if t.get("dropped", 0) > 0:
                w.append(f"{v['file']}: {t['at_s']}초 부근 빠진 프레임 {t['dropped']}개 (최대 간격 {t['interval_ms_max']}ms)")
        dark = [s for s in v["samples"] if s["brightness"] < 60]
        if dark:
            w.append(f"{v['file']}: 어두운 구간 {len(dark)}곳 (밝기 60 미만)")
    if len(vids) == 2 and vids[0]["fps"] and vids[1]["fps"] and abs(vids[0]["fps"] - vids[1]["fps"]) > 1:
        w.append(f"두 영상 프레임 수가 다름 ({vids[0]['fps']} / {vids[1]['fps']}fps): 분석은 시각 기준이라 가능, "
                 "검출은 초당 같은 횟수로(rate_hz) 맞춤")
    s = report.get("sync")
    if s and "error" not in s:
        confs = [c for c in (s.get("conf_start"), s.get("conf_end")) if c is not None]
        if confs and min(confs) < 6:
            w.append("소리로 맞춘 시간 차의 신뢰도가 낮음 (6 미만): 손뼉 장면을 보고 직접 확인 필요")
        if s.get("drift") and abs(s["drift"]) > 300e-6:
            w.append(f"시계 속도 차이가 큼 ({s['drift'] * 1e6:+.0f} ppm): 끝 구간 손뼉을 확인")
    elif s:
        w.append(f"시간 맞추기 실패: {s['error']}")
    return w


def render(report: dict) -> str:
    lines = ["# 촬영 영상 점검 결과", ""]
    for v in report["videos"]:
        lines += [f"## {v['file']}",
                  f"- {v['width']}x{v['height']}, {v['fps']}fps, {v['frames']}프레임, "
                  f"{(v['duration_s'] or 0) / 60:.1f}분, {v['size_gb']}GB, {v.get('bitrate_mbps')}Mbps",
                  f"- 녹화 시작(파일 이름): {v['start_from_name']}, 메타데이터: {v['creation_time']}"]
        for t in v["timing"]:
            if "error" in t:
                lines.append(f"- {t['at_s']}초 부근: {t['error']}")
            else:
                lines.append(f"- {t['at_s']}초 부근 프레임 간격: 실제 {t['measured_fps']}fps, 최대 {t['interval_ms_max']}ms, "
                             f"불규칙 {t['irregular_pct']}%, 빠짐 {t['dropped']}")
        lines.append("- 밝기/선명도: " + ", ".join(f"{s['t']} {s['brightness']:.0f}/{s['sharpness']:.0f}" for s in v["samples"]))
        lines.append("")
    s = report.get("sync")
    if s:
        lines.append("## 시간 맞추기 (소리)")
        if "error" in s:
            lines.append(f"- 실패: {s['error']} (파일 이름 기준 대략 차이 {s['hint']:+.0f}s)")
        else:
            lines.append(f"- 파일 이름 기준 대략 차이: {s['hint']:+.0f}s")
            lines.append(f"- 시작 구간: {s['offset_start']:+.3f}s (신뢰도 {s['conf_start']:.1f})")
            if s.get("offset_end") is not None:
                lines.append(f"- 끝 구간: {s['offset_end']:+.3f}s (신뢰도 {s['conf_end']:.1f}), 시계 속도 차이 {s['drift'] * 1e6:+.0f} ppm")
            lines.append(f"- 분석에 넣을 값: {{\"offset\": {s['offset']:.3f}, \"drift\": {s['drift']:.3e}}}")
        lines.append("")
    lines.append("## 주의")
    lines += [f"- {x}" for x in report["warnings"]] or ["- 없음"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.check", description="촬영 영상 점검")
    ap.add_argument("videos", nargs="+", help="영상 1~2개 (두 개면 첫 번째가 기준)")
    ap.add_argument("--out", default="check_report")
    ap.add_argument("--window", type=float, default=120.0, help="시간 맞추기에 쓸 시작·끝 구간 길이(초)")
    ap.add_argument("--max-lag", type=float, default=30.0, help="파일 이름 기준 차이에서 더 찾아볼 범위(초)")
    a = ap.parse_args(argv)
    print("점검 중... (긴 영상은 소리 읽기에 몇 분 걸립니다)")
    report = check(a.videos, a.out, a.window, a.max_lag)
    print(render(report))
    print(f"→ {a.out}/report.txt 와 *_sheet.jpg 를 공유해 주세요")


if __name__ == "__main__":
    main()
