"""기준점 탭으로 좌표 보정 만들기 (PC, 브라우저만 있으면 됨)

  1) python -m futsal.calibtool frame cam1.mp4 --at 600 --camera cam1 --out calib/cam1
     → calib/cam1/tap.html 을 브라우저로 열어, 오른쪽 그림에 빨갛게 표시된 점을 영상에서 클릭 (안 보이면 건너뛰기)
     → 끝나면 "저장"을 눌러 taps.json 을 calib/cam1/ 에 저장
  2) python -m futsal.calibtool fit calib/cam1/taps.json
     → calib/cam1/calib.json (보정값) + calib/cam1/check.jpg (코트 선을 영상 위에 겹쳐 그린 확인용 그림)

탭 화면의 코트 그림은 언제나 "내 폰"이 왼쪽 아래 모서리에 오도록 그려집니다 (cam1, cam2 모두).
그림에서 내 폰 옆 짧은 선이 "내 쪽 골라인(골대 있는 선)", 아래 긴 선이 "내 쪽 사이드라인"입니다.
cam2는 저장할 때 실제 코트 이름(180도 돌린 이름)으로 바뀌어, 두 카메라가 같은 코트 좌표를 씁니다.
점 이름을 180도 뒤집어 찍어도 fit이 알아채고 바로잡습니다.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np

from .court import Court
from .homography import Calibration, calibrate, framing

# 탭 화면의 코트 그림은 언제나 "내 폰"이 왼쪽 아래에 오도록 돌려서 그립니다. 아래 설명도 그 그림 기준입니다.
# (cam1은 그대로, cam2는 180도 돌린 그림 → 저장할 때 실제 코트 이름으로 바꿈)
LABELS = {
    "corner_bl": "내 폰 바로 앞 모서리", "corner_br": "내 쪽 긴 사이드라인의 반대쪽 끝 모서리",
    "corner_tr": "건너편 폰 쪽 모서리 (대각선 반대편)", "corner_tl": "내 쪽 골라인(골대 있는 짧은 선)의 반대쪽 끝 모서리",
    "halfway_bottom": "하프라인이 내 쪽 사이드라인과 만나는 점", "halfway_top": "하프라인이 건너편 사이드라인과 만나는 점",
    "centre": "센터 마크", "circle_bottom": "센터서클의 내 쪽 사이드라인 쪽 끝", "circle_top": "센터서클의 건너편 사이드라인 쪽 끝",
    "penalty_left": "내 쪽 골대 앞 페널티 마크 (골라인에서 6m)", "second_penalty_left": "내 쪽 제2 페널티 마크 (10m)",
    "area_goalline_top_left": "내 쪽 페널티 에어리어가 골라인과 만나는 점 (건너편 사이드라인 쪽)",
    "area_goalline_bottom_left": "내 쪽 페널티 에어리어가 골라인과 만나는 점 (내 쪽 사이드라인 쪽)",
    "area_line_top_left": "내 쪽 페널티 에어리어 직선 부분의 끝 (건너편 사이드라인 쪽)",
    "area_line_bottom_left": "내 쪽 페널티 에어리어 직선 부분의 끝 (내 쪽 사이드라인 쪽)",
    "post_top_left": "내 쪽 골대 기둥 바닥 (건너편 사이드라인 쪽)", "post_bottom_left": "내 쪽 골대 기둥 바닥 (내 쪽 사이드라인 쪽)",
    "penalty_right": "건너편 골대 앞 페널티 마크 (6m)", "second_penalty_right": "건너편 제2 페널티 마크 (10m)",
    "area_goalline_top_right": "건너편 페널티 에어리어가 골라인과 만나는 점 (건너편 사이드라인 쪽)",
    "area_goalline_bottom_right": "건너편 페널티 에어리어가 골라인과 만나는 점 (내 쪽 사이드라인 쪽)",
    "area_line_top_right": "건너편 페널티 에어리어 직선 부분의 끝 (건너편 사이드라인 쪽)",
    "area_line_bottom_right": "건너편 페널티 에어리어 직선 부분의 끝 (내 쪽 사이드라인 쪽)",
    "post_top_right": "건너편 골대 기둥 바닥 (건너편 사이드라인 쪽)", "post_bottom_right": "건너편 골대 기둥 바닥 (내 쪽 사이드라인 쪽)",
}

# 실제 구장에서 잘 보이는 점부터. 교체 구역 표시는 구장마다 위치가 달라(규격과 1m 넘게 다르기도 함) 쓰지 않음
ORDER = ["corner_bl", "corner_br", "corner_tr", "corner_tl", "halfway_bottom", "halfway_top", "centre", "circle_bottom",
         "circle_top", "area_goalline_top_left", "area_goalline_bottom_left", "area_line_top_left", "area_line_bottom_left",
         "penalty_left", "second_penalty_left", "post_top_left", "post_bottom_left",
         "area_goalline_top_right", "area_goalline_bottom_right", "area_line_top_right", "area_line_bottom_right",
         "penalty_right", "second_penalty_right", "post_top_right", "post_bottom_right"]

# 어느 카메라가 어느 모서리에 있는지 (위에서 본 코트 기준). 그림을 돌리는 각도가 여기서 정해짐
CORNER = {"cam1": "bl", "cam2": "tr"}


def rotated_name(court: Court, name: str) -> str | None:
    """The keypoint at the same spot after turning the court 180 degrees (corner_bl <-> corner_tr, ...)."""
    kp = court.keypoints()
    x, y = kp[name]
    for n, (u, v) in kp.items():
        if abs(u - (court.length - x)) < 1e-6 and abs(v - (court.width - y)) < 1e-6:
            return n
    return None


def grab_frame(video: str, at_s: float) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    cap.release()
    if not ok:
        raise RuntimeError(f"{at_s}초 프레임을 읽을 수 없습니다")
    return frame, t


def make_tap_page(video: str, at_s: float, out_dir: str, court: Court, camera: str = "cam1") -> str:
    os.makedirs(out_dir, exist_ok=True)
    frame, t = grab_frame(video, at_s)
    frame_path = os.path.abspath(os.path.join(out_dir, "frame.jpg"))
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    kp = court.keypoints()
    flip = CORNER.get(camera, "bl") == "tr"
    # drawn as if this phone stood at the bottom-left corner; saved under the real court name
    data = {
        "video": os.path.abspath(video), "at": round(t, 3), "camera": camera, "frame_path": frame_path,
        "width": int(frame.shape[1]), "height": int(frame.shape[0]), "court": [court.length, court.width],
        "keypoints": [{"name": rotated_name(court, n) if flip else n, "label": LABELS.get(n, n), "xy": list(kp[n])}
                      for n in ORDER if n in kp],
        "markings": [m.tolist() for m in court.markings()],
    }
    html = _PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace(
        "__IMG__", "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode())
    path = os.path.join(out_dir, "tap.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def fit(taps_path: str, out_dir: str | None = None, frame_path: str | None = None) -> dict:
    with open(taps_path, encoding="utf-8") as f:
        d = json.load(f)
    court = Court(*d["court"])
    taps = {k: tuple(v) for k, v in d["taps"].items()}
    # The same taps read as if the court were turned 180 degrees: a classic mix-up (the far goal tapped as the
    # near one). The court is symmetric, so both readings fit equally well; what differs is where the phone
    # stands. The bottom-centre of the picture is the ground right in front of the phone, which must be near
    # this camera's corner (cam1: bottom-left, cam2: top-right).
    corner = {"bl": (0.0, 0.0), "tr": (court.length, court.width)}.get(CORNER.get(d.get("camera") or "", ""))
    rot = {rotated_name(court, k): v for k, v in taps.items() if rotated_name(court, k)}
    best = None
    for cand, is_rot in ((taps, False), (rot, True)):
        if len(cand) < 4:
            continue
        try:
            c = calibrate(cand, court, ransac_px=8 if len(cand) >= 6 else None, image_size=(d["width"], d["height"]))
        except ValueError:
            continue
        f = framing(c, d["width"], d["height"], court)
        near = c.to_pitch([[d["width"] / 2, d["height"] * 0.98]])[0]
        miss = float(np.hypot(*(near - corner))) if corner is not None and np.all(np.isfinite(near)) else 0.0
        score = (miss, -f["visible"])
        if best is None or score < best[0]:
            best = (score, c, f, is_rot)
    cal, fr, flipped = (best[1], best[2], best[3]) if best else (None, None, False)
    if cal is None:
        raise ValueError("기준점이 4개 미만이거나 보정을 만들 수 없습니다")
    if flipped:
        taps = rot
    out_dir = out_dir or os.path.dirname(os.path.abspath(taps_path))
    os.makedirs(out_dir, exist_ok=True)
    res = cal.to_json() | {"taps": {k: list(v) for k, v in taps.items()}, "flipped_180": flipped, "at": d["at"], "width": d["width"], "height": d["height"],
                           "video": d.get("video"), "camera": d.get("camera"), "court": d["court"], "framing": fr}
    with open(os.path.join(out_dir, "calib.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    frame_path = frame_path or (d.get("frame_path") if d.get("frame_path") and os.path.exists(d["frame_path"])
                                else os.path.join(os.path.dirname(os.path.abspath(taps_path)), "frame.jpg"))
    img = cv2.imread(frame_path) if frame_path and os.path.exists(frame_path) else None
    if img is not None:
        cv2.imwrite(os.path.join(out_dir, "check.jpg"), draw_check(img, cal, court, taps), [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["rejected"] = sorted(set(taps) - set(cal.used))
    return res


def draw_check(img: np.ndarray, cal: Calibration, court: Court, taps: dict) -> np.ndarray:
    """Court lines projected with the fitted calibration (red) over the frame, taps in green: lines must sit on the paint."""
    out = img.copy()
    h, w = out.shape[:2]
    for m in court.markings():
        q = np.vstack([np.linspace(m[i], m[i + 1], 30) for i in range(len(m) - 1)])
        uv = cal.to_image(q)
        good = np.all(np.isfinite(uv), axis=1) & (np.abs(uv[:, 0]) < 4 * w) & (np.abs(uv[:, 1]) < 4 * h)
        for a, b, ga, gb in zip(uv[:-1], uv[1:], good[:-1], good[1:]):
            if ga and gb:
                cv2.line(out, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), (0, 0, 255), 2, cv2.LINE_AA)
    for name, (u, v) in taps.items():
        cv2.circle(out, (int(round(u)), int(round(v))), 7, (0, 255, 0), 2)
        cv2.putText(out, name, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.calibtool", description="기준점 탭으로 좌표 보정 만들기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frame", help="탭할 프레임과 탭 화면(tap.html) 만들기")
    f.add_argument("video"); f.add_argument("--at", type=float, default=60.0, help="몇 초 프레임 (카메라가 움직인 뒤가 좋음)")
    f.add_argument("--camera", default="cam1", help="cam1(왼쪽 아래 모서리) 또는 cam2(오른쪽 위 모서리)")
    f.add_argument("--court", default="40x20"); f.add_argument("--out", required=True)
    g = sub.add_parser("fit", help="taps.json → calib.json + check.jpg")
    g.add_argument("taps"); g.add_argument("--out", default=None); g.add_argument("--frame", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "frame":
        p = make_tap_page(a.video, a.at, a.out, Court.parse(a.court), a.camera)
        print(f"브라우저로 여세요: {os.path.abspath(p)}\n다 찍으면 '저장'을 눌러 taps.json 을 {os.path.abspath(a.out)} 에 두고:\n"
              f"  python -m futsal.calibtool fit {os.path.join(a.out, 'taps.json')}")
    else:
        r = fit(a.taps, a.out, a.frame)
        out = a.out or os.path.dirname(os.path.abspath(a.taps))
        if r["flipped_180"]:
            print("주의: 점 이름이 코트를 180도 돌린 것처럼 찍혀 있어서 자동으로 바로잡았습니다 (건너편 골대를 내 쪽으로 찍은 경우).")
        print(f"재투영 오차 {r['rms_px']:.1f}px (기준점 {len(r['used'])}개 사용"
              + (f", 빗나간 점 제외: {', '.join(r['rejected'])}" if r["rejected"] else "") + ")")
        if r.get("k1"):
            print(f"렌즈 왜곡 보정: k1 = {r['k1']:+.3f} (화면 가장자리 선이 휘는 것을 반영)")
        if r["rms_px"] > 5:
            print("주의: 오차가 큽니다. check.jpg에서 빨간 선이 코트 선과 어긋난 곳의 점을 다시 찍으세요.")
        fr = r["framing"]
        print(f"코트가 화면에 보이는 비율: {fr['visible'] * 100:.0f}%" + (f" → {fr['advice']}" if fr["advice"] else ""))
        print(f"→ {os.path.join(out, 'calib.json')}, 확인용 그림 {os.path.join(out, 'check.jpg')}")


_PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>기준점 탭</title>
<style>
:root{--bg:#f6f7f8;--fg:#1c1f23;--muted:#666;--card:#fff;--line:#d6d9dd;--accent:#d92d20;--ok:#16a34a}
@media (prefers-color-scheme: dark){:root{--bg:#15171a;--fg:#e8eaed;--muted:#9aa0a6;--card:#1f2226;--line:#33373c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Malgun Gothic",sans-serif}
.wrap{display:flex;gap:16px;padding:16px;align-items:flex-start}
.left{flex:1;min-width:0;position:relative}
.left img{width:100%;display:block;cursor:crosshair;user-select:none}
#marks{position:absolute;left:0;top:0;pointer-events:none}
.side{width:360px;flex:none;display:flex;flex-direction:column;gap:12px;position:sticky;top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
h1{font-size:16px;margin:0 0 6px}
.now{font-size:17px;font-weight:600;color:var(--accent)}
.muted{color:var(--muted);font-size:13px}
button{font:inherit;padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.row{display:flex;gap:8px;flex-wrap:wrap}
canvas{max-width:100%}
@media (max-width:900px){.wrap{flex-direction:column}.side{width:auto;position:static}}
</style></head><body>
<div class="wrap">
  <div class="left"><img id="img" src="__IMG__" alt="탭할 프레임"><canvas id="marks"></canvas></div>
  <div class="side">
    <div class="card">
      <h1>기준점 탭 <span id="cam" class="muted"></span></h1>
      <div class="muted">오른쪽 그림의 <b style="color:var(--accent)">빨간 점</b>이 영상 어디인지 찾아 클릭하세요. 안 보이면 "안 보임". 최소 4개, <b>6개 이상을 화면 곳곳에서</b> 찍으면 정확합니다.</div>
    </div>
    <div class="card">
      <canvas id="court" width="336" height="200"></canvas>
      <div class="now" id="now"></div>
      <div class="muted" id="progress"></div>
      <div class="row" style="margin-top:8px">
        <button id="skip">안 보임 (S)</button><button id="back">이전 (←)</button><button id="clear">이 점 지우기</button>
      </div>
    </div>
    <div class="card"><canvas id="loupe" width="336" height="200"></canvas><div class="muted">확대 화면 (마우스 위치 4배)</div></div>
    <div class="card"><div class="row"><button class="primary" id="save">저장 (taps.json)</button></div>
      <div class="muted" id="saveinfo" style="margin-top:6px"></div></div>
  </div>
</div>
<script>
const D = __DATA__;
const img = document.getElementById('img'), marks = document.getElementById('marks');
const court = document.getElementById('court'), loupe = document.getElementById('loupe');
const taps = {}; let cur = 0;
document.getElementById('cam').textContent = D.camera + ' · ' + D.at.toFixed(1) + '초';
function sizeMarks(){ marks.width = img.clientWidth; marks.height = img.clientHeight; drawMarks(); }
function scale(){ return img.clientWidth / D.width; }
function drawMarks(){
  const g = marks.getContext('2d'); g.clearRect(0,0,marks.width,marks.height); const s = scale();
  D.keypoints.forEach((k,i)=>{ const p = taps[k.name]; if(!p) return;
    g.beginPath(); g.arc(p[0]*s, p[1]*s, 6, 0, 7); g.lineWidth = 2; g.strokeStyle = i===cur ? '#d92d20' : '#16a34a'; g.stroke();
    g.font = '12px system-ui'; g.fillStyle = '#16a34a'; g.fillText(String(i+1), p[0]*s+8, p[1]*s-6); });
}
function drawCourt(){
  const g = court.getContext('2d'), W = court.width, H = court.height, L = D.court[0], Wd = D.court[1];
  const m = 22, sx = (W-2*m)/L, sy = (H-2*m)/Wd, s = Math.min(sx, sy), ox = (W - L*s)/2, oy = (H - Wd*s)/2;
  const P = (x,y)=>[ox + x*s, oy + (Wd - y)*s];
  g.clearRect(0,0,W,H); g.fillStyle = '#2f7d46'; g.fillRect(ox-6, oy-6, L*s+12, Wd*s+12);
  g.strokeStyle = '#fff'; g.lineWidth = 1.2;
  D.markings.forEach(m=>{ g.beginPath(); m.forEach((q,i)=>{ const [x,y]=P(q[0],q[1]); i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke(); });
  const phone = (x,y,t)=>{ const [px,py]=P(x,y); g.fillStyle='#111'; g.fillRect(px-7,py-5,14,10); g.fillStyle='#fff'; g.font='10px system-ui'; g.fillText(t, px-12, py+(y>0?-8:18)); };
  phone(-1.3,-1.3,'내 폰'); phone(L+1.3, Wd+1.3,'건너편 폰');
  D.keypoints.forEach((k,i)=>{ const [x,y]=P(k.xy[0],k.xy[1]); g.beginPath(); g.arc(x,y, i===cur?6:3, 0, 7);
    g.fillStyle = i===cur ? '#d92d20' : (taps[k.name] ? '#16a34a' : '#ffffffaa'); g.fill(); });
}
function update(){
  const k = D.keypoints[cur];
  document.getElementById('now').textContent = k ? (cur+1) + '. ' + k.label : '끝까지 왔습니다. 저장하세요.';
  const n = Object.keys(taps).length;
  document.getElementById('progress').textContent = '찍은 점 ' + n + '개' + (n < 4 ? ' (최소 4개)' : n < 6 ? ' (6개 이상 권장)' : '');
  drawCourt(); drawMarks();
}
function next(){ if(cur < D.keypoints.length) cur++; update(); }
img.addEventListener('click', e=>{ const k = D.keypoints[cur]; if(!k) return; const r = img.getBoundingClientRect(), s = scale();
  taps[k.name] = [ +((e.clientX - r.left)/s).toFixed(1), +((e.clientY - r.top)/s).toFixed(1) ]; next(); });
img.addEventListener('mousemove', e=>{ const r = img.getBoundingClientRect(), s = scale(); const u = (e.clientX - r.left)/s, v = (e.clientY - r.top)/s;
  const g = loupe.getContext('2d'), z = 4, w = loupe.width/z, h = loupe.height/z;
  g.imageSmoothingEnabled = false; g.clearRect(0,0,loupe.width,loupe.height); g.drawImage(img, u-w/2, v-h/2, w, h, 0, 0, loupe.width, loupe.height);
  g.strokeStyle = '#d92d20'; g.beginPath(); g.moveTo(loupe.width/2,0); g.lineTo(loupe.width/2,loupe.height); g.moveTo(0,loupe.height/2); g.lineTo(loupe.width,loupe.height/2); g.stroke(); });
document.getElementById('skip').onclick = next;
document.getElementById('back').onclick = ()=>{ if(cur>0) cur--; update(); };
document.getElementById('clear').onclick = ()=>{ const k = D.keypoints[cur]; if(k) delete taps[k.name]; update(); };
document.addEventListener('keydown', e=>{ if(e.key==='s'||e.key==='S') next(); if(e.key==='ArrowLeft'){ if(cur>0) cur--; update(); } });
document.getElementById('save').onclick = ()=>{
  const n = Object.keys(taps).length; if(n < 4){ alert('최소 4개를 찍어야 합니다 (지금 ' + n + '개).'); return; }
  const out = {video: D.video, at: D.at, camera: D.camera, frame_path: D.frame_path, width: D.width, height: D.height, court: D.court, taps};
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type:'application/json'}));
  a.download = 'taps.json'; a.click();
  document.getElementById('saveinfo').textContent = '다운로드 폴더에 taps.json 이 저장됩니다. 이 tap.html 과 같은 폴더로 옮긴 뒤 fit 명령을 실행하세요.';
};
window.addEventListener('resize', sizeMarks); img.onload = sizeMarks; if(img.complete) sizeMarks(); update();
</script></body></html>
"""

if __name__ == "__main__":
    main()
