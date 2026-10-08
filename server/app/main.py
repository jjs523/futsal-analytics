"""개발용 PC 도구의 HTTP API (운영 서버 아님: 운영 분석은 폰에서 함, docs/architecture.md).
팀원이 PC에서 실제 영상을 올려 정답 결과를 만들고 앱 결과와 비교하는 데 씁니다.

HTTP API. Both capture tracks use the same endpoints:

Track A (record, then upload):  one segment per camera (seq 0), uploaded after the match.
Track B (in-app segmented recording): seq 0, 1, 2 ... uploaded while the match is still running;
                                      each finished segment is analysed immediately.

Flow:  POST /api/matches -> upload segments (HEAD/PATCH/complete) -> POST calibration per camera
       -> POST /api/matches/{id}/finish -> poll GET /api/matches/{id} -> GET /api/matches/{id}/tracks
Run:   uvicorn server.app.main:app --reload     (viewer at http://localhost:8000/viewer/)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from futsal import Court
from futsal.homography import calibrate

from . import worker as worker_mod
from .store import Store

ROOT = Path(__file__).resolve().parents[2]
CAMERA_RE = re.compile(r"^cam[1-4]$")


class NewMatch(BaseModel):
    court: str = Field("40x20", pattern=r"^\d+(\.\d+)?x\d+(\.\d+)?$", description="length x width in metres")
    title: str | None = None


class Taps(BaseModel):
    taps: dict[str, tuple[float, float]] = Field(..., description="keypoint name -> [u, v] pixel in the analysed video's frame")


class Finish(BaseModel):
    offsets: dict[str, float | dict[str, float]] | None = Field(
        None, description='seconds to add to each camera\'s clock, or {"offset": s, "drift": ratio}; omit to sync by audio')



def create_app(data_dir: str | None = None, start_worker: bool = True) -> FastAPI:
    store = Store(data_dir or os.environ.get("FUTSAL_DATA", ROOT / "data"))
    app = FastAPI(title="Futsal Analytics API", version="0.1.0")
    app.state.store = store
    app.state.worker = None
    if start_worker:
        app.state.worker = worker_mod.Worker(store)
        app.state.worker.start()

    def kick():
        if app.state.worker:
            app.state.worker.wake.set()

    def get_match(mid: str) -> dict:
        m = store.match(mid)
        if m is None:
            raise HTTPException(404, "match not found")
        return m

    def check_cam(cam: str) -> None:
        if not CAMERA_RE.match(cam):
            raise HTTPException(400, "camera must be cam1..cam4")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/court")
    def court(size: str = "40x20"):
        return Court.parse(size).to_json()

    @app.post("/api/matches", status_code=201)
    def new_match(body: NewMatch):
        return store.create_match(body.court, body.title)

    @app.get("/api/matches")
    def list_matches():
        return store.matches()

    @app.get("/api/matches/{mid}")
    def match(mid: str):
        m = get_match(mid)
        m["jobs"] = store.jobs(mid)
        return m

    # ---- resumable segment upload ----
    @app.head("/api/matches/{mid}/cameras/{cam}/segments/{seq}")
    def segment_offset(mid: str, cam: str, seq: int):
        get_match(mid); check_cam(cam)
        seg = store.segment(mid, cam, seq)
        return Response(headers={"Upload-Offset": str(seg["received"] if seg else 0),
                                 "Upload-Complete": "1" if seg and seg["complete"] else "0"})

    @app.patch("/api/matches/{mid}/cameras/{cam}/segments/{seq}")
    async def segment_append(mid: str, cam: str, seq: int, request: Request, upload_offset: int = Header(0)):
        m = get_match(mid); check_cam(cam)
        data = await request.body()
        try:
            received = store.append(mid, cam, seq, upload_offset, data)
        except ValueError as e:
            raise HTTPException(409, str(e))
        if m["status"] == "created":
            store.set_status(mid, "uploading")
        return Response(status_code=204, headers={"Upload-Offset": str(received)})

    @app.post("/api/matches/{mid}/cameras/{cam}/segments/{seq}/complete")
    def segment_complete(mid: str, cam: str, seq: int, t0: float | None = Body(None, embed=True)):
        get_match(mid); check_cam(cam)
        seg = store.segment(mid, cam, seq)
        if not seg or seg["received"] == 0:
            raise HTTPException(409, "nothing uploaded for this segment")
        store.complete_segment(mid, cam, seq, t0)
        job = store.enqueue(mid, "segment", {"camera": cam, "seq": seq})
        kick()
        return {"job": job}

    # ---- calibration ----
    @app.get("/api/matches/{mid}/cameras/{cam}/frame")
    def frame(mid: str, cam: str, t: float = 1.0, seq: int = 0):
        """A JPEG still from an uploaded segment, for the calibration (tap the keypoints) screen."""
        get_match(mid); check_cam(cam)
        path = store.segment_path(mid, cam, seq)
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404, "no frame at that time")
        return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), media_type="image/jpeg")

    @app.post("/api/matches/{mid}/cameras/{cam}/calibration")
    def set_calibration(mid: str, cam: str, body: Taps):
        m = get_match(mid); check_cam(cam)
        try:
            cal = calibrate(body.taps, Court.parse(m["court"]), ransac_px=8 if len(body.taps) >= 6 else None)
        except ValueError as e:
            raise HTTPException(422, str(e))
        data = cal.to_json() | {"taps": body.taps}
        store.save_calibration(mid, cam, data)
        warn = None if cal.rms_px < 5 else "reprojection error is high: re-check the tapped points"
        return {"rms_px": round(cal.rms_px, 2), "used": cal.used, "rejected": sorted(set(body.taps) - set(cal.used)), "warning": warn}

    # ---- results ----
    @app.post("/api/matches/{mid}/finish")
    def finish(mid: str, body: Finish | None = None):
        m = get_match(mid)
        if not m["segments"]:
            raise HTTPException(409, "no video uploaded")
        if body and body.offsets is not None:
            store.set_offsets(mid, body.offsets)
        store.set_status(mid, "processing")
        job = store.enqueue(mid, "finalize", {})
        kick()
        return {"job": job}

    @app.get("/api/matches/{mid}/tracks")
    def tracks(mid: str):
        get_match(mid)
        p = store.tracks_path(mid)
        if not p.exists():
            raise HTTPException(404, "not analysed yet")
        return FileResponse(p, media_type="application/json")

    @app.post("/api/demo", status_code=201)
    def demo(court: str = "40x20", seconds: float = 300, seed: int = 0):
        """A match filled with simulated tracks, so the viewer can be developed before real footage exists."""
        from futsal.sim import layout, observe, scenario
        c = Court.parse(court)
        m = store.create_match(court, f"데모 경기 (시뮬레이션 {seconds:g}초)")
        truth = scenario.synthetic_match(c, seconds, seed=seed)
        est, _ = observe.run(truth, layout.diagonal(c, 5), layout.HFOV_AVERAGE, seed=seed)
        est.compute_stats()
        est.dump(str(store.tracks_path(m["id"])))
        store.set_status(m["id"], "done")
        return store.match(m["id"])

    viewer = ROOT / "web" / "viewer"
    if viewer.exists():
        app.mount("/viewer", StaticFiles(directory=viewer, html=True), name="viewer")

        @app.get("/")
        def root():
            return RedirectResponse("/viewer/")

    return app


app = create_app() if os.environ.get("FUTSAL_NO_AUTOAPP") != "1" else None
