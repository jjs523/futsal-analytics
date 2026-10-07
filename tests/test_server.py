"""End-to-end through the HTTP API with synthetic videos (no GPU / YOLO needed)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from futsal import Court
from futsal.pipeline import detect
from futsal.sim import layout, scenario, video
from server.app import worker
from server.app.main import create_app

COURT = Court(40, 20)
SCALE = 0.5            # videos are rendered at 960x540


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(worker.DETECTOR_FACTORY, "factory", lambda: detect.color_blob_detector(video.TEAM_BGR))
    monkeypatch.setattr(worker, "STRIDE", 1)
    app = create_app(str(tmp_path / "data"), start_worker=False)
    with TestClient(app) as c:
        c.store = app.state.store
        yield c


def upload(client, mid, cam, seq, path, chunk=50_000):
    data = open(path, "rb").read()
    url = f"/api/matches/{mid}/cameras/{cam}/segments/{seq}"
    off = 0
    while off < len(data):
        r = client.head(url)
        assert int(r.headers["Upload-Offset"]) == off
        r = client.patch(url, content=data[off:off + chunk], headers={"Upload-Offset": str(off)})
        assert r.status_code == 204
        off = int(r.headers["Upload-Offset"])
    return client.post(url + "/complete", json={"t0": None})


def taps_for(cam):
    kp = COURT.keypoints()
    uv, ok = cam.project(np.array(list(kp.values())))
    return {n: [float(u * SCALE), float(v * SCALE)] for n, (u, v), o in zip(kp, uv, ok) if o}


def test_full_match_flow(client, tmp_path):
    truth = scenario.synthetic_match(COURT, seconds=12, fps=10, seed=7)
    cams = layout.diagonal(COURT, 5).build(layout.HFOV_AVERAGE)
    m = client.post("/api/matches", json={"court": "40x20", "title": "test"}).json()
    mid = m["id"]
    assert client.get(f"/api/matches/{mid}/tracks").status_code == 404
    for name, cam in cams.items():
        path = tmp_path / f"{name}.mp4"
        video.render(cam, COURT, truth, str(path), scale=SCALE)
        assert upload(client, mid, name, 0, path).status_code == 200
        r = client.get(f"/api/matches/{mid}/cameras/{name}/frame", params={"t": 1})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        # the scaled-down video is calibrated in its own pixel space, like a real frame grab
        r = client.post(f"/api/matches/{mid}/cameras/{name}/calibration", json={"taps": taps_for(cam)})
        assert r.status_code == 200 and r.json()["rms_px"] < 1
    assert client.post(f"/api/matches/{mid}/finish", json={"offsets": {"cam1": 0, "cam2": 0}}).status_code == 200
    assert worker.drain(client.store) == 3
    m = client.get(f"/api/matches/{mid}").json()
    assert m["status"] == "done", m
    tr = client.get(f"/api/matches/{mid}/tracks").json()
    assert tr["court"] == {"length": 40.0, "width": 20.0}
    assert 8 <= len(tr["players"]) <= 16
    assert {p["team"] for p in tr["players"]} == {"A", "B"}


def test_upload_offset_mismatch_is_rejected(client):
    mid = client.post("/api/matches", json={}).json()["id"]
    url = f"/api/matches/{mid}/cameras/cam1/segments/0"
    assert client.patch(url, content=b"abc", headers={"Upload-Offset": "0"}).status_code == 204
    assert client.patch(url, content=b"def", headers={"Upload-Offset": "0"}).status_code == 409
    assert client.head(url).headers["Upload-Offset"] == "3"


def test_bad_inputs(client):
    assert client.post("/api/matches", json={"court": "big"}).status_code == 422
    mid = client.post("/api/matches", json={}).json()["id"]
    assert client.head(f"/api/matches/{mid}/cameras/phone/segments/0").status_code == 400
    r = client.post(f"/api/matches/{mid}/cameras/cam1/calibration", json={"taps": {"centre": [1, 2]}})
    assert r.status_code == 422
    assert client.post(f"/api/matches/{mid}/finish", json={}).status_code == 409


def test_failed_detection_marks_match_failed(client, tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("no GPU")
    monkeypatch.setitem(worker.DETECTOR_FACTORY, "factory", boom)
    mid = client.post("/api/matches", json={}).json()["id"]
    url = f"/api/matches/{mid}/cameras/cam1/segments/0"
    client.patch(url, content=b"not a video", headers={"Upload-Offset": "0"})
    client.post(url + "/complete", json={})
    worker.drain(client.store)
    m = client.get(f"/api/matches/{mid}").json()
    assert m["status"] == "failed" and "no GPU" in m["error"]


def test_demo_and_court(client):
    m = client.post("/api/demo", params={"seconds": 20}).json()
    tr = client.get(f"/api/matches/{m['id']}/tracks").json()
    assert len(tr["players"]) == 10 and tr["n_frames"] == 200
    assert all("distance_m" in p["stats"] for p in tr["players"])
    c = client.get("/api/court", params={"size": "38x18"}).json()
    assert c["length"] == 38 and len(c["keypoints"]) == 29
