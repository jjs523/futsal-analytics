import cv2
import numpy as np

from futsal.ball import dataset, track

W, H = 1920, 1080


def _frame(ball_xy=None, r=5):
    img = np.full((H, W, 3), (42, 122, 59), np.uint8)
    if ball_xy is not None:
        cv2.circle(img, (int(ball_xy[0]), int(ball_xy[1])), r, (240, 240, 240), -1)
    return img


def _white_blob_detector(tile):
    """가짜 조각 검출기: 흰 원을 찾음 (실제로는 YOLO)."""
    m = cv2.inRange(tile, (200, 200, 200), (255, 255, 255))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [(x, y, x + w, y + h, 0.9) for x, y, w, h, a in stats[1:] if a >= 5]


def test_tiles_cover_frame_and_keep_ball_labels(tmp_path):
    origins = dataset.tile_origins(W, H)
    assert (0, 0) in origins and (W - 640, H - 640) in origins
    img_dir, lab_dir = tmp_path / "img", tmp_path / "lab"
    img_dir.mkdir(); lab_dir.mkdir()
    for k, xy in enumerate([(300, 200), (1500, 900), None]):
        cv2.imwrite(str(img_dir / f"cam1_{k:06d}.jpg"), _frame(xy))
        boxes = [] if xy is None else [dataset.Box(0, xy[0] - 6, xy[1] - 6, xy[0] + 6, xy[1] + 6)]
        dataset.write_yolo_labels(lab_dir / f"cam1_{k:06d}.txt", boxes, W, H)
    stats = dataset.build_tiles(str(img_dir), str(lab_dir), str(tmp_path / "ds"), "train")
    assert stats["positive_tiles"] >= 2
    for lab in (tmp_path / "ds" / "labels" / "train").glob("*.txt"):
        for b in dataset.read_yolo_labels(lab, 640, 640):
            img = cv2.imread(str(tmp_path / "ds" / "images" / "train" / (lab.stem + ".jpg")))
            cx, cy = int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2)
            assert img[cy, cx].min() > 180                    # the label sits on the white ball


def test_split_by_time_keeps_tail_of_each_video_for_validation():
    from pathlib import Path
    imgs = [Path(f"cam1_{i:06d}.jpg") for i in range(10)] + [Path(f"cam2_{i:06d}.jpg") for i in range(10)]
    tr, va = dataset.split_by_time(imgs, 0.2)
    assert {p.name for p in va} == {"cam1_000008.jpg", "cam1_000009.jpg", "cam2_000008.jpg", "cam2_000009.jpg"}


def test_merge_keeps_only_ball_class(tmp_path):
    src = tmp_path / "rf" / "train"
    (src / "images").mkdir(parents=True); (src / "labels").mkdir()
    cv2.imwrite(str(src / "images" / "a.jpg"), _frame((100, 100)))
    (src / "labels" / "a.txt").write_text("1 0.5 0.5 0.1 0.2\n2 0.05 0.09 0.01 0.01\n")   # player, ball
    dataset.merge_yolo_dataset(str(tmp_path / "rf"), str(tmp_path / "out"), ball_class_ids=[2])
    assert (tmp_path / "out" / "labels" / "train" / "ext_a.txt").read_text().strip() == "0 0.05 0.09 0.01 0.01"


def test_tiled_detection_finds_a_small_ball_anywhere():
    for xy in [(50, 60), (959, 540), (1900, 1070), (640, 300)]:
        found = track.detect_tiled(_frame(xy), _white_blob_detector)
        assert len(found) == 1                                  # overlap duplicates merged by NMS
        x1, y1, x2, y2, _ = found[0]
        assert abs((x1 + x2) / 2 - xy[0]) < 2 and abs((y1 + y2) / 2 - xy[1]) < 2


def test_link_follows_the_ball_ignores_noise_and_fills_short_gaps():
    rng = np.random.default_rng(0)
    n = 60
    true = np.stack([200 + 25 * np.arange(n), 500 + 8 * np.sin(np.arange(n) / 5)], 1)
    cands = []
    for i in range(n):
        c = [] if 20 <= i < 25 else [(true[i, 0] + rng.normal(0, 1), true[i, 1] + rng.normal(0, 1), 0.8)]
        c += [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.2, 0.4)) for _ in range(2)]   # false alarms (heads, lines)
        cands.append(c)
    out = track.link(cands, 30.0, max_speed_px=60)
    for i in range(n):
        assert out[i] is not None
        assert np.hypot(out[i][0] - true[i, 0], out[i][1] - true[i, 1]) < 6
    assert all(out[i][3] == "interpolated" for i in range(20, 25))


def test_sample_frames(tmp_path):
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(30):
        img = np.zeros((180, 320, 3), np.uint8); cv2.circle(img, (10 * i, 90), 5, (255, 255, 255), -1); vw.write(img)
    vw.release()
    saved = dataset.sample_frames(path, str(tmp_path / "frames"), every_s=1.0)
    assert len(saved) >= 3
