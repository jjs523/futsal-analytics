"""라벨링용 프레임 뽑기, 조각(타일) 학습 데이터 만들기, 공개 데이터셋 합치기. 라벨은 YOLO 형식
(클래스 x_중심 y_중심 너비 높이, 모두 0~1 비율)이고 공 클래스 번호는 0 하나만 씁니다."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BALL = 0
IMG_EXT = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- 라벨링용 프레임

def sample_frames(video: str, out_dir: str, every_s: float = 1.0, motion_boost: bool = True,
                  max_frames: int | None = None) -> list[str]:
    """`every_s`초마다 한 장, 그리고 화면 변화가 큰 순간(슛·패스로 공이 빠를 때)을 추가로 뽑습니다.
    원본 해상도 그대로 PNG가 아니라 고화질 JPG로 저장합니다 (작은 공이 압축으로 뭉개지지 않게 quality 95)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_s * fps)))
    stem, saved, prev, i = Path(video).stem, [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        moving = prev is not None and motion_boost and float(np.mean(cv2.absdiff(small, prev))) > 6.0
        prev = small
        if i % step == 0 or (moving and i % max(1, step // 4) == 0):
            p = out / f"{stem}_{i:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(str(p))
            if max_frames and len(saved) >= max_frames:
                break
        i += 1
    cap.release()
    return saved


# ---------------------------------------------------------------- 조각(타일)

@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


def read_yolo_labels(path: str | Path, w: int, h: int) -> list[Box]:
    p = Path(path)
    if not p.exists():
        return []
    boxes = []
    for line in p.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:5])
        boxes.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def write_yolo_labels(path: str | Path, boxes: list[Box], w: int, h: int) -> None:
    lines = [f"{b.cls} {(b.x1 + b.x2) / 2 / w:.6f} {(b.y1 + b.y2) / 2 / h:.6f} {(b.x2 - b.x1) / w:.6f} {(b.y2 - b.y1) / h:.6f}"
             for b in boxes]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def tile_origins(w: int, h: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int]]:
    """화면 전체를 덮는 조각 시작점들. 이웃 조각과 `overlap`만큼 겹쳐서 경계에 걸친 공도 어느 한 조각엔 온전히 들어갑니다."""
    def starts(n):
        if n <= tile:
            return [0]
        stride = int(tile * (1 - overlap))
        s = list(range(0, n - tile, stride))
        return s + [n - tile]
    return [(x, y) for y in starts(h) for x in starts(w)]


def crop_boxes(boxes: list[Box], x0: int, y0: int, tile: int, min_visible: float = 0.6) -> list[Box]:
    """조각 안으로 옮긴 라벨. 공이 `min_visible` 비율 미만만 보이면 버립니다 (반쯤 잘린 공은 혼란만 줌)."""
    out = []
    for b in boxes:
        ix1, iy1 = max(b.x1, x0), max(b.y1, y0)
        ix2, iy2 = min(b.x2, x0 + tile), min(b.y2, y0 + tile)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        area = (b.x2 - b.x1) * (b.y2 - b.y1)
        if area <= 0 or (ix2 - ix1) * (iy2 - iy1) / area < min_visible:
            continue
        out.append(Box(b.cls, ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0))
    return out


def list_images(images_dir: str) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXT)


def split_by_time(images: list[Path], val_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    """영상(파일 이름 앞부분)마다 '뒤쪽 시간 구간'을 검증용으로 뗍니다. 이웃 프레임은 거의 같은 장면이라
    무작위로 나누면 검증 점수가 부풀려집니다."""
    groups: dict[str, list[Path]] = {}
    for p in images:
        groups.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    train, val = [], []
    for g in groups.values():
        g = sorted(g)
        k = int(round(len(g) * (1 - val_ratio)))
        train += g[:k]; val += g[k:]
    return train, val


def build_tiles(images: list[Path] | str, labels_dir: str, out_dir: str, split: str = "train", tile: int = 640,
                overlap: float = 0.2, negatives_per_positive: float = 1.0, seed: int = 0) -> dict:
    """원본 프레임 + YOLO 라벨 → 640px 조각 데이터. 공이 있는 조각은 모두, 공이 없는 조각은
    `negatives_per_positive` 비율만큼만 넣습니다 (빈 잔디만 잔뜩 있으면 학습이 '공 없음'으로 쏠림)."""
    rng = np.random.default_rng(seed)
    img_out = Path(out_dir) / "images" / split; img_out.mkdir(parents=True, exist_ok=True)
    lab_out = Path(out_dir) / "labels" / split; lab_out.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    for img_path in (list_images(images) if isinstance(images, str) else images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_labels(Path(labels_dir) / (img_path.stem + ".txt"), w, h)
        pos, neg = [], []
        for x0, y0 in tile_origins(w, h, tile, overlap):
            inside = crop_boxes(boxes, x0, y0, tile)
            (pos if inside else neg).append((x0, y0, inside))
        n_keep_neg = min(len(neg), int(round(max(len(pos), 1) * negatives_per_positive)))
        chosen = pos + [neg[i] for i in rng.choice(len(neg), n_keep_neg, replace=False)] if neg else pos
        for x0, y0, inside in chosen:
            name = f"{img_path.stem}_x{x0}_y{y0}"
            crop = img[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:      # 프레임이 조각보다 작으면 채우기
                pad = np.zeros((tile, tile, 3), np.uint8); pad[:crop.shape[0], :crop.shape[1]] = crop; crop = pad
            cv2.imwrite(str(img_out / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_labels(lab_out / f"{name}.txt", inside, tile, tile)
            n_pos += bool(inside); n_neg += not inside
    return {"positive_tiles": n_pos, "negative_tiles": n_neg}


def write_data_yaml(out_dir: str) -> str:
    p = Path(out_dir) / "data.yaml"
    p.write_text(f"path: {Path(out_dir).resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    return str(p)


def merge_yolo_dataset(src_dir: str, out_dir: str, ball_class_ids: list[int], split_map: dict | None = None,
                       prefix: str = "ext") -> dict:
    """공개 데이터셋(예: Roboflow에서 YOLO 형식으로 내보낸 폴더)을 합칩니다. `ball_class_ids`에 해당하는
    클래스만 공(0)으로 남기고 선수·심판 등 나머지 라벨은 버립니다."""
    split_map = split_map or {"train": "train", "valid": "val", "val": "val", "test": "val"}
    counts = {}
    for src_split, dst_split in split_map.items():
        img_dir = Path(src_dir) / src_split / "images"
        lab_dir = Path(src_dir) / src_split / "labels"
        if not img_dir.exists():
            continue
        (Path(out_dir) / "images" / dst_split).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
        n = 0
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lines = []
            lab = lab_dir / (img.stem + ".txt")
            if lab.exists():
                for line in lab.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and int(parts[0]) in ball_class_ids:
                        lines.append(" ".join([str(BALL)] + parts[1:5]))
            shutil.copy(img, Path(out_dir) / "images" / dst_split / f"{prefix}_{img.name}")
            (Path(out_dir) / "labels" / dst_split / f"{prefix}_{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
        counts[dst_split] = counts.get(dst_split, 0) + n
    return counts
