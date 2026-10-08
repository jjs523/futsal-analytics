"""python -m futsal.ball <명령> ...   (설명은 futsal/ball/__init__.py)"""
from __future__ import annotations

import argparse
import json

from . import dataset, track


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m futsal.ball", description="공 검출 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frames", help="라벨링할 프레임 뽑기")
    f.add_argument("video"); f.add_argument("--out", required=True)
    f.add_argument("--every", type=float, default=1.0, help="몇 초마다 한 장")
    f.add_argument("--max", type=int, default=None)

    t = sub.add_parser("tiles", help="원본 프레임 + 라벨 → 640px 조각 학습 데이터")
    t.add_argument("--images", required=True); t.add_argument("--labels", required=True); t.add_argument("--out", required=True)
    t.add_argument("--val-ratio", type=float, default=0.2); t.add_argument("--tile", type=int, default=640)
    t.add_argument("--neg", type=float, default=1.0, help="공 있는 조각 1개당 빈 조각 수")

    m = sub.add_parser("merge", help="공개 데이터셋(YOLO 형식) 합치기")
    m.add_argument("--src", required=True); m.add_argument("--out", required=True)
    m.add_argument("--ball-class", type=int, nargs="+", required=True, help="원본 데이터셋에서 공 클래스 번호")
    m.add_argument("--prefix", default="ext")

    r = sub.add_parser("train", help="YOLO 학습 (GPU 권장, pip install '.[vision]')")
    r.add_argument("--data", required=True); r.add_argument("--model", default="yolo11s.pt")
    r.add_argument("--epochs", type=int, default=100); r.add_argument("--batch", type=int, default=16)
    r.add_argument("--name", default="ball")

    d = sub.add_parser("detect", help="영상에서 공 찾기 + 궤적 정리")
    d.add_argument("video"); d.add_argument("--model", required=True); d.add_argument("--out", required=True)
    d.add_argument("--conf", type=float, default=0.15); d.add_argument("--stride", type=int, default=1)

    a = ap.parse_args(argv)
    if a.cmd == "frames":
        print(f"{len(dataset.sample_frames(a.video, a.out, a.every, max_frames=a.max))}장 저장: {a.out}")
    elif a.cmd == "tiles":
        tr, va = dataset.split_by_time(dataset.list_images(a.images), a.val_ratio)
        s1 = dataset.build_tiles(tr, a.labels, a.out, "train", a.tile, negatives_per_positive=a.neg)
        s2 = dataset.build_tiles(va, a.labels, a.out, "val", a.tile, negatives_per_positive=a.neg)
        print(json.dumps({"train": s1, "val": s2, "data_yaml": dataset.write_data_yaml(a.out)}, ensure_ascii=False))
    elif a.cmd == "merge":
        print(dataset.merge_yolo_dataset(a.src, a.out, a.ball_class, prefix=a.prefix))
        dataset.write_data_yaml(a.out)
    elif a.cmd == "train":
        from ultralytics import YOLO
        # 작은 공용 설정: 조각(640) 단위 학습, 축소 증강은 약하게(공이 더 작아져 사라지지 않게), 뒤집기·색 변화는 유지
        YOLO(a.model).train(data=a.data, imgsz=640, epochs=a.epochs, batch=a.batch, name=a.name,
                            scale=0.3, mosaic=1.0, close_mosaic=15, fliplr=0.5, hsv_v=0.4, degrees=0.0, patience=30)
    elif a.cmd == "detect":
        pts = track.track_video(a.video, track.yolo_tile_detector(a.model, a.conf), stride=a.stride)
        track.save(pts, a.out)
        found = sum(p.source == "detected" for p in pts); filled = sum(p.source == "interpolated" for p in pts)
        print(f"프레임 {len(pts)}개: 검출 {found}, 보간 {filled}, 없음 {len(pts) - found - filled} → {a.out}")


if __name__ == "__main__":
    main()
