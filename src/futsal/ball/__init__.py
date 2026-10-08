"""공 검출: 라벨링용 프레임 뽑기 → 조각(타일) 학습 데이터 → YOLO 학습 → 원본 해상도 조각 검출 → 공 궤적 정리.

공은 1080p 화면에서 수~수십 픽셀밖에 안 되기 때문에, 화면을 통째로 640px로 줄이면 사라집니다.
그래서 학습과 검출 모두 원본 해상도를 640px 조각으로 나눠서 합니다.

  python -m futsal.ball frames  cam1.mp4 --out labeling/cam1 --every 1.0     라벨링할 프레임 뽑기
  python -m futsal.ball tiles   --images ... --labels ... --out datasets/ball 조각 학습 데이터 만들기
  python -m futsal.ball merge   --src roboflow_export --map 0 --out datasets/ball  공개 데이터셋 합치기
  python -m futsal.ball train   --data datasets/ball/data.yaml --model yolo11s.pt  학습 (GPU, ultralytics 필요)
  python -m futsal.ball detect  cam1.mp4 --model best.pt --out cam1.ball.json    영상에서 공 찾기 + 궤적 정리
"""
