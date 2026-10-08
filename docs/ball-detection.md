# 공 검출: 학습 계획과 도구

COCO로 학습된 YOLO 사전학습 모델에도 `sports ball`(클래스 32)이 있지만, 우리 촬영 조건에서는 거의 못 찾습니다.
그래서 **지금부터 우리 영상 라벨링 + 공개 풋살 데이터로 공 전용 모델을 학습**합니다. 코드는 `src/futsal/ball/`에 있습니다.

## 왜 기본 모델로는 안 되나: 공이 너무 작음

풋살공(4호) 지름은 약 20cm입니다. 1x 렌즈(가로 화각 약 70°), 1080p 기준 화면 속 공 크기:

| 카메라와 거리 | 1080p 원본 | 640px로 줄였을 때 (YOLO 기본 입력) |
|---|---|---|
| 10m | 약 27px | 약 9px |
| 20m | 약 14px | 약 5px |
| 30m | 약 9px | 약 3px |
| 45m (반대편 코너) | 약 6px | 약 2px |

YOLO는 화면 전체를 640px로 줄여서 보기 때문에 먼 공은 2~5px 점이 되어 사라집니다. 게다가 빠르게 차면 번지고(모션 블러), 선수 발·몸에 자주 가려집니다.

**대책 세 가지**
1. **조각(타일) 검출**: 원본 해상도를 640px 조각으로 나눠 학습·검출합니다. 1080p 한 장 = 조각 8개 (20% 겹침). 공이 줄어들지 않습니다.
2. **공 전용 파인튜닝**: 공 한 클래스만, 우리 구장·조명·공으로 학습합니다.
3. **궤적 정리**: 공은 한 프레임에 하나뿐이고 연속으로 움직이므로, 프레임마다 후보 중 궤적에 이어지는 것 하나를 고르고 짧게 놓친 구간은 앞뒤로 채웁니다. 오검출(머리, 흰 신발, 라인)을 상당 부분 걸러냅니다.

## 전체 흐름

```bash
pip install -e ".[vision]"

# 1) 라벨링할 프레임 뽑기 (1초에 한 장 + 화면 변화가 큰 순간 추가)
python -m futsal.ball frames cam1.mp4 --out labeling/cam1 --every 1.0
python -m futsal.ball frames cam2.mp4 --out labeling/cam2 --every 1.0

# 2) 라벨링 (아래 규칙) → YOLO 형식 라벨을 labeling/labels/ 에 저장

# 3) 원본 프레임 + 라벨 → 640px 조각 학습 데이터 (영상마다 뒤쪽 20% 시간 구간을 검증용으로 분리)
python -m futsal.ball tiles --images labeling/cam1 --labels labeling/labels --out datasets/ball
python -m futsal.ball tiles --images labeling/cam2 --labels labeling/labels --out datasets/ball

# 4) 공개 데이터셋 합치기 (공 클래스 번호만 지정하면 나머지 라벨은 버림)
python -m futsal.ball merge --src downloads/futsal-ball --ball-class 0 --out datasets/ball --prefix rf1

# 5) 학습 (GPU 필요 → Colab 권장, 아래 참고)
python -m futsal.ball train --data datasets/ball/data.yaml --model yolo11s.pt --epochs 100

# 6) 영상에서 공 찾기 + 궤적 정리 → 프레임별 공 위치 JSON
python -m futsal.ball detect cam1.mp4 --model runs/detect/ball/weights/best.pt --out cam1.ball.json
```

결과 `cam1.ball.json`은 프레임마다 `{frame, t, u, v, conf, source}`이고 `source`는 `detected`(검출) / `interpolated`(보간) / `missing`(없음)입니다. 좌표는 **픽셀**입니다 (이유는 [공중에 뜬 공](#한계-공중에-뜬-공) 참고).

## 데이터

### 1) 우리 영상 (가장 중요)
공개 데이터는 카메라 높이·거리·공·조명이 달라서, **우리 촬영 조건의 라벨이 성능을 결정합니다.**
- 첫 목표: **두 카메라 합쳐 300~500장.** 2주 차에 1,000장까지.
- 고르게 담기: 가까운 공 / 먼 공, 바닥 공 / 뜬 공, 드리블 중(발에 붙음), 슛(번짐), 골키퍼 손, 공 없는 장면(교체·휴식).
- 같은 경기에서 이웃 프레임은 거의 같은 장면이므로, 검증은 **시간 구간으로 나눕니다** (`tiles` 명령이 자동으로 처리). 무작위로 나누면 점수가 부풀려집니다.

### 2) 공개 데이터셋 (보조)
Roboflow Universe(universe.roboflow.com)에서 `futsal ball`로 검색하면 풋살 공 데이터셋이 여러 개 나옵니다. 지금 확인된 것:

| 데이터셋 (Roboflow 작성자 / 이름) | 규모 | 비고 |
|---|---|---|
| masters-program / futsal-ball-detection | - | 풋살 공 전용 |
| david-carvalho / Futsal Ball detection | 약 986장 | 풋살 공 전용 |
| arnathorn / pro-futsal-detection | 약 836장 | 선수 등 여러 클래스 → 공 클래스만 남김 |

- 내려받을 때 형식은 **YOLOv8 (YOLO txt)**, 그대로 `merge` 명령에 넣습니다. 데이터셋마다 공 클래스 번호가 다르니 `data.yaml`의 `names`에서 확인해 `--ball-class`로 지정합니다.
- **라이선스를 반드시 확인**합니다 (대부분 CC BY 4.0 → 발표 자료에 출처 표기).
- 축구 데이터인 **SoccerNet-Tracking**에도 공 박스가 많지만(약 21만 개), 방송 화면이라 시점이 다르고 이용 조건(신청·연구용)을 확인해야 합니다. 1차에는 풋살 데이터만으로 시작합니다.

## 라벨링 규칙

도구는 아무거나 괜찮습니다 (Roboflow, CVAT, Label Studio). **YOLO 형식으로 내보내기**만 맞추면 됩니다.

- 클래스는 **`ball` 하나**(번호 0).
- 박스는 **보이는 공에 딱 맞게.** 너무 크게 그리면 작은 공에서 위치가 크게 틀어집니다.
- **번진 공**: 번진 모양 전체를 감쌉니다 (실제 영상에서 그렇게 보이니까).
- **가려진 공**: 절반 이상 보이면 라벨, 그보다 적게 보이면 라벨하지 않습니다.
- **경기 밖 공**(벤치, 옆 코트): 공처럼 보이면 라벨합니다. 경기 공 고르기는 궤적 정리가 맡습니다.
- 헷갈리는 장면은 건너뛰지 말고 팀 채널에 캡처를 올려 기준을 맞춥니다. **사람마다 기준이 다르면 학습이 망가집니다.**
- 파일 이름 규칙: `frames` 명령이 만드는 `cam1_000123.jpg` 그대로 둡니다 (영상별 시간 분리에 사용).

## 학습 (Colab)

GPU가 없으면 Google Colab 무료 T4로 충분합니다.

```python
!git clone https://github.com/jjs523/futsal-analytics && cd futsal-analytics && pip install -e ".[vision]"
from google.colab import drive; drive.mount('/content/drive')
# datasets/ball 폴더를 Drive에 올려두고 경로 지정 (data.yaml 안의 path도 Colab 경로로 바꾸기)
!cd futsal-analytics && python -m futsal.ball train --data /content/drive/MyDrive/ball/data.yaml --model yolo11s.pt --epochs 100
```

- 학습 설정(`train` 명령 안): 축소 증강은 약하게(`scale=0.3`, 공이 더 작아져 사라지지 않게), 좌우 뒤집기·밝기 변화는 유지, 회전 없음, 30 에폭 개선 없으면 조기 종료.
- 모델 크기: **YOLO11s로 시작**해서 정확도를 확인하고, 폰 속도 문제가 있으면 YOLO11n과 비교합니다.
- 결과 `runs/detect/ball/weights/best.pt`를 Drive에 저장해 팀이 공유합니다.

## 평가

1. **검증 데이터 점수**: 학습 끝에 나오는 mAP50, 재현율(recall). 공은 하나라서 **재현율(놓치지 않기)**이 특히 중요합니다.
2. **거리별 확인**: 가까운 공·먼 공으로 나눠 놓친 비율을 봅니다. 먼 쪽이 약하면 그쪽 라벨을 더 모읍니다.
3. **궤적 정리 후 결과**: 실제 영상 1~2분을 `detect`로 돌려 `missing` 비율, 엉뚱한 곳으로 튀는 횟수를 셉니다.

비교 기준선은 **COCO 사전학습 모델 + 조각 검출**입니다 (같은 `detect` 명령에 `--model yolo11s.pt`). 파인튜닝 효과를 이 기준과 비교해 발표에 씁니다.

## 폰에서 돌릴 때 비용

조각 검출은 1080p 한 장에 조각 8개 = **선수 검출 1번의 약 8배.** 게다가 공은 빨라서 선수보다 촘촘하게(매 프레임~2프레임마다) 봐야 합니다. 그대로는 폰에서 너무 느리므로:

- **공 근처 조각만 보기**: 공을 찾은 뒤에는 예측 위치 주변 조각 1~2개만 검출하고, 놓쳤을 때만 전체를 다시 훑습니다. 평균 비용이 크게 줄어듭니다.
- **선수+공 한 모델**: 선수와 공을 한 모델(클래스 2개)로 함께 학습하면 검출을 한 번만 돌리면 됩니다. 1차는 공 모델을 따로 만들어 품질을 확인한 뒤 합칠지 정합니다.
- 공은 **트랙 A(PC)에서 먼저 품질을 확인**하고, 폰 적용은 [폰 속도 측정](phone-benchmark.md) 결과를 보고 정합니다.

## 한계: 공중에 뜬 공

선수는 발이 바닥에 있으므로 바닥 기준 좌표 변환(호모그래피)으로 미터 좌표를 얻습니다. **공은 뜨면 이 변환이 틀립니다** (높이 뜬 공이 실제보다 먼 곳에 찍힘). 그래서 1차 결과는 픽셀 좌표로 두고:
- 공이 바닥에 있을 때(풋살에서는 대부분)는 바닥 변환을 그대로 써도 됩니다.
- 두 카메라에서 동시에 보이면 두 시선의 교차점으로 **높이까지 계산**할 수 있습니다 (다음 단계).

## 다음 단계 아이디어
- 공 근처 조각만 보는 검출 (폰 비용 절감)
- 연속 프레임 여러 장을 함께 보는 공 전용 모델 (TrackNet 계열, 번진 공·가려진 공에 강함)
- 두 카메라로 공 높이 계산, 슛 속도·패스 거리 같은 공 지표
