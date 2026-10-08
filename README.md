# futsal-analytics

일반 휴대폰 2대로 풋살 경기를 찍어서 **선수 위치를 2D 지도에 표시하고 이동거리·속도·히트맵을 계산**하는 시스템입니다. (AX 캡스톤 디자인)

**서버 없이 휴대폰에서 모두 처리합니다.** 각 폰이 경기 후 자기 영상에서 선수를 검출하고, 결과(수 MB)만 한 폰으로 모아 병합·추적·기록 계산까지 합니다. 영상은 폰 밖으로 나가지 않아 서버 비용이 들지 않습니다. → [시스템 구조](docs/architecture.md)

![2D 지도 데모](docs/img/demo.png)

## 무엇이 들어 있나

| 경로 | 내용 | 상태 |
|---|---|---|
| `src/futsal/` | **분석 기준 구현**: FIFA 규격 코트, 카메라 모델, 기준점 보정, 두 카메라 병합, 지표, 시간 맞추기, 결과 포맷. 앱(Dart)으로 옮길 원본이자 정답 비교용 | ✅ 테스트 있음 |
| `src/futsal/sim/` | 배치·화각 시뮬레이터, 합성 경기·합성 영상 | ✅ |
| `src/futsal/pipeline/` | PC에서 실제 영상 → 검출 → 병합 → 추적 (앱 결과와 비교할 정답 생성) | ✅ 합성 영상으로 검증, 실제 영상 검증 필요 |
| `server/` | **개발용 PC 도구** (운영 서버 아님): 영상 올려서 분석·확인 | ✅ 테스트 있음 |
| `web/viewer/` | 2D 지도 뷰어 (재생, 꼬리, 선수 기록, 히트맵). 앱 웹뷰·공유용 | ✅ |
| `app/` | Flutter 앱 (촬영 → 폰 안에서 검출 → 결과) | ⏳ 다음 단계 |
| `docs/` | 구조·촬영 가이드·시뮬레이션·로드맵 | |

## 빠른 시작 (PC)

```bash
pip install -e ".[server,sim,dev]"        # 실제 영상 검출까지 하려면 ".[vision]" 추가
pytest                                     # 전체 테스트

uvicorn server.app.main:app --reload       # 개발용 도구: http://localhost:8000 → "데모 경기 만들기"
```

서버 없이 뷰어만 보려면: `python -m http.server -d web/viewer 8080` 후 `http://localhost:8080/?src=sample_tracks.json`

시뮬레이터:
```bash
python -m futsal.sim figures --court 40x20 --twist 5 --out docs/img   # 규격 도면, 커버리지, 데모
python -m futsal.sim fov --out docs/img                                # 화각 그림
python -m futsal.sim twist --court 42x25                               # 그 구장에서 안전한 회전각
python -m futsal.sim accuracy --hfov 67.3                              # 합성 경기 위치 오차
```

## 문서
- [시스템 구조 (폰에서 모두 처리)](docs/architecture.md)
- [촬영 가이드 (설치 위치·각도·설정)](docs/capture-guide.md)
- [시뮬레이션 방법과 결과](docs/simulation.md)
- [로드맵과 다음 단계](docs/roadmap.md)
- [팀원 폰 YOLO 검출 속도 측정 방법](docs/phone-benchmark.md)
- [선수 ID 추적: 기존 방법 조사와 우리 선택](docs/tracking-survey.md)
