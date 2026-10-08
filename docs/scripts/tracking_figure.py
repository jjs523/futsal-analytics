"""docs/img/tracking_overview.png: 기존 다중 객체 추적(MOT) 방법 정리 → 우리 파이프라인이 무엇을 가져왔는지.

python docs/scripts/tracking_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

OUT = Path(__file__).resolve().parents[1] / "img" / "tracking_overview.png"
INK, INK2, LINE, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
FAM = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]       # 계열별 색 (검증된 순서)
OURS = "#1f7a5c"

FAMILIES = [
    ("움직임 기반", "SORT · ByteTrack · OC-SORT", ["다음 위치 예측 + 박스 겹침으로 연결", "아주 빠름 (폰에서 거의 공짜)",
                                               "같은 옷 선수가 엇갈리면 ID 바뀜"], "폰 적합 ●●●"),
    ("생김새(Re-ID) 결합", "DeepSORT · FairMOT · BoT-SORT", ["외형을 숫자 벡터로 바꿔 비교", "엇갈림·재등장에 강함",
                                                      "같은 유니폼끼리는 구분 약함"], "폰 적합 ●●○"),
    ("경기 후 전역 정리", "MPNTrack · SUSHI · GTA(스포츠)", ["미래 프레임까지 보고 한꺼번에 묶기", "섞인 트랙 쪼개기 + 끊긴 조각 잇기",
                                                    "실시간 불가 (경기 후 분석엔 문제없음)"], "폰 적합 ●●●"),
    ("트랜스포머 일체형", "TrackFormer · MOTR · MOTIP", ["검출과 ID 연결을 한 모델이 처리", "최신 연구 성능",
                                                  "무겁고 학습 데이터 많이 필요"], "폰 적합 ○○○"),
]
SPORTS = "스포츠 특화: SportsMOT · Deep-EIoU · 등번호 인식 · 팀 자동 분류 · SoccerNet 경기 상태 재구성(미니맵)"

PIPE = [
    ("① 검출", "YOLO\n발 위치 + 상·하체 색", None, []),
    ("② 경기장 좌표", "기준점 보정 →\n두 카메라 병합", None, []),
    ("③ 짧은 조각", "움직임 예측으로 연결\n헷갈리면 끊기", 0, [("← 움직임 기반", 0)]),
    ("④ 경기 후 다시 묶기", "색 · 이동 가능 거리 ·\n시간 겹침 없음", 2, [("← 전역 정리(GTA)", 2), ("← 생김새: 헷갈릴 때만", 1)]),
    ("⑤ 사람 확인", "시작 때 이름 탭\n헷갈린 장면만 확인", None, []),
]


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def main():
    fig = plt.figure(figsize=(15, 9.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.955, "선수 ID 추적: 기존 방법과 우리가 고른 조합", fontsize=19, weight="bold", color=INK)
    ax.text(0.03, 0.918, "다중 객체 추적(MOT)의 네 계열 → 폰에서 돌릴 수 있는 것만 골라 조합", fontsize=12, color=INK2)

    # 위: 네 계열
    w, h, gap, top = 0.225, 0.33, 0.012, 0.545
    for i, (name, ex, lines, fit) in enumerate(FAMILIES):
        x = 0.03 + i * (w + gap)
        used = i in (0, 1, 2)
        box(ax, x, top, w, h, SURF, FAM[i], lw=2.4 if used else 1.0)
        ax.add_patch(plt.Rectangle((x, top + h - 0.012), w, 0.012, color=FAM[i], lw=0))
        ax.text(x + 0.015, top + h - 0.05, name, fontsize=14, weight="bold", color=INK)
        ax.text(x + 0.015, top + h - 0.088, ex, fontsize=10, color=INK2)
        for k, ln in enumerate(lines):
            ax.text(x + 0.015, top + h - 0.15 - k * 0.055, ("· " if k < 2 else "△ ") + ln, fontsize=10.5, color=INK)
        ax.text(x + 0.015, top + 0.03, fit, fontsize=11, color=INK2)
        if used:
            ax.text(x + w - 0.015, top + 0.03, "사용", fontsize=11, weight="bold", color=FAM[i], ha="right")
    box(ax, 0.03, 0.47, 0.94, 0.05, "#f2f1ec", LINE, r=0.01)
    ax.text(0.045, 0.495, SPORTS, fontsize=11, color=INK, va="center")

    # 아래: 우리 파이프라인
    ax.text(0.03, 0.405, "우리 파이프라인 (각 폰 + 경기 후)", fontsize=14, weight="bold", color=OURS)
    pw, ph, py = 0.168, 0.22, 0.14
    xs = [0.03 + k * (pw + 0.025) for k in range(len(PIPE))]
    for k, (title, body, fam, tags) in enumerate(PIPE):
        ec = FAM[fam] if fam is not None else OURS
        box(ax, xs[k], py, pw, ph, "#eef6f2", ec, lw=2)
        ax.text(xs[k] + 0.012, py + ph - 0.04, title, fontsize=13, weight="bold", color=INK)
        for j, (tag, tf) in enumerate(tags):
            ax.text(xs[k] + 0.012, py + ph - 0.08 - j * 0.035, tag, fontsize=10, weight="bold", color=FAM[tf])
        ax.text(xs[k] + 0.012, py + 0.035, body, fontsize=10.5, color=INK, va="bottom", linespacing=1.4)
        if k < len(PIPE) - 1:
            ax.add_patch(FancyArrowPatch((xs[k] + pw + 0.003, py + ph / 2), (xs[k + 1] - 0.003, py + ph / 2),
                                         arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.5))
    ax.text(0.03, 0.06, "결과(합성 2분 경기, 10명): IDF1 0.24~0.26 (움직임만) → 0.53~0.68 (생김새 + 경기 후 다시 묶기).  "
            "실제 영상 라벨링 후 재측정 예정", fontsize=11, color=INK2)
    fig.savefig(OUT, dpi=110, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
