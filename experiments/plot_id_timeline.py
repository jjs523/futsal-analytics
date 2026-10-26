"""When each identity is seen over the whole match, for two runs side by side (one row per id, by first appearance).

python experiments/plot_id_timeline.py experiments/full_match26 experiments/full_match26_gta --out docs/img/id_timeline.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
TEAM = {"A": "#2a78d6", "B": "#eb6834"}          # categorical slots 1, 2; undecided team in muted ink
LABEL = {"A": "A팀", "B": "B팀", "U": "팀 미정"}


def segments(seen: np.ndarray, fps: float, min_gap_s: float = 2.0) -> list[tuple[int, int]]:
    """Runs of seen frames, gaps shorter than min_gap_s bridged (a dot per frame would be unreadable)."""
    idx = np.flatnonzero(seen)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > min_gap_s * fps)
    starts, ends = np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]
    return list(zip(starts, ends + 1))


def load(run_dir: str):
    d = json.load(open(os.path.join(run_dir, "tracks.json"), encoding="utf-8"))
    fps, start = float(d["fps"]), float(d.get("start", 0.0))
    rows = []
    for p in d["players"]:
        seen = np.array([v is not None for v in p["x"]])
        if seen.any():
            rows.append((int(np.argmax(seen)), p.get("team") or "U", seen))
    rows.sort(key=lambda r: r[0])
    return rows, fps, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--titles", nargs=2, default=["5+1 자리 배정 (기본)", "GTA 방식"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams["font.family"] = "Malgun Gothic"
    data = [load(r) for r in a.runs]
    n = [len(r[0]) for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.13 * max(n) + 1.6), facecolor=SURFACE, sharex=True)
    for ax, (rows, fps, start), title in zip(axes, data, a.titles):
        ax.set_facecolor(SURFACE)
        total = max(len(s) for *_, s in rows)
        for y, (_, team, seen) in enumerate(rows):
            for s, e in segments(seen, fps):
                ax.barh(y, (e - s) / fps / 60, left=(start + s / fps) / 60, height=0.62,
                        color=TEAM.get(team, MUTED), linewidth=0)
        long = sum(1 for *_, s in rows if s.mean() > 0.8)
        ax.set_title(f"{title}: ID {len(rows)}개, 경기의 80% 이상 이어진 ID {long}개", loc="left", fontsize=11, color=INK)
        ax.set_ylim(max(n) - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xlabel("경기 시각 (cam1 기준, 분)", color=MUTED, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAM["A"], TEAM["B"], MUTED)]
    fig.legend(handles, [LABEL["A"], LABEL["B"], LABEL["U"]], loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.text(0.01, 0.005, "한 줄이 ID 하나(처음 나온 순서), 막대가 그 ID가 보인 구간입니다. 10/8 테스트 경기 53분, YOLO26x 검출.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(a.out, n)


if __name__ == "__main__":
    main()
