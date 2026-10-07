"""휴대폰 화각 관련 그림: 기종별 1x 화각, 한 대가 보는 범위, 1x와 0.5x 비교."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from ..camera import Camera, yaw_towards
from ..court import Court
from .figures import RUNOFF, coverage_figure, draw_pitch
from .layout import CAMERA_OFFSET, HFOV_AVERAGE, HFOV_NARROW, HFOV_WIDE, TRIPOD_HEIGHT, diagonal

# 35mm 환산 초점거리 (자료: docs/simulation.md). 동영상 16:9 가로 화각 = 2·atan(17.31 / f)
# (4:3 센서의 가로폭을 그대로 쓰는 16:9 녹화 기준, 손떨림 보정으로 잘리는 부분 제외)
PHONES = [
    ("iPhone 17 Pro · 16 Pro", "Apple", 24.0, "24mm"),
    ("iPhone 17 · 16 · 16e", "Apple", 26.0, "26mm"),
    ("Galaxy S26 · S26+", "Samsung", 23.0, "23mm"),
    ("Galaxy S26 Ultra", "Samsung", 23.5, "23~24mm"),
    ("Galaxy Z Flip7", "Samsung", 23.0, "23mm"),
    ("Galaxy Z Fold7", "Samsung", 24.0, "24mm"),
    ("Galaxy S25 · S25 Ultra", "Samsung", 24.0, "24mm"),
    ("Pixel 10 · 10 Pro · 9", "Google", 25.0, "25mm"),
]
HFOV_ULTRAWIDE = 102.0          # 0.5x (13~14mm 환산) 대표값
BRAND = {"Apple": "#2a78d6", "Samsung": "#eb6834", "Google": "#1baf7a"}   # 검증된 3색 (all-pairs 통과)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]   # 좁은 → 넓은 1x (순서형 파랑)
ULTRA = "#eb6834"


def video_hfov(focal_mm: float) -> float:
    return float(np.degrees(2 * np.arctan(17.31 / focal_mm)))


def phone_fov_figure(out: str):
    rows = sorted(PHONES, key=lambda r: video_hfov(r[2]))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ys = np.arange(len(rows))
    for y, (name, brand, f, flabel) in zip(ys, rows):
        h = video_hfov(f)
        label_x = h + 0.45
        if name == "Galaxy S26 Ultra":       # 자료가 23mm / 24mm로 엇갈림 -> 범위로 표시
            lo, hi = video_hfov(24.0), video_hfov(23.0)
            ax.plot([lo, hi], [y, y], color=BRAND[brand], lw=2, solid_capstyle="round", zorder=3)
            label_x = hi + 0.45
            txt = f"{lo:.1f}~{hi:.1f}°  ({flabel})"
        else:
            txt = f"{h:.1f}°  ({flabel})"
        ax.plot(h, y, "o", ms=10, color=BRAND[brand], mec="white", mew=2, zorder=4)
        ax.text(label_x, y, txt, va="center", fontsize=10, color=INK2)
    avg = np.mean([video_hfov(r[2]) for r in PHONES])
    ax.axvline(avg, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(avg - 0.15, -0.45, f"평균 {avg:.1f}°", ha="right", va="center", fontsize=10, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(64, 78)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("1x 렌즈 동영상(16:9) 가로 화각 (°)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=c, label=b) for b, c in BRAND.items()],
              loc="lower right", frameon=False, fontsize=10)
    ax.set_title("휴대폰 1x 렌즈 화각 비교 (최신·이전 세대)", loc="left", fontsize=14, color=INK, pad=26)
    ax.text(0, 1.02, f"범위 {min(video_hfov(r[2]) for r in PHONES):.1f}~{max(video_hfov(r[2]) for r in PHONES):.1f}°. "
            f"제조사 표기(사진 대각선 82~85°)보다 약 12~13° 좁음. 0.5x 광각은 약 100~106°",
            transform=ax.transAxes, fontsize=10, color=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=120, facecolor="white"); plt.close(fig)


def _visible_mask(court: Court, cam: Camera, cell: float = 0.2):
    xs = np.arange(-RUNOFF, court.length + RUNOFF, cell)
    ys = np.arange(-RUNOFF, court.width + RUNOFF, cell)
    gx, gy = np.meshgrid(xs, ys)
    m = cam.sees_player(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape)
    return xs, ys, m.astype(float)


def _light_pitch(ax, court: Court):
    """색 윤곽선이 잘 보이도록 밝은 바탕에 회색 라인으로 그린 코트."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-RUNOFF, -RUNOFF), court.length + 2 * RUNOFF, court.width + 2 * RUNOFF, color="#f4f6f2", zorder=0))
    for m in court.markings():
        ax.plot(m[:, 0], m[:, 1], color="#9aa39d", lw=1.2, zorder=2)
    for x0 in (0.0, court.length):
        ax.plot([x0, x0], [court.width / 2 - 1.5, court.width / 2 + 1.5], color="#333", lw=3, zorder=3)
    ax.set_xlim(-RUNOFF - .5, court.length + RUNOFF + .5); ax.set_ylim(-RUNOFF - .5, court.width + RUNOFF + .5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def single_phone_reach_figure(court: Court, out: str, twist: float = 5.0):
    """폰 1 한 대가 (발과 머리까지) 볼 수 있는 범위를 화각별로 겹쳐 그림."""
    pos = (-CAMERA_OFFSET, -CAMERA_OFFSET)
    other = (court.length + CAMERA_OFFSET, court.width + CAMERA_OFFSET)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    # 왼쪽: 1x 세 가지 화각, 추천 방향(마주보기에서 왼쪽으로 twist)
    ax = axes[0]; _light_pitch(ax, court)
    yaw = yaw_towards(pos, other, twist)
    handles = []
    for hfov, color, label in zip((HFOV_NARROW, HFOV_AVERAGE, HFOV_WIDE), RAMP, ("좁은 폰 67.3°", "평균 71.5°", "넓은 폰 73.9°")):
        cam = Camera.on_tripod(pos, TRIPOD_HEIGHT, yaw, hfov)
        xs, ys, m = _visible_mask(court, cam)
        inside = m[(ys[:, None] >= 0) & (ys[:, None] <= court.width) & (xs[None, :] >= 0) & (xs[None, :] <= court.length)].mean() * 100
        ax.contourf(xs, ys, m, levels=[.5, 1.5], colors=[color], alpha=.12, zorder=1)
        ax.contour(xs, ys, m, levels=[.5], colors=[color], linewidths=2.2, zorder=4)
        handles.append(Line2D([], [], color=color, lw=2.2, label=f"{label} → 코트의 {inside:.0f}%"))
    ax.plot(*pos, "o", ms=12, mfc="white", mec="#e34948", mew=2.5, zorder=6)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3, fontsize=9.5, frameon=False)
    ax.set_title(f"1x 렌즈: 폰 한 대가 보는 범위 (마주보기에서 왼쪽으로 {twist:g}°)", fontsize=12)
    # 오른쪽: 0.5x, 코너 대각선(45°) 방향
    ax = axes[1]; _light_pitch(ax, court)
    yaw_uw = yaw_towards(pos, (pos[0] + 1, pos[1] + 1))
    cam = Camera.on_tripod(pos, TRIPOD_HEIGHT, yaw_uw, HFOV_ULTRAWIDE)
    xs, ys, m = _visible_mask(court, cam)
    inside = m[(ys[:, None] >= 0) & (ys[:, None] <= court.width) & (xs[None, :] >= 0) & (xs[None, :] <= court.length)].mean() * 100
    ax.contourf(xs, ys, m, levels=[.5, 1.5], colors=[ULTRA], alpha=.12, zorder=1)
    ax.contour(xs, ys, m, levels=[.5], colors=[ULTRA], linewidths=2.2, zorder=4)
    ax.plot(*pos, "o", ms=12, mfc="white", mec="#e34948", mew=2.5, zorder=6)
    ax.legend(handles=[Line2D([], [], color=ULTRA, lw=2.2, label=f"0.5x 약 {HFOV_ULTRAWIDE:g}° → 코트의 {inside:.0f}%")],
              loc="upper center", bbox_to_anchor=(.5, -.01), fontsize=9.5, frameon=False)
    ax.set_title("0.5x 광각: 폰 한 대가 보는 범위 (코너 대각선 45° 방향)", fontsize=12)
    fig.suptitle(f"{court.length:g}×{court.width:g} m 코트 · 삼각대 {TRIPOD_HEIGHT:g} m · 선수 발과 머리가 모두 화면에 들어오는 범위", fontsize=11, color=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=110, facecolor="white"); plt.close(fig)


def lens_compare_figure(court: Court, out: str, twist: float = 5.0):
    """두 대 배치에서 1x(추천 방향)와 0.5x(코너 대각선 방향) 커버리지·정확도 비교."""
    one_x = diagonal(court, twist)
    pos = (-CAMERA_OFFSET, -CAMERA_OFFSET)
    other = (court.length + CAMERA_OFFSET, court.width + CAMERA_OFFSET)
    bisector_twist = yaw_towards(pos, (pos[0] + 1, pos[1] + 1)) - yaw_towards(pos, other)
    ultra = diagonal(court, bisector_twist)
    coverage_figure(court, one_x, out,
                    fovs=((HFOV_WIDE, "1x (가장 넓은 기종)", one_x), (HFOV_ULTRAWIDE, "0.5x 광각, 코너 대각선 방향", ultra)),
                    title=f"렌즈별 비교 · 대각선 코너 2대 · 삼각대 {TRIPOD_HEIGHT:g} m · 1080p")


def all_fov_figures(court: Court, outdir: str, twist: float = 5.0):
    phone_fov_figure(f"{outdir}/fov_phones.png")
    single_phone_reach_figure(court, f"{outdir}/fov_reach.png", twist)
    lens_compare_figure(court, f"{outdir}/fov_lens_compare.png", twist)
