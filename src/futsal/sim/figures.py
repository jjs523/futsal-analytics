"""Figures for docs/: pitch spec drawing, coverage/accuracy maps and a synthetic end-to-end demo."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

try:
    import koreanize_matplotlib  # noqa: F401  (Korean labels)
except ImportError:  # pragma: no cover
    pass

from ..court import GOAL_DEPTH, GOAL_WIDTH, Court
from ..camera import PLAYER_HEIGHT
from . import coverage as cov
from . import observe, scenario
from .layout import HFOV_AVERAGE, HFOV_NARROW, HFOV_WIDE, TRIPOD_HEIGHT, Layout, diagonal

RUNOFF = 2.8
PITCH_GREEN, LINE = "#23b14d", "white"
ZONE_COLORS = ["#cfe8d6", "#f2c14e", "#5aa9e6", "#e5383b"]
TEAM_COLORS = {"A": "#e8473b", "B": "#2f6fe0"}


def draw_pitch(ax, court: Court, lw: float = 1.6, runoff: bool = True):
    L, W = court.length, court.width
    if runoff:
        ax.add_patch(Rectangle((-RUNOFF, -RUNOFF), L + 2 * RUNOFF, W + 2 * RUNOFF, color=PITCH_GREEN, zorder=0))
    for m in court.markings():
        ax.plot(m[:, 0], m[:, 1], color=LINE, lw=lw, zorder=4)
    for x, y in court.spots():
        ax.plot(x, y, "o", ms=3, color=LINE, zorder=4)
    for x0, dx in ((0.0, -GOAL_DEPTH), (L, GOAL_DEPTH)):
        ax.add_patch(Rectangle((min(x0, x0 + dx), W / 2 - GOAL_WIDTH / 2), GOAL_DEPTH, GOAL_WIDTH,
                               fc="#f4f4f4", ec="#222", lw=1.2, hatch="////", zorder=8))
        ax.plot([x0, x0], [W / 2 - GOAL_WIDTH / 2, W / 2 + GOAL_WIDTH / 2], color="#111", lw=3, zorder=9)
    ax.set_xlim(-RUNOFF - .5, L + RUNOFF + .5); ax.set_ylim(-RUNOFF - .5, W + RUNOFF + .5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def draw_phones(ax, layout: Layout, length: float = 4.0):
    for c in layout.cameras:
        a = np.radians(c.yaw_deg)
        ax.annotate("", xy=(c.xy[0] + length * np.cos(a), c.xy[1] + length * np.sin(a)), xytext=c.xy,
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2), zorder=10)
        ax.plot(*c.xy, "o", ms=12, mfc="white", mec="red", mew=2.5, zorder=11)


def spec_figure(court: Court, layout: Layout, out: str):
    fig, ax = plt.subplots(figsize=(13, 7.4))
    draw_pitch(ax, court, lw=2)
    for i, c in enumerate(layout.cameras):
        half = np.radians(HFOV_AVERAGE / 2); a = np.radians(c.yaw_deg)
        for e in (a - half, a + half):
            ax.plot([c.xy[0], c.xy[0] + 60 * np.cos(e)], [c.xy[1], c.xy[1] + 60 * np.sin(e)], ":", color="red", lw=1, zorder=5)
        ax.plot(*c.xy, "o", ms=13, mfc="white", mec="red", mew=2.5, zorder=10)
        ax.annotate(f"폰 {i + 1}", c.xy, xytext=(10, -14 if i == 0 else 8), textcoords="offset points", color="red", weight="bold")
    for x, y in court.keypoints().values():
        ax.plot(x, y, "x", ms=6, mew=1.6, color="#ffd400", zorder=11)
    L, W = court.length, court.width

    def dim(x0, y0, x1, y1, txt, dy=0.0):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="<->", lw=1), zorder=12)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", va="center", fontsize=9, zorder=12, bbox=dict(fc="white", ec="none", pad=1))
    dim(0, -2.2, L, -2.2, f"터치라인 {L:g} m (국제 38~42)")
    dim(L + 2.0, 0, L + 2.0, W, f"골라인\n{W:g} m\n(국제 18~25)")
    dim(L / 2, W / 2, L / 2 + 3, W / 2, "3 m", .7)
    dim(0, W / 2 + 4.5, 6, W / 2 + 4.5, "6 m", .6)
    dim(0, W / 2 - 4.8, 10, W / 2 - 4.8, "10 m (제2 페널티마크)", -.7)
    xs = court.substitution_marks_x()
    dim(xs[0], -1.0, xs[1], -1.0, "교체구역 5 m", -.6)
    ax.set_xlim(-RUNOFF - .5, L + RUNOFF + .5); ax.set_ylim(-RUNOFF - .5, W + RUNOFF + .5)
    ax.set_title(f"FIFA 풋살 규격 {L:g}×{W:g} m · 골대 3×2 m · 페널티구역 반경 6 m · 센터서클 반경 3 m\n"
                 f"빨간 점선 = 폰 시야(1x, 평균 {HFOV_AVERAGE:g}°) · 노란 × = 보정 때 탭할 수 있는 기준점 {len(court.keypoints())}개", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def coverage_figure(court: Court, layout: Layout, out: str, fovs=((HFOV_NARROW, "화각 좁은 폰 (26mm, 아이폰 기본형)"), (HFOV_WIDE, "화각 넓은 폰 (23mm, 갤럭시 S26·Flip7)"))):
    fig, axes = plt.subplots(2, len(fovs), figsize=(7.5 * len(fovs), 10.5), gridspec_kw=dict(hspace=.45), squeeze=False)
    cmap = ListedColormap(ZONE_COLORS)
    for col, (hfov, label) in enumerate(fovs):
        c = cov.analyse(court, layout, hfov)
        s = c.summary()
        ext = (0, court.length, 0, court.width)
        ax = axes[0, col]; draw_pitch(ax, court)
        ax.imshow(c.zone, extent=ext, origin="lower", cmap=cmap, vmin=-.5, vmax=3.5, alpha=.95, zorder=2)
        draw_phones(ax, layout)
        ax.set_title(f"{label} {hfov:g}°\n두 대 모두 {s['both_pct']:.0f}% · 한 대만 {s['single_pct']:.0f}% · 음영 {s['blind_m2']:.0f} m²")
        ax = axes[1, col]; draw_pitch(ax, court)
        im = ax.imshow(c.err, extent=ext, origin="lower", cmap="viridis_r", vmin=0, vmax=.75, alpha=.95, zorder=2)
        ax.contour(c.xs, c.ys, c.err, levels=[0.4], colors="white", linewidths=1.5, linestyles="--", zorder=5)
        draw_phones(ax, layout)
        ax.set_title(f"발 위치 1px 어긋날 때 좌표 오차: 중앙값 {s['err_median']:.2f} m, 최대 {s['err_max']:.2f} m\n점선 안쪽 = 0.4 m 이상 약한 구역 ({s['weak_pct']:.0f}%)")
        fig.colorbar(im, ax=ax, fraction=.025, label="m / px")
    fig.legend(handles=[Patch(color=z, label=l) for z, l in zip(ZONE_COLORS, ["두 대 모두 보임", "폰1만", "폰2만", "음영"])],
               loc="upper center", ncol=4, bbox_to_anchor=(.5, .985), frameon=False)
    fig.suptitle(f"{layout.name} · 1x · 삼각대 {TRIPOD_HEIGHT:g} m · 1080p  (선수 발과 머리가 모두 화면에 있어야 '보임')", y=.94)
    fig.subplots_adjust(top=.86, bottom=.03, left=.03, right=.97, wspace=.12)
    fig.savefig(out, dpi=105); plt.close(fig)


def _render_view(ax, cam, court: Court, truth, frame: int, title: str):
    ax.set_facecolor("#2a7a3b")

    def poly(P3):
        P3 = np.asarray(P3, float)
        Xc = (cam.R @ (P3 - cam.C).T).T
        uv = (cam.K @ Xc.T).T
        uv = uv[:, :2] / uv[:, 2:]
        uv[Xc[:, 2] < .1] = np.nan
        return uv
    for m in court.markings():
        q = np.vstack([np.linspace(m[i], m[i + 1], 12) for i in range(len(m) - 1)])
        uv = poly(np.hstack([q, np.zeros((len(q), 1))])); ax.plot(uv[:, 0], uv[:, 1], color="white", lw=1.3)
    for g in court.goal_frames():
        uv = poly(g); ax.plot(uv[:, 0], uv[:, 1], color="white", lw=2.2)
    order = sorted(truth.players, key=lambda p: -np.linalg.norm(p.xy[frame] - cam.C[:2]))
    for p in order:
        foot, ok = cam.project(p.xy[frame]); head, _ = cam.project(np.r_[p.xy[frame], PLAYER_HEIGHT])
        if not ok[0]:
            continue
        h = foot[0, 1] - head[0, 1]
        ax.add_patch(Rectangle((foot[0, 0] - h * .175, head[0, 1]), h * .35, h, color=TEAM_COLORS[p.team], ec="black", lw=.4, zorder=6))
    ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def demo_figure(court: Court, layout: Layout, out: str, hfov: float = HFOV_AVERAGE, seed: int = 1):
    truth = scenario.synthetic_match(court, seconds=60, fps=10, seed=seed)
    est, cals = observe.run(truth, layout, hfov, seed=seed)
    errs = observe.position_errors(truth, est)
    frame = 300
    cams = layout.build(hfov)
    fig = plt.figure(figsize=(15, 9.6))
    axs = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)]
    for ax, (name, cam), cal in zip(axs, cams.items(), cals.values()):
        _render_view(ax, cam, court, truth, frame, f"{name} 화면 (합성) · 보정 기준점 {len(cal.used)}개 · 재투영 {cal.rms_px:.1f}px")
    ax = axs[2]; draw_pitch(ax, court)
    for p, q in zip(truth.players, est.players):
        tail = slice(frame - 30, frame + 1)
        ax.plot(q.xy[tail, 0], q.xy[tail, 1], "-", color=TEAM_COLORS[q.team], lw=1.2, alpha=.6, zorder=11)
        ax.plot(*p.xy[frame], "o", ms=15, mfc="none", mec="white", mew=2, zorder=12)
        ax.plot(*q.xy[frame], "o", ms=10, color=TEAM_COLORS[q.team], mec="black", mew=.6, zorder=13)
    for c in layout.cameras:
        ax.plot(*c.xy, "s", ms=10, color="black", zorder=13)
    ax.set_title(f"2D 지도 (t = {frame / truth.fps:.0f} s, 꼬리 = 최근 3초) · 색 점: 추정 · 흰 원: 실제 · "
                 f"60초 전체 오차 중앙값 {np.median(errs):.2f} m, 상위5% {np.percentile(errs, 95):.2f} m")
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def all_figures(court: Court, twist: float, outdir: str):
    lay = diagonal(court, twist)
    lay.name = f"대각선 코너 · 마주보기에서 왼쪽으로 {twist:g}°"
    spec_figure(court, lay, f"{outdir}/court_spec.png")
    coverage_figure(court, lay, f"{outdir}/coverage.png")
    demo_figure(court, lay, f"{outdir}/demo.png")
