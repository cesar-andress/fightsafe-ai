#!/usr/bin/env python3
"""
Generate publication-style figures for FightSafe AI (PNG + SVG).

Run from repository root:
    python docs/figures/generate_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


# --- Consistent paper style (muted, white background) ---

DPI = 300
STY: dict = {
    "text": "#2B2D42",
    "grid": "#DDE1E4",
    "arrow": "#4A5568",
    "box_edge": "#5C6770",
    "box_fill": "#F5F6F8",
    "box_fill2": "#EBEEF0",
    "optional_edge": "#6B7C8E",
    "optional_fill": "#F0F3F6",
    "llm_dash": (4, 3),
    "high": "#8B6F47",
    "high_fill": "#D4C4A8",
    "critical": "#6B3A3A",
    "critical_fill": "#C4A5A5",
    "event": "#3D5A73",
    "skeleton": "#2F3A44",
    "arrow_move": "#5A6570",
    "low": "#5A6B78",
    "low_fill": "#E8ECEF",
    "med_fill": "#DDD6C8",
    "med_edge": "#9A8B6E",
    "fs_title": 11,
    "fs_label": 9.5,
    "fs_axis": 9,
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "font.size": STY["fs_label"],
        "axes.edgecolor": STY["text"],
        "axes.labelcolor": STY["text"],
        "xtick.color": STY["text"],
        "ytick.color": STY["text"],
        "axes.titlesize": STY["fs_title"],
    }
)


def out_dir() -> Path:
    p = Path(__file__).resolve().parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_both(name: str, fig: plt.Figure) -> None:
    o = out_dir()
    for ext in ("png", "svg"):
        p = o / f"{name}.{ext}"
        fig.savefig(
            p,
            dpi=DPI,
            format=ext,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Wrote {p}", file=sys.stderr)


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            color=STY["arrow"],
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    small: bool = False,
    face: str | None = None,
    edge: str | None = None,
    linestyle: tuple | str = "solid",
) -> None:
    fs = (STY["fs_label"] - 0.5) if small else STY["fs_label"]
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.08",
        facecolor=face or STY["box_fill"],
        edgecolor=edge or STY["box_edge"],
        linewidth=0.8,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=STY["text"],
    )


def fig_architecture() -> None:
    """Schematic: processing chain + optional LLM + human decision."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="tight")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    w, h = 0.95, 0.55
    y1 = 3.5
    # First row: core pipeline
    row1 = [
        "Input\nvideo",
        "Frame\nextraction",
        "Pose\nestimation",
        "Feature\ncomputation",
    ]
    x0 = 0.25
    xgap = 0.1
    xs: list[float] = []
    x = x0
    for i, lab in enumerate(row1):
        _box(ax, (x, y1), w, h, lab, small=True)
        xs.append(x)
        if i < len(row1) - 1:
            _arrow(ax, x + w, y1 + h * 0.5, x + w + xgap, y1 + h * 0.5)
        x += w + xgap
    # Second row (connected from last of row1)
    y2 = 2.4
    row2 = [
        "Temporal\nanalysis",
        "Risk\nscoring",
        "Event\ndetection",
        "Visualization",
    ]
    # Arrow from "Feature" to row2 first box
    last1 = x0 + 3 * (w + xgap) + w * 0.5
    first2x = x0
    _arrow(ax, last1, y1, first2x + w * 0.5, y2 + h)
    x = x0
    for i, lab in enumerate(row2):
        _box(ax, (x, y2), w, h, lab, small=True)
        if i < len(row2) - 1:
            _arrow(ax, x + w, y2 + h * 0.5, x + w + xgap, y2 + h * 0.5)
        x += w + xgap
    # Optional LLM
    w_ll, h_ll = 1.55, 0.6
    x_ll = 6.35
    y_ll = 1.4
    p = FancyBboxPatch(
        (x_ll, y_ll),
        w_ll,
        h_ll,
        boxstyle="round,pad=0.2,rounding_size=0.1",
        facecolor=STY["optional_fill"],
        edgecolor=STY["optional_edge"],
        linestyle=(0, STY["llm_dash"]),
        linewidth=1.0,
    )
    ax.add_patch(p)
    ax.text(
        x_ll + w_ll / 2,
        y_ll + h_ll / 2,
        "Optional: LLM explanations\n(e.g. Ollama, local)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=STY["text"],
    )
    # from Event detection to LLM
    ev_cx = x0 + 2 * (w + xgap) + w * 0.5
    _arrow(ax, ev_cx, y2, x_ll + 0.25, y_ll + h_ll)
    # from Risk to LLM (optional text features)
    rk_cx = x0 + 1 * (w + xgap) + w * 0.5
    _arrow(ax, rk_cx, y2, x_ll + 0.2, y_ll + h_ll)
    # Human
    y_h = 0.4
    _box(ax, (2.4, y_h), 1.5, 0.55, "Human-in-the-loop\ndecision", small=True)
    # Viz -> human
    vz_cx = x0 + 3 * (w + xgap) + w * 0.5
    _arrow(ax, vz_cx, y2, 2.4 + 0.75, y_h + 0.55)
    # LLM -> human
    _arrow(ax, x_ll + w_ll * 0.5, y_ll, 2.4 + 0.75, y_h + 0.55)
    ax.text(
        0.1,
        4.3,
        "System architecture: offline core pipeline, optional generative layer, and human review.",
        fontsize=STY["fs_title"] + 0.5,
        color=STY["text"],
    )
    save_both("architecture", fig)
    plt.close(fig)


def fig_risk_timeline() -> None:
    t = np.linspace(0, 20, 800)
    rng = np.random.default_rng(7)
    base = 0.2 + 0.15 * np.sin(0.3 * t) + 0.1 * np.sin(1.2 * t)
    s2 = 0.12 * np.exp(-0.3 * (t - 9.0) ** 2)
    s3 = 0.1 * np.exp(-0.4 * (t - 15.5) ** 2) * 0.25
    y = (
        base
        + 0.35 * s2
        + 0.55 * s3
        + 0.02 * rng.standard_normal(800)
        + 0.25 * np.exp(-0.2 * (t - 4.0) ** 2)
    )
    y = np.clip(y, 0, 1.0)
    y = np.convolve(y, np.ones(12) / 12, mode="same")
    y = np.clip(y, 0, 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 2.7), layout="tight")
    ax.set_facecolor("white")
    ax.set_xlabel("Time (s)", fontsize=STY["fs_axis"])
    ax.set_ylabel("Risk score (0–1)", fontsize=STY["fs_axis"])
    ax.axhspan(0.0, 0.55, facecolor=STY["box_fill2"], zorder=0, alpha=0.4)
    ax.axhspan(0.55, 0.75, facecolor=STY["high_fill"], zorder=0, alpha=0.45)
    ax.axhspan(0.75, 1.0, facecolor=STY["critical_fill"], zorder=0, alpha=0.4)
    ax.plot(t, y, color=STY["skeleton"], linewidth=1.3, zorder=3, label="Risk (frame-level)")
    for e in (3.8, 9.0, 15.2):
        ax.axvline(
            e,
            color=STY["event"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )
        yv = float(np.interp(e, t, y))
        ax.scatter(
            e,
            yv,
            s=42,
            color=STY["event"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
    leg = [
        Patch(
            facecolor=STY["box_fill2"],
            edgecolor="none",
            alpha=0.6,
            label="LOW",
        ),
        Patch(
            facecolor=STY["high_fill"],
            edgecolor=STY["high"],
            linewidth=0.3,
            label="HIGH (band 0.55–0.75)",
        ),
        Patch(
            facecolor=STY["critical_fill"],
            edgecolor=STY["critical"],
            linewidth=0.3,
            label="CRITICAL (band ≥0.75)",
        ),
    ]
    ax.legend(handles=leg, loc="upper right", framealpha=0.92, fontsize=7, borderpad=0.4)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_title(
        "Risk signal and detected events (synthetic example)",
        fontsize=STY["fs_title"],
        color=STY["text"],
        pad=8,
    )
    save_both("risk_timeline", fig)
    plt.close(fig)


def fig_risk_levels() -> None:
    """
    Schematic: ordered risk bands with example rule-linked cues (illustration only).
    """
    fig, ax = plt.subplots(figsize=(7.6, 2.85), layout="tight")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    title = "Multi-level risk bands and example interpretable signals (schematic, not normative)"
    ax.text(
        5.0, 2.38, title, ha="center", va="top", fontsize=STY["fs_title"] + 0.3, color=STY["text"]
    )

    # Main progression arrow (increasing concern)
    _arrow(ax, 0.65, 2.05, 9.35, 2.05)
    ax.text(
        5.0,
        2.18,
        "Increasing concern →",
        ha="center",
        va="bottom",
        fontsize=STY["fs_axis"] - 0.5,
        color=STY["arrow"],
        style="italic",
    )

    # Four level boxes
    w, h, yb = 2.05, 0.52, 0.88
    gap = 0.1
    x0 = 0.85
    specs: list[tuple[str, str, str, str, str]] = [
        ("LOW", STY["low_fill"], STY["low"], "Upright stance", "—"),
        ("MEDIUM", STY["med_fill"], STY["med_edge"], "Low guard", "Turning back"),
        ("HIGH", STY["high_fill"], STY["high"], "Falling", "Instability"),
        (
            "CRITICAL",
            STY["critical_fill"],
            STY["critical"],
            "No movement (near ground)",
            "Surrender (tap-out)",
        ),
    ]
    for i, (name, face, edge, s1, s2) in enumerate(specs):
        x = x0 + i * (w + gap)
        rect = FancyBboxPatch(
            (x, yb),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w * 0.5,
            yb + h * 0.5,
            name,
            ha="center",
            va="center",
            fontsize=STY["fs_label"] + 0.5,
            color=STY["text"],
            fontweight="600",
        )
        sig = f"{s1}\n{s2}" if s2 != "—" else s1
        ax.text(
            x + w * 0.5,
            yb - 0.1,
            sig,
            ha="center",
            va="top",
            fontsize=STY["fs_axis"] - 0.2,
            color=STY["text"],
            linespacing=1.35,
        )
        if i < len(specs) - 1:
            xa = x + w + 0.02
            xb = x + w + gap - 0.02
            ym = yb + h * 0.5
            _arrow(ax, xa, ym, xb, ym)

    ax.text(
        5.0,
        0.12,
        "Examples map to project rule names in configuration; thresholds are tunable. Not a clinical model.",
        ha="center",
        va="bottom",
        fontsize=STY["fs_axis"] - 0.4,
        color=STY["text"],
        alpha=0.78,
        style="italic",
    )

    save_both("risk_levels", fig)
    plt.close(fig)


def fig_event_detection() -> None:
    n_frames = 14
    rng = np.random.default_rng(42)
    t = np.arange(n_frames, dtype=float)
    r = 0.15 + 0.1 * np.sin(0.45 * t) + 0.2 * (t / n_frames) ** 1.3
    r[4:6] = 0.7
    r[8:11] = 0.85
    r += 0.02 * rng.standard_normal(n_frames)
    r = np.clip(r, 0, 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 4.0), height_ratios=[0.5, 1, 0.4])
    # Frames strip
    ax0 = axes[0]
    ax0.set_xlim(-0.2, n_frames)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    wbox = 0.6
    for i in range(n_frames):
        c = (
            STY["box_fill"]
            if r[i] < 0.55
            else (STY["high_fill"] if r[i] < 0.75 else STY["critical_fill"])
        )
        ax0.add_patch(
            Rectangle(
                (i + 0.18, 0.1), wbox, 0.8, facecolor=c, edgecolor=STY["box_edge"], linewidth=0.45
            )
        )
        ax0.text(i + 0.5, 0.5, f"F{i}", ha="center", va="center", fontsize=6.5, color=STY["text"])
    ax0.text(
        0,
        0.95,
        "Sampled video frames (color: risk at frame)",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=STY["text"],
    )
    # Risk (lollipop)
    ax1 = axes[1]
    ax1.vlines(t, 0, r, color=STY["skeleton"], linewidth=0.8, alpha=0.75)
    ax1.plot(
        t, r, "o", color=STY["event"], markersize=3.5, markeredgewidth=0.3, markeredgecolor="white"
    )
    ax1.axhspan(0.55, 0.75, color=STY["high_fill"], zorder=0, alpha=0.35)
    ax1.axhspan(0.75, 1.0, color=STY["critical_fill"], zorder=0, alpha=0.35)
    ax1.set_ylabel("Frame risk", fontsize=STY["fs_axis"])
    ax1.set_xlabel("Frame index (sampled)", fontsize=STY["fs_axis"])
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.3, n_frames - 0.2)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    # Events
    ax2 = axes[2]
    ax2.set_xlim(-0.2, n_frames)
    ax2.set_ylim(0, 1.1)
    ax2.axis("off")
    yb = 0.3
    events = [
        (1.0, 3.2, "E1"),
        (3.6, 6.3, "E2"),
        (7.0, 11.5, "E3"),
    ]
    for a, b, name in events:
        rbox = mpatches.FancyBboxPatch(
            (a, yb - 0.1),
            b - a,
            0.38,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=STY["box_fill2"],
            edgecolor=STY["event"],
            linewidth=0.65,
        )
        ax2.add_patch(rbox)
        ax2.text(
            (a + b) / 2,
            yb,
            f"{name}: event interval",
            ha="center",
            va="center",
            fontsize=7.5,
            color=STY["text"],
        )
    ax2.text(
        0,
        0.95,
        "Event segments (aggregation over frames)",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=STY["text"],
    )
    fig.suptitle(
        "From frames to risk curve to event-level intervals",
        fontsize=STY["fs_title"],
        y=0.99,
        color=STY["text"],
    )
    fig.subplots_adjust(hspace=0.35, top=0.88, bottom=0.08, left=0.1, right=0.97)
    save_both("event_detection", fig)
    plt.close(fig)


def fig_pose_features() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 5.0), layout="tight")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    j = {
        "l_sh": (0.4, 0.62),
        "r_sh": (0.6, 0.62),
        "l_elb": (0.34, 0.45),
        "r_elb": (0.66, 0.45),
        "l_hip": (0.44, 0.4),
        "r_hip": (0.56, 0.4),
        "l_kn": (0.42, 0.1),
        "r_kn": (0.58, 0.1),
    }
    hip = ((j["l_hip"][0] + j["r_hip"][0]) / 2, (j["l_hip"][1] + j["r_hip"][1]) / 2)
    for a, b in [
        ("l_sh", "l_elb"),
        ("r_sh", "r_elb"),
        ("l_sh", "r_sh"),
        ("l_sh", "l_hip"),
        ("r_sh", "r_hip"),
        ("l_hip", "r_hip"),
        ("l_hip", "l_kn"),
        ("r_hip", "r_kn"),
    ]:
        ax.plot(
            [j[a][0], j[b][0]],
            [j[a][1], j[b][1]],
            color=STY["skeleton"],
            linewidth=1.8,
            solid_capstyle="round",
        )
    # neck/head
    ax.plot((0.5, 0.5), (0.62, 0.75), color=STY["skeleton"], linewidth=1.8, solid_capstyle="round")
    ax.add_patch(
        mpatches.Circle(
            (0.5, 0.88),
            0.04,
            facecolor=STY["box_fill2"],
            edgecolor=STY["skeleton"],
            linewidth=1,
            zorder=3,
        )
    )
    # Vertical reference
    ax.plot(
        (hip[0], hip[0]), (0.1, 0.96), color=STY["grid"], linewidth=0.4, linestyle="--", zorder=0
    )
    # Torso angle
    ax.add_patch(
        Arc(hip, 0.3, 0.3, angle=0, theta1=80, theta2=100, color=STY["event"], linewidth=0.8)
    )
    ax.text(0.2, 0.42, "Torso vs. vertical (θ)", fontsize=7.5, color=STY["text"])
    # Hip
    ax.add_patch(
        mpatches.Circle(
            hip, 0.02, facecolor=STY["event"], edgecolor=STY["skeleton"], linewidth=0.6, zorder=4
        )
    )
    ax.text(
        0.6,
        0.36,
        "Hip (center / COM proxy)",
        fontsize=7.2,
        color=STY["text"],
        ha="left",
        va="center",
    )
    # Head
    ax.text(0.56, 0.92, "Head position", fontsize=7.2, color=STY["text"], ha="left")
    ax.add_patch(
        FancyArrowPatch(
            (0.55, 0.88),
            (0.7, 0.82),
            arrowstyle="-|>",
            color=STY["arrow_move"],
            linewidth=0.7,
            mutation_scale=7,
        )
    )
    ax.text(
        0.72,
        0.8,
        "instantaneous\ndisplacement",
        fontsize=6.5,
        color=STY["text"],
        ha="left",
        va="center",
    )
    ax.set_title(
        "Pose features: joint geometry and kinematic cues (schematic)",
        fontsize=STY["fs_title"],
        color=STY["text"],
        pad=6,
    )
    save_both("pose_features", fig)
    plt.close(fig)


def fig_framework_architecture() -> None:
    """
    Extended system diagram: pose, tracking, action, anomaly, fusion, HCI, LLM.
    (Publication asset: ``framework_architecture``.)
    """
    fig, ax = plt.subplots(figsize=(10.2, 5.6), layout="tight")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    w, h = 1.05, 0.5
    gap = 0.12
    tx = STY["text"]
    y_title = 5.15
    ax.text(
        5.1,
        y_title,
        "FightSafe AI: extended processing architecture (schematic)",
        ha="center",
        va="top",
        fontsize=STY["fs_title"] + 0.4,
        color=tx,
    )
    y = 4.45
    row_a = [
        "Input\nvideo",
        "Frame\nextraction",
        "Pose\nestimation",
    ]
    x = 0.5
    for i, lab in enumerate(row_a):
        _box(ax, (x, y), w, h, lab, small=True)
        if i < len(row_a) - 1:
            _arrow(ax, x + w, y + h * 0.5, x + w + gap, y + h * 0.5)
        x += w + gap
    y_tr = 3.6
    _box(ax, (0.5 + w + gap + w + gap, y_tr), 1.15, h, "Fighter\ntracking", small=True)
    _arrow(
        ax,
        0.5 + 2 * (w + gap) + w * 0.5,
        y,
        0.5 + 2 * (w + gap) + 0.6,
        y_tr + h,
    )
    y_f = 2.75
    w_f = 1.12
    _box(ax, (0.45, y_f), w_f, h, "Biomechanical\nfeatures", small=True)
    _box(ax, (1.7, y_f), w_f, h, "Action\nlayer", small=True)
    _box(ax, (2.95, y_f), w_f, h, "Anomaly\nlayer", small=True)
    _arrow(ax, 0.5 + w * 0.5, y - 0.02, 0.45 + w_f * 0.5, y_f + h)
    _arrow(ax, 0.5 + (w + gap) + w * 0.5, y - 0.02, 1.7 + w_f * 0.5, y_f + h)
    _arrow(ax, 0.5 + 2 * (w + gap) + w * 0.5, y - 0.02, 2.95 + w_f * 0.5, y_f + h)
    _arrow(ax, 1.25 + 1.15, y_tr + h * 0.5, 2.2, y_f + h * 0.5)
    y_fu = 1.85
    wf, hf = 2.35, 0.55
    xf = 1.1
    _box(
        ax,
        (xf, y_fu),
        wf,
        hf,
        "Risk fusion\n(interpretable rules, multi-signal)",
        small=True,
    )
    for x0, cx in ((0.45, xf + 0.35), (1.7, xf + 0.95), (2.95, xf + 1.6)):
        _arrow(ax, x0 + w_f * 0.5, y_f, cx, y_fu + hf)
    y_hci = 0.95
    w_h, h_h = 2.1, 0.52
    _box(
        ax,
        (1.2, y_hci),
        w_h,
        h_h,
        "Human-in-the-loop\ndecision-support (HCI, referee alerts)",
        small=True,
    )
    _arrow(ax, xf + wf * 0.5, y_fu, 1.2 + w_h * 0.5, y_hci + h_h)
    w_l, h_l, x_l, y_l = 1.6, 0.52, 6.45, 1.55
    p = FancyBboxPatch(
        (x_l, y_l),
        w_l,
        h_l,
        boxstyle="round,pad=0.2,rounding_size=0.1",
        facecolor=STY["optional_fill"],
        edgecolor=STY["optional_edge"],
        linestyle=(0, STY["llm_dash"]),
        linewidth=1.0,
    )
    ax.add_patch(p)
    ax.text(
        x_l + w_l / 2,
        y_l + h_l / 2,
        "Optional: LLM explainability\n(post-hoc; does not set risk)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=tx,
    )
    _arrow(ax, xf + wf, y_fu + hf * 0.65, x_l, y_l + h_l * 0.5)
    _arrow(
        ax,
        x_l + w_l * 0.5,
        y_l,
        1.2 + w_h * 0.5,
        y_hci + h_h,
    )
    ax.text(
        5.0,
        0.2,
        "Dashed: optional. Pipeline outputs are advisory; not autonomous officiation.",
        ha="center",
        fontsize=STY["fs_axis"] - 0.5,
        color=tx,
        alpha=0.82,
        style="italic",
    )
    save_both("framework_architecture", fig)
    plt.close(fig)


def fig_risk_fusion_model() -> None:
    """
    How heterogeneous signals map through fusion to risk bands.
    (Publication asset: ``risk_fusion_model``.)
    """
    fig, ax = plt.subplots(figsize=(9.4, 3.2), layout="tight")
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.text(
        4.7,
        2.9,
        "Multi-signal fusion to ordered risk level (illustration)",
        ha="center",
        va="top",
        fontsize=STY["fs_title"] + 0.3,
        color=STY["text"],
    )
    w, h = 1.25, 0.45
    y0 = 1.55
    x0 = 0.4
    inputs = [
        "Biomech\nprobes",
        "Action\nsignals",
        "Anomaly\ndetectors",
        "Tracking\ncues",
    ]
    for k, lab in enumerate(inputs):
        _box(ax, (x0 + k * (w + 0.1), y0 + 0.5), w, h, lab, small=True)
    wfu, hfu = 1.65, 0.85
    xfu = 2.0
    yfu = 0.85
    _box(
        ax,
        (xfu, yfu),
        wfu,
        hfu,
        "Aggregation +\nthresholds\n(rules, weights\nfrom config)",
        small=True,
    )
    y_in = y0 + 0.5
    for k in range(4):
        _arrow(
            ax,
            x0 + k * (w + 0.1) + w * 0.5,
            y_in,
            xfu + wfu * 0.5,
            yfu + hfu,
        )
    specs = [
        ("LOW", STY["low_fill"], STY["low"]),
        ("MEDIUM", STY["med_fill"], STY["med_edge"]),
        ("HIGH", STY["high_fill"], STY["high"]),
        ("CRITICAL", STY["critical_fill"], STY["critical"]),
    ]
    bw, bh, gx = 1.35, 0.42, 0.1
    xb = 5.0
    yb = 1.45
    for i, (name, face, edge) in enumerate(specs):
        x = xb + i * (bw + gx)
        b = FancyBboxPatch(
            (x, yb),
            bw,
            bh,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
        ax.add_patch(b)
        ax.text(
            x + bw / 2,
            yb + bh / 2,
            name,
            ha="center",
            va="center",
            fontsize=STY["fs_label"] + 0.3,
            color=STY["text"],
            fontweight="600",
        )
        if i < len(specs) - 1:
            _arrow(
                ax,
                x + bw * 0.7,
                yb + bh * 0.5,
                x + bw + gx - 0.12,
                yb + bh * 0.5,
            )
    _arrow(ax, xfu + wfu, yfu + hfu * 0.5, xb - 0.15, yb + bh * 0.5)
    ax.text(
        4.7,
        0.22,
        "Ordering is fixed (LOW<…<CRITICAL); cut-offs are tunable research parameters.",
        ha="center",
        fontsize=STY["fs_axis"] - 0.4,
        color=STY["text"],
        style="italic",
        alpha=0.78,
    )
    save_both("risk_fusion_model", fig)
    plt.close(fig)


def fig_human_in_the_loop_alerts() -> None:
    """
    AI suggests review; human referee retains authority.
    (Publication asset: ``human_in_the_loop_alerts``.)
    """
    fig, ax = plt.subplots(figsize=(8.0, 3.4), layout="tight")
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.text(
        4.0,
        2.95,
        "Alerting the referee: recommendation, not a replacement",
        ha="center",
        va="top",
        fontsize=STY["fs_title"] + 0.4,
        color=STY["text"],
    )
    w, h = 1.7, 0.75
    _box(ax, (0.5, 1.5), w, h, "Algorithmic\nsafety & fusion\noutput", small=True)
    w2, h2 = 1.5, 0.55
    p = FancyBboxPatch(
        (2.6, 1.65),
        w2,
        h2,
        boxstyle="round,pad=0.15,rounding_size=0.1",
        facecolor=STY["box_fill2"],
        edgecolor=STY["event"],
        linewidth=0.9,
    )
    ax.add_patch(p)
    ax.text(
        2.6 + w2 / 2,
        1.65 + h2 / 2,
        "Referee-oriented\nalerts (INFO…STOP_REC.)",
        ha="center",
        va="center",
        fontsize=9.0,
        color=STY["text"],
    )
    w3, h3 = 1.85, 0.7
    _box(
        ax, (4.7, 1.52), w3, h3, "Human referee /\nsafety review", small=True, face=STY["low_fill"]
    )
    _arrow(ax, 0.5 + w, 1.5 + h * 0.5, 2.6, 1.65 + h2 * 0.5)
    _arrow(ax, 2.6 + w2, 1.65 + h2 * 0.5, 4.7, 1.52 + h3 * 0.5)
    ax.text(
        2.0,
        0.6,
        "Not medical diagnosis. Not an autonomous stoppage command.",
        ha="center",
        va="center",
        fontsize=STY["fs_axis"] + 0.2,
        color=STY["critical"],
        fontweight="600",
    )
    ax.text(
        4.0,
        0.25,
        "Dotted boundary: the system may highlight time spans and cues; the official applies rules and judgment.",
        ha="center",
        fontsize=STY["fs_axis"] - 0.5,
        color=STY["text"],
        alpha=0.8,
        style="italic",
    )
    ax.add_patch(
        mpatches.Circle(
            (6.85, 0.6),
            0.2,
            facecolor="white",
            edgecolor=STY["critical"],
            linewidth=1.1,
        )
    )
    ax.text(6.85, 0.6, "×", ha="center", va="center", fontsize=14, color=STY["critical"])
    ax.text(
        7.45,
        0.6,
        "autonomous\nfight outcome",
        ha="center",
        va="center",
        fontsize=7,
        color=STY["text"],
    )
    save_both("human_in_the_loop_alerts", fig)
    plt.close(fig)


def fig_combat_safety_signal_taxonomy() -> None:
    """
    Grouping signals: pre-critical, critical, contextual.
    (Publication asset: ``combat_safety_signal_taxonomy``.)
    """
    fig, ax = plt.subplots(figsize=(9.6, 3.8), layout="tight")
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    ax.text(
        4.8,
        3.45,
        "Combat-safety signal taxonomy (research labels; not clinical)",
        ha="center",
        va="top",
        fontsize=STY["fs_title"] + 0.3,
        color=STY["text"],
    )
    cols: list[tuple[str, str, str, str]] = [
        (
            "Pre-critical / early",
            "Proxies for posture, guard, and\nreduced activity before a severe band.",
            "Low guard, turned back,\ninactivity, limb-motion cues.",
            STY["low_fill"],
        ),
        (
            "Critical (high-urgency bands)",
            "Severe interpretive fusion, falls,\nsurrender-like gestures, STOP_RECOMMENDED\nmapping in HCI (review-oriented).",
            "Max fusion, fall/surrender\nheuristics, high instability.",
            STY["high_fill"],
        ),
        (
            "Contextual",
            "Affects interpretation and calibration;\nof themselves not a verdict.",
            "Camera & crop, round timing,\ndataset/clip metadata, visibility.",
            STY["med_fill"],
        ),
    ]
    w, h, gap = 2.7, 2.45, 0.35
    x = 0.4
    for i, (title, desc, ex, face) in enumerate(cols):
        bx = x + i * (w + gap)
        b = FancyBboxPatch(
            (bx, 0.4),
            w,
            h,
            boxstyle="round,pad=0.1,rounding_size=0.12",
            facecolor=face,
            edgecolor=STY["box_edge"],
            linewidth=0.8,
        )
        ax.add_patch(b)
        top = 0.4 + h
        ax.text(
            bx + w * 0.5,
            top + 0.1,
            title,
            ha="center",
            va="bottom",
            fontsize=STY["fs_label"] + 0.5,
            color=STY["text"],
            fontweight="600",
        )
        ax.text(
            bx + 0.12,
            top - 0.2,
            desc,
            ha="left",
            va="top",
            fontsize=7.5,
            color=STY["text"],
            linespacing=1.35,
        )
        ex_one = " ".join(line.strip() for line in ex.split("\n"))
        ax.text(
            bx + 0.12,
            0.55,
            "Examples: " + ex_one,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=STY["text"],
            style="italic",
        )
    save_both("combat_safety_signal_taxonomy", fig)
    plt.close(fig)


def fig_evaluation_protocol() -> None:
    """
    Offline evaluation: data, labels, metrics, ablations.
    (Publication asset: ``evaluation_protocol``.)
    """
    fig, ax = plt.subplots(figsize=(9.0, 1.9), layout="tight")
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    ax.text(
        4.5,
        1.62,
        "Offline evaluation protocol (reproducible research)",
        ha="center",
        va="top",
        fontsize=STY["fs_title"] + 0.35,
        color=STY["text"],
    )
    n = 4
    margin = 0.35
    gap = 0.18
    w = (8.0 - 2 * margin - gap * (n - 1)) / n
    y0, h = 0.38, 0.52
    steps = [
        "Dataset / clips\n& splits",
        "Reference\nannotations",
        "Frame & event\nmetrics",
        "Ablation\nscenarios",
    ]
    for i, lab in enumerate(steps):
        x = margin + i * (w + gap)
        _box(ax, (x, y0), w, h, lab, small=True)
        if i < n - 1:
            _arrow(ax, x + w, y0 + h * 0.5, x + w + gap, y0 + h * 0.5)
    ax.text(
        4.5,
        0.1,
        "Time alignment, IoU threshold, and prevalence should be reported alongside any score.",
        ha="center",
        va="bottom",
        fontsize=STY["fs_axis"] - 0.4,
        color=STY["text"],
        style="italic",
        alpha=0.78,
    )
    save_both("evaluation_protocol", fig)
    plt.close(fig)


def fig_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 1.25), layout="tight")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    steps = ["Raw video", "Frames", "Keypoints", "Features", "Risk", "Events", "Report"]
    n = len(steps)
    gap = 0.1
    margin = 0.2
    avail = 8.0 - 2 * margin
    w = (avail - gap * (n - 1)) / n
    y0, h = 0.38, 0.5
    x0 = 1.0
    for i, lab in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, (x, y0), w, h, lab, small=True)
        if i < n - 1:
            _arrow(ax, x + w, y0 + h * 0.5, x + w + gap, y0 + h * 0.5)
    ax.text(
        0.15,
        1.0,
        "End-to-end data flow (logical pipeline)",
        fontsize=STY["fs_title"] + 0.5,
        color=STY["text"],
        va="top",
    )
    save_both("pipeline_flow", fig)
    plt.close(fig)


def main() -> int:
    print("Generating FightSafe AI paper figures (PNG + SVG)…", file=sys.stderr)
    out_dir()  # ensure dir exists
    fig_architecture()
    fig_framework_architecture()
    fig_risk_timeline()
    fig_risk_levels()
    fig_risk_fusion_model()
    fig_event_detection()
    fig_pose_features()
    fig_pipeline_flow()
    fig_human_in_the_loop_alerts()
    fig_combat_safety_signal_taxonomy()
    fig_evaluation_protocol()
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
