#!/usr/bin/env python3.12
"""Generate EAAI manuscript figures and tables from the canonical checkpoint only.

Paths are resolved relative to this repository workspace (no absolute machine paths).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Dual layout: Zenodo staging package OR lab paper1 workspace.
if (ROOT / "results" / "run_20260730_005150").is_dir():
    PKG = ROOT
    CANON = PKG / "results" / "run_20260730_005150"
    SOFT = PKG
    ANALYSIS = PKG / "analysis"
    FIG = PKG / "figures"
    TAB = PKG / "tables"
    SUPP = PKG / "supplementary"
else:
    WORKSPACE = ROOT.parent
    CANON = WORKSPACE / "legacy" / "eaai_checkpoint_2026" / "run_20260730_005150"
    SOFT = WORKSPACE / "fightsafe-ai"
    ANALYSIS = ROOT / "analysis"
    FIG = ROOT / "figures"
    TAB = ROOT / "tables"
    SUPP = ROOT / "supplementary"
for d in (FIG, TAB, SUPP, ANALYSIS):
    d.mkdir(exist_ok=True)

C_EQ, C_W, C_MAX = "#2F4F4F", "#1F4E79", "#8B0000"
C_ON, C_OFF = "#1F4E79", "#6B6B6B"
C_EX, C_NV = "#0B6E4F", "#A35C00"


def fmt(x: float, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


RULE_SHORT = {
    "limb_anomaly_instability_high_review": "limb+instability",
    "low_guard_inbound_strike_review": "low\\_guard+strike",
    "low_guard_instability_medium_tendency": "low\\_guard+instability",
    "facing_away_instability_elevated": "facing\\_away+instability",
    "surrender_like_critical_candidate_review": "surrender\\_like",
    "fall_like_and_inactivity_critical_tendency": "fall\\_like+inactivity",
}


def tex_ident(name: str) -> str:
    """Identifier that can break at underscores in narrow columns."""
    return "\\_\\allowbreak{}".join(name.split("_"))


def short_rule(name: str) -> str:
    return RULE_SHORT.get(name, tex_ident(name))


def write_tex_table(
    path: Path,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
    *,
    colspec: str | None = None,
    notes: str | None = None,
    tabularx: bool = False,
) -> None:
    """Emit a table at uniform \\small size (never upscale narrow tables)."""
    cols = colspec if colspec is not None else ("l" + "r" * (len(header) - 1))
    lines = [
        "% Auto-generated from canonical checkpoint; do not hand-edit numbers.",
        "% Source: results/run_20260730_005150 (canonical checkpoint)",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    env = "tabularx" if tabularx else "tabular"
    open_args = f"{{\\linewidth}}{{{cols}}}" if tabularx else f"{{{cols}}}"
    lines.extend(
        [
            f"\\begin{{{env}}}{open_args}",
            "\\toprule",
            " & ".join(header) + " \\\\",
            "\\midrule",
        ]
    )
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.extend([f"\\bottomrule", f"\\end{{{env}}}"])
    if notes:
        lines.append(f"\\\\[2pt]\\footnotesize {notes}")
    lines.extend(["\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_save(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close()


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F5F5F5", ec="#333"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(0.2, 1.4, 1.6, 1.2, "Held-fixed\nfeatures", fc="#E8E8E8")
    box(2.1, 1.4, 1.8, 1.2, "Channels\n$(c_i,\\alpha_i)$", fc="#E8E8E8")
    box(4.2, 2.35, 2.0, 1.0, "Varied risk path\nagg. + interactions", fc="#D9E8F5")
    box(4.2, 0.55, 2.0, 1.0, "Fixed strike path\n(held constant)", fc="#F5E6D3")
    box(6.5, 1.4, 1.6, 1.2, "Combined\ntimeline", fc="#EDEDED")
    box(8.3, 1.4, 1.5, 1.2, "Matcher\nIoU / tol.", fc="#EDEDED")
    for x0, x1, y in [(1.8, 2.1, 2.0), (3.9, 4.2, 2.0), (6.2, 6.5, 2.0), (8.1, 8.3, 2.0)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(5.2, 2.35), xytext=(5.2, 2.6), arrowprops=dict(arrowstyle="-", color="#1F4E79"))
    ax.plot([5.2, 5.2], [1.55, 2.35], color="#1F4E79", lw=1)
    ax.plot([5.2, 6.5], [2.0, 2.0], color="#333", lw=1)
    ax.text(5.2, 3.55, "Fixed vs varied components (canonical protocol)", ha="center", fontsize=10)
    fig_save("fig_architecture")


def main() -> None:
    summary = pd.read_csv(CANON / "experiment_summary.csv")
    pv = pd.read_csv(CANON / "per_video_metrics.csv")
    paired = pd.read_csv(CANON / "paired_comparisons.csv")
    dropout = pd.read_csv(CANON / "dropout_results.csv")
    fails = pd.read_csv(CANON / "failure_counts.csv")
    firings = pd.read_csv(CANON / "interaction_rule_firings.csv")
    meta = pd.read_csv(CANON / "matrices" / "matrix_meta.csv")
    contrib = pd.read_csv(ANALYSIS / "per_video_contribution_weighted.csv")
    lovo = pd.read_csv(ANALYSIS / "lovo_micro_f1_weighted.csv")
    match = pd.read_csv(ANALYSIS / "matching_sensitivity.csv")
    timelines = pd.read_csv(ANALYSIS / "timeline_components.csv")
    drop_vid = pd.read_csv(ANALYSIS / "dropout_per_video_means.csv")

    fusion = yaml.safe_load((SOFT / "configs" / "risk_fusion.yaml").read_text())
    weights = fusion["signal_weights"]
    bands = fusion["level_thresholds"]
    rules = fusion["interaction_rules"]

    # ---- Dataset table ----
    write_tex_table(
        TAB / "tab_dataset.tex",
        "BoxingVI stem statistics (held-fixed matrices).",
        "tab:dataset",
        ["Video", "Frames", "Pos. frames", "Pos. fraction"],
        [
            [r.video_id, str(int(r.n_rows)), str(int(r.n_positive_frames)), fmt(r.pos_frac, 3)]
            for _, r in meta.iterrows()
        ],
        notes="Natural availability $\\alpha{\\equiv}1$ on these matrices; missingness is synthetic only.",
    )

    # ---- Aggregation (equal, weighted, max) ----
    agg = summary[summary.experiment == "aggregation"].copy()
    order = {"equal": 0, "weighted": 1, "max": 2}
    agg["ord"] = agg.method.map(order)
    agg = agg.sort_values("ord")
    write_tex_table(
        TAB / "tab_aggregation.tex",
        "Aggregation on the combined timeline (interactions ON).",
        "tab:aggregation",
        ["Method", "Micro-P", "Micro-R", "Micro-F1", "Macro-F1", "Macro-F1 95\\% CI"],
        [
            [
                r.method,
                fmt(r.micro_precision),
                fmt(r.micro_recall),
                fmt(r.micro_f1),
                fmt(r.macro_f1),
                f"[{fmt(r.bootstrap_macro_f1_lo)}, {fmt(r.bootstrap_macro_f1_hi)}]",
            ]
            for _, r in agg.iterrows()
        ],
        notes="Max is an aggressive single-channel stress test, not a competitive external baseline."
    )

    # ---- Interactions ----
    inter = summary[summary.experiment == "interactions"].copy()
    inter = inter.sort_values("interactions")
    write_tex_table(
        TAB / "tab_interactions.tex",
        "Interaction-rule ablation (weighted aggregation).",
        "tab:interactions",
        ["Int.", "Micro-F1", "Macro-F1", "95\\% CI", "\\#pred", "H/C\\%"],
        [
            [
                "Off" if r.interactions == "off" else "On",
                fmt(r.micro_f1),
                fmt(r.macro_f1),
                f"[{fmt(r.bootstrap_macro_f1_lo)}, {fmt(r.bootstrap_macro_f1_hi)}]",
                fmt(r.mean_n_pred_events, 1),
                fmt(r.mean_pct_high_critical, 2),
            ]
            for _, r in inter.iterrows()
        ],
        notes="H/C\\%: mean HIGH/CRITICAL frame share. Paired $p{\\approx}0.048$ ($n{=}10$).",
    )


    # ---- Firings summary (main compact) ----
    on = firings[firings.interactions == "on"]
    rule_tot = (
        on[~on.rule_name.str.startswith("__")]
        .groupby("rule_name", as_index=False)["fire_count"]
        .sum()
        .sort_values("fire_count", ascending=False)
    )
    write_tex_table(
        TAB / "tab_firings.tex",
        "Interaction-rule firing totals (interactions ON).",
        "tab:firings",
        ["Rule", "Total firings"],
        [
            [short_rule(r.rule_name), str(int(r.fire_count))]
            for _, r in rule_tot.iterrows()
        ]
        + [["surrender\\_like (inactive)", "0"], ["fall\\_like+inactivity", "0"]],
        notes="Frame-level firings; \\texttt{surrender\\_gesture} channel absent.",
    )


    # ---- Dropout ----
    mm = dropout[dropout.video_id == "__MICRO_MACRO__"]
    rows = []
    for p in [0.0, 0.1, 0.3, 0.5]:
        if p == 0.0:
            sub = mm[mm.p == 0.0]
            rows.append(
                [
                    "0.0",
                    "No dropout",
                    fmt(sub.micro_f1.mean()),
                    "---",
                    fmt(sub.micro_precision.mean()),
                    fmt(sub.micro_recall.mean()),
                ]
            )
        else:
            for mode, label in [
                ("explicit_alpha0", "Explicit masking"),
                ("naive_zero", "Naive zero"),
            ]:
                sub = mm[(mm.p == p) & (mm["mode"] == mode)]
                rows.append(
                    [
                        fmt(p, 1),
                        label,
                        fmt(sub.micro_f1.mean()),
                        fmt(sub.micro_f1.std()),
                        fmt(sub.micro_precision.mean()),
                        fmt(sub.micro_recall.mean()),
                    ]
                )
    write_tex_table(
        TAB / "tab_dropout.tex",
        "Synthetic dropout under weighted aggregation.",
        "tab:dropout",
        ["$p$", "Encoding", "Micro-F1", "SD", "Micro-P", "Micro-R"],
        rows,
        notes="Means over seeds for $p{>}0$; natural $\\alpha{\\equiv}1$.",
    )

    # ---- Paired ----
    label_map = {
        "weighted_minus_equal": "Weighted $-$ equal",
        "weighted_minus_max": "Weighted $-$ max",
        "weighted_intON_minus_intOFF": "Interactions ON $-$ OFF",
        "explicit_minus_naive_p0.1": "Explicit $-$ naive ($p{=}0.1$)",
        "explicit_minus_naive_p0.3": "Explicit $-$ naive ($p{=}0.3$)",
        "explicit_minus_naive_p0.5": "Explicit $-$ naive ($p{=}0.5$)",
    }
    write_tex_table(
        TAB / "tab_paired.tex",
        "Paired video-level comparisons.",
        "tab:paired",
        ["Comparison", "$n$", "Mean $\\Delta$F1", "Cliff $\\delta$", "$p$"],
        [
            [
                label_map.get(r.comparison, r.comparison),
                str(int(r.n_videos)),
                fmt(r.mean_delta_f1, 3),
                fmt(r.cliffs_delta, 2),
                fmt(r.permutation_pvalue_twosided, 3),
            ]
            for _, r in paired.iterrows()
        ],
        notes="Permutation two-sided $p$; Cliff's $\\delta$. Dropout contrasts exploratory.",
    )

    # ---- Failures with rates ----
    # Use weighted intON as reference n_pred from summary
    n_pred_ref = {
        "equal_intON": float(agg.loc[agg.method == "equal", "mean_n_pred_events"].iloc[0] * 10),
        "weighted_intON": float(agg.loc[agg.method == "weighted", "mean_n_pred_events"].iloc[0] * 10),
        "max_intON": float(agg.loc[agg.method == "max", "mean_n_pred_events"].iloc[0] * 10),
        "weighted_intOFF": float(
            inter.loc[inter.interactions == "off", "mean_n_pred_events"].iloc[0] * 10
        ),
    }
    # failure_counts is long format? check
    fc = fails.copy()
    if "method" in fc.columns and "failure_type" in fc.columns:
        pivot = fc.pivot_table(index="method", columns="failure_type", values="count", aggfunc="sum").fillna(0)
    else:
        pivot = fc
    # Write rates table for main methods
    fail_rows = []
    for method in ["equal_intON", "weighted_intON", "max_intON", "weighted_intOFF"]:
        if method not in pivot.index:
            continue
        npred = n_pred_ref.get(method, np.nan)
        fp = float(pivot.loc[method].get("false_positive", 0))
        fn = float(pivot.loc[method].get("false_negative", 0))
        frag = float(pivot.loc[method].get("fragmented_detection", 0))
        merged = float(pivot.loc[method].get("merged_distinct_events", 0))
        fail_rows.append(
            [
                method.replace("_intON", "").replace("_intOFF", " (int.\ off)").replace("_", " "),
                str(int(npred)) if npred == npred else "---",
                str(int(fp)),
                str(int(fn)),
                str(int(frag)),
                str(int(merged)),
            ]
        )
    write_tex_table(
        TAB / "tab_failures.tex",
        "Failure taxonomy counts (shared matcher).",
        "tab:failures",
        ["Method", "Approx.\\#pred", "FP", "FN", "Fragment.", "Merged"],
        fail_rows,
        notes="Counts are not normalised across methods because prediction cardinality differs (especially max).",
    )

    # ---- Contribution / LOVO SI ----
    write_tex_table(
        SUPP / "tab_si_video_contribution.tex",
        "Per-video contribution (weighted, interactions ON).",
        "tab:si-video-contribution",
        ["Video", "GT", "TP", "FP", "FN", "Share GT", "Share TP", "F1"],
        [
            [
                r.video_id,
                str(int(r.n_gt)),
                str(int(r.tp)),
                str(int(r.fp)),
                str(int(r.fn)),
                fmt(r.share_gt, 3),
                fmt(r.share_tp, 3),
                fmt(r.f1, 3),
            ]
            for _, r in contrib.iterrows()
        ]
    )
    write_tex_table(
        SUPP / "tab_si_lovo.tex",
        "Leave-one-video-out pooled micro-F1.",
        "tab:si-lovo",
        ["Excluded", "Pooled micro-F1", "TP", "FP", "FN"],
        [
            [r.excluded, fmt(r.micro_f1, 3), str(int(r.tp)), str(int(r.fp)), str(int(r.fn))]
            for _, r in lovo.iterrows()
        ],
        notes="Descriptive sensitivity only; not a new experimental run.",
    )
    write_tex_table(
        SUPP / "tab_si_matching.tex",
        "Post-hoc matching sensitivity.",
        "tab:si-matching",
        ["IoU", "Tolerance (s)", "Micro-F1", "P", "R", "TP", "FP", "FN"],
        [
            [
                fmt(r.iou, 2),
                fmt(r.tol, 1),
                fmt(r.micro_f1, 3),
                fmt(r.precision, 3),
                fmt(r.recall, 3),
                str(int(r.tp)),
                str(int(r.fp)),
                str(int(r.fn)),
            ]
            for _, r in match.iterrows()
        ],
        notes="Does not replace the canonical protocol (IoU$=$0.01, tolerance$=$0.5\\,s)."
    )
    write_tex_table(
        SUPP / "tab_si_timelines.tex",
        "Timeline components (weighted, interactions ON).",
        "tab:si-timelines",
        ["Timeline", "Micro-F1", "P", "R", "TP", "FP", "FN", "\\#pred"],
        [
            [
                {"events": "Combined", "events_risk_only": "Risk-only", "strike_events": "Strike-only"}[
                    r.timeline
                ],
                fmt(r.micro_f1, 3),
                fmt(r.precision, 3),
                fmt(r.recall, 3),
                str(int(r.tp)),
                str(int(r.fp)),
                str(int(r.fn)),
                str(int(r.n_pred)),
            ]
            for _, r in timelines.iterrows()
        ],
        notes="Strike path is held fixed across aggregation methods; risk path is re-aggregated."
    )

    # Channel SI table (compact; source/availability moved to notes)
    channel_rows = []
    related = {r["name"]: r["required_signals"] for r in rules}
    for ch, w in weights.items():
        if ch == "vlm_review_hint":
            continue
        rel = [short_rule(name) for name, reqs in related.items() if ch in reqs]
        channel_rows.append(
            [
                tex_ident(ch),
                fmt(w, 4),
                (", ".join(rel) if rel else "---"),
            ]
        )
    write_tex_table(
        SUPP / "tab_si_channels.tex",
        "Channel inventory and fixed weights.",
        "tab:si-channels",
        ["Channel", "Weight", "Related rules"],
        channel_rows,
        colspec=r">{\raggedright\arraybackslash}p{0.36\linewidth} r >{\raggedright\arraybackslash}X",
        tabularx=True,
        notes="Pose/feature heuristics; natural $\\alpha{\\equiv}1$ on checkpoint matrices.",
    )
    write_tex_table(
        SUPP / "tab_si_rules.tex",
        "Configured interaction rules.",
        "tab:si-rules",
        ["Rule", "Channels", "Boost", "Thr.", "Status"],
        [
            [
                short_rule(r["name"]),
                ", ".join(tex_ident(x) for x in r["required_signals"]),
                fmt(r["boost"], 2),
                fmt(r.get("signal_threshold", fusion.get("interaction_signal_threshold", 0.08)), 2),
                (
                    "inactive"
                    if "surrender_gesture" in r["required_signals"]
                    else ("0 firings" if r["name"].startswith("fall_like") else "fires")
                ),
            ]
            for r in rules
        ],
        colspec=r">{\raggedright\arraybackslash}p{0.30\linewidth} >{\raggedright\arraybackslash}X c c l",
        tabularx=True,
    )
    write_tex_table(
        SUPP / "tab_si_protocol.tex",
        "Fixed temporal and matching parameters.",
        "tab:si-protocol",
        ["Parameter", "Value"],
        [
            ["FPS", "30"],
            ["Risk merge gap (frames)", "2"],
            ["Risk $d_{\\min}$ (s)", "0"],
            ["Combined merge (frames)", "8"],
            ["Matcher IoU", "0.01"],
            ["Matcher tolerance (s)", "0.5"],
            ["Band mins (M/H/C)", f"{bands['medium_min']} / {bands['high_min']} / {bands['critical_min']}"],
            ["Evaluation timeline", "Combined"],
            ["Strike component", "Fixed baseline cache"],
            ["Bootstrap $B$", "1000 (video-level)"],
        ],
        colspec=r"l >{\raggedright\arraybackslash}X",
        tabularx=True,
    )


    # numbers.json for provenance
    numbers = {
        "canon_relative": "results/run_20260730_005150",
        "micro_f1_equal": float(agg.loc[agg.method == "equal", "micro_f1"].iloc[0]),
        "micro_f1_weighted": float(agg.loc[agg.method == "weighted", "micro_f1"].iloc[0]),
        "micro_f1_max": float(agg.loc[agg.method == "max", "micro_f1"].iloc[0]),
        "macro_f1_int_off": float(inter.loc[inter.interactions == "off", "macro_f1"].iloc[0]),
        "macro_f1_int_on": float(inter.loc[inter.interactions == "on", "macro_f1"].iloc[0]),
        "paired_int_p": float(
            paired.loc[paired.comparison == "weighted_intON_minus_intOFF", "permutation_pvalue_twosided"].iloc[0]
        ),
        "v6_share_gt": float(contrib.loc[contrib.video_id == "V6", "share_gt"].iloc[0]),
        "v6_share_tp": float(contrib.loc[contrib.video_id == "V6", "share_tp"].iloc[0]),
        "dropout_p05_explicit": float(
            mm[(mm.p == 0.5) & (mm["mode"] == "explicit_alpha0")].micro_f1.mean()
        ),
        "dropout_p05_naive": float(mm[(mm.p == 0.5) & (mm["mode"] == "naive_zero")].micro_f1.mean()),
        "risk_only_f1": float(timelines.loc[timelines.timeline == "events_risk_only", "micro_f1"].iloc[0]),
        "strike_only_f1": float(timelines.loc[timelines.timeline == "strike_events", "micro_f1"].iloc[0]),
        "combined_f1": float(timelines.loc[timelines.timeline == "events", "micro_f1"].iloc[0]),
    }
    (TAB / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    # ---- Figures ----
    architecture_figure()

    # Aggregation: grouped bars with hatch
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    methods = ["equal", "weighted", "max"]
    x = np.arange(len(methods))
    wbar = 0.35
    micro = [float(agg.loc[agg.method == m, "micro_f1"].iloc[0]) for m in methods]
    macro = [float(agg.loc[agg.method == m, "macro_f1"].iloc[0]) for m in methods]
    lo = [float(agg.loc[agg.method == m, "bootstrap_macro_f1_lo"].iloc[0]) for m in methods]
    hi = [float(agg.loc[agg.method == m, "bootstrap_macro_f1_hi"].iloc[0]) for m in methods]
    yerr = np.vstack([np.array(macro) - np.array(lo), np.array(hi) - np.array(macro)])
    b1 = ax.bar(x - wbar / 2, micro, wbar, label="Micro-F1", color=C_W, hatch="///", edgecolor="black")
    b2 = ax.bar(
        x + wbar / 2,
        macro,
        wbar,
        label="Macro-F1",
        color=C_EQ,
        hatch="\\\\\\",
        edgecolor="black",
        yerr=yerr,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Aggregation on combined timeline (interactions ON)")
    ax.legend(frameon=False)
    ax.axhline(0.5, color="#bbb", lw=0.8, ls=":")
    fig_save("fig_aggregation")

    # Per-video
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    wpv = pv[(pv.experiment == "aggregation") & (pv.interactions == "on")]
    vids = [f"V{i}" for i in range(1, 11)]
    for method, color, marker, hatch in [
        ("equal", C_EQ, "o", "//"),
        ("weighted", C_W, "s", "\\\\"),
        ("max", C_MAX, "^", "xx"),
    ]:
        ys = [
            float(wpv[(wpv.method == method) & (wpv.video_id == v)].f1.iloc[0]) for v in vids
        ]
        ax.plot(vids, ys, marker=marker, color=color, label=method, lw=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-video F1")
    ax.set_title("Per-video heterogeneity ($n{=}10$; V6 dominates pooled counts)")
    ax.legend(frameon=False, ncol=3)
    fig_save("fig_per_video")

    # Interactions: two panels
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    labs = ["Off", "On"]
    macros = [
        float(inter.loc[inter.interactions == "off", "macro_f1"].iloc[0]),
        float(inter.loc[inter.interactions == "on", "macro_f1"].iloc[0]),
    ]
    occ = [
        float(inter.loc[inter.interactions == "off", "mean_pct_high_critical"].iloc[0]),
        float(inter.loc[inter.interactions == "on", "mean_pct_high_critical"].iloc[0]),
    ]
    axes[0].bar(labs, macros, color=[C_OFF, C_ON], edgecolor="black", hatch=["..", "//"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("Interaction ablation")
    axes[1].bar(labs, occ, color=[C_OFF, C_ON], edgecolor="black", hatch=["..", "//"])
    axes[1].set_ylabel("Mean HIGH/CRITICAL frame %")
    axes[1].set_title("Band occupancy")
    fig.tight_layout()
    fig_save("fig_interactions")

    # Dropout: full 0-1 axis + seed spread
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ps = [0.0, 0.1, 0.3, 0.5]
    ex_means, ex_stds, nv_means, nv_stds = [], [], [], []
    for p in ps:
        if p == 0:
            v = float(mm[mm.p == 0].micro_f1.mean())
            ex_means.append(v)
            nv_means.append(v)
            ex_stds.append(0.0)
            nv_stds.append(0.0)
        else:
            e = mm[(mm.p == p) & (mm["mode"] == "explicit_alpha0")].micro_f1
            n = mm[(mm.p == p) & (mm["mode"] == "naive_zero")].micro_f1
            ex_means.append(float(e.mean()))
            nv_means.append(float(n.mean()))
            ex_stds.append(float(e.std()))
            nv_stds.append(float(n.std()))
    ax.errorbar(
        ps,
        ex_means,
        yerr=ex_stds,
        marker="o",
        color=C_EX,
        label="Explicit availability masking",
        capsize=3,
        lw=1.5,
    )
    ax.errorbar(
        ps,
        nv_means,
        yerr=nv_stds,
        marker="s",
        color=C_NV,
        label="Naive zero-valued evidence",
        capsize=3,
        lw=1.5,
        ls="--",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Synthetic dropout probability $p$")
    ax.set_ylabel("Pooled micro-F1")
    ax.set_title("Synthetic missingness only (natural $\\alpha\\equiv 1$)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.02, 0.05, "Axis [0,1]; not a robustness claim", fontsize=7, transform=ax.transAxes)
    fig_save("fig_dropout")

    # Failures figure with rates where possible
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    methods_f = ["equal_intON", "weighted_intON", "max_intON"]
    types = [
        ("false_positive", "false positive"),
        ("false_negative", "false negative"),
        ("fragmented_detection", "fragmented"),
        ("merged_distinct_events", "merged"),
    ]
    x = np.arange(len(methods_f))
    width = 0.2
    for i, (ft, lab) in enumerate(types):
        vals = []
        for m in methods_f:
            if m in pivot.index and ft in pivot.columns:
                vals.append(float(pivot.loc[m, ft]))
            else:
                vals.append(0.0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["equal", "weighted", "max"])
    ax.set_ylabel("Count")
    ax.set_title("Failure counts (denominators differ; see Table)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig_save("fig_failures")

    # Contribution figure
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(contrib.video_id, contrib.share_tp, color=C_W, edgecolor="black", label="Share of pooled TP")
    ax.plot(contrib.video_id, contrib.share_gt, "o--", color=C_MAX, label="Share of pooled GT")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title("V6 dominance in pooled counts (weighted, interactions ON)")
    ax.legend(frameon=False)
    fig_save("fig_video_contribution")

    print("Assets written to", FIG, TAB, SUPP)


if __name__ == "__main__":
    main()
