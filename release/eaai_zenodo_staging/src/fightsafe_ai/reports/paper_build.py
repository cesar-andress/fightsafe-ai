"""
LaTeX placeholder fragments for the living paper when local run exports are absent.

The build must stay \textbf{compilable} without fabricating numeric results.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Ablation table (empty body: label preserved for cross-refs)
# ---------------------------------------------------------------------------
ABLATION_TABLE_PLACEHOLDER = r"""
% [TODO] Auto-generated placeholder — no valid ablation_all_runs.csv in this workspace.
% Regenerate: \texttt{fightsafe risk-ablation-all --base-dir runs/case\_studies --output runs/case\_studies/ablation\_summary}
% then: \texttt{python scripts/generate\_paper\_assets.py}
\begin{table}[t]
\centering
\caption{\textbf{TODO:} Aggregate ablation table not regenerated (missing or incomplete \path{ablation\_all\_runs.csv}). Do not invent numbers. Same label kept for cross-references.}
\label{tab:risk-ablation-selected}
\small
\begin{tabular}{@{}p{0.92\linewidth}@{}}
\toprule
\textit{No table body: run the case-study ablation export and \texttt{scripts/generate\_paper\_assets.py} to populate from disk.} \\
\bottomrule
\end{tabular}
\end{table}
""".lstrip()

# ---------------------------------------------------------------------------
# Quantitative summary table
# ---------------------------------------------------------------------------
QUANTITATIVE_TABLE_PLACEHOLDER = r"""
% [TODO] Auto-generated placeholder — quantitative_observations.tex needs ablation CSV + per-run risk_series exports.
\begin{table}[t]
\centering
\caption{\textbf{TODO:} Aggregate behavioral statistics not regenerated (missing \path{risk\_series\_<mode>.csv} under \path{runs/case\_studies/ablation\_summary/<case>/} or aggregate CSV). Do not invent numbers.}
\label{tab:quantitative-ablation-summary}
\small
\begin{tabular}{@{}p{0.92\linewidth}@{}}
\toprule
\textit{Run \texttt{fightsafe risk-ablation-all} then \texttt{python scripts/generate\_paper\_assets.py}, or \texttt{python tools/generate\_quantitative\_observations\_tex.py}.} \\
\bottomrule
\end{tabular}
\end{table}
""".lstrip()

# ---------------------------------------------------------------------------
# BoxingVI batch evaluation tables (copied into paper/tables from batch exports)
# ---------------------------------------------------------------------------
BOXINGVI_RESULTS_ALL_PLACEHOLDER = r"""
% [TODO] Auto-generated placeholder — run BoxingVI batch evaluation and copy
% \path{outputs/evaluation/boxingvi_batch/boxingvi_results_all.tex} or the full_fusion copy under \path{baselines/full_fusion/}.
\begin{table}[t]
\centering
\caption{\textbf{TODO:} BoxingVI batch aggregate table not present on disk. Do not invent numbers. Regenerate with \texttt{python -m fightsafe\_ai.evaluation.boxingvi\_batch\_eval} (and \texttt{python -m fightsafe\_ai.paper.build\_all} to copy into \path{paper/tables/}).}
\label{tab:boxingvi-batch-eval}
\small
\begin{tabular}{@{}p{0.92\linewidth}@{}}
\toprule
\textit{No table body until batch metrics are exported from local runs.} \\
\bottomrule
\end{tabular}
\end{table}
""".lstrip()

BASELINE_COMPARISON_PLACEHOLDER = r"""
% [TODO] Auto-generated placeholder — run batch evaluation with --compare-baselines and copy baseline_comparison.tex.
\begin{table}[t]
\centering
\caption{\textbf{TODO:} Baseline comparison fragment not present. Do not invent numbers. Run \texttt{python -m fightsafe\_ai.evaluation.boxingvi\_batch\_eval --compare-baselines} then copy \path{outputs/evaluation/boxingvi\_batch/baseline\_comparison.tex} to \path{paper/tables/}.}
\label{tab:boxingvi-baselines}
\small
\begin{tabular}{@{}p{0.92\linewidth}@{}}
\toprule
\textit{No table body until baseline comparison export exists locally.} \\
\bottomrule
\end{tabular}
\end{table}
""".lstrip()


def write_boxingvi_results_all_placeholder(paper_dir: Path) -> Path:
    path = paper_dir / "tables" / "boxingvi_results_all.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BOXINGVI_RESULTS_ALL_PLACEHOLDER, encoding="utf-8")
    return path


def write_baseline_comparison_placeholder(paper_dir: Path) -> Path:
    path = paper_dir / "tables" / "baseline_comparison.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BASELINE_COMPARISON_PLACEHOLDER, encoding="utf-8")
    return path


def ensure_boxingvi_paper_table_fragments(paper_dir: Path) -> list[Path]:
    """Write BoxingVI table stubs under ``paper/tables/`` only when files are missing."""
    paper_dir = Path(paper_dir).expanduser().resolve()
    written: list[Path] = []
    p1 = paper_dir / "tables" / "boxingvi_results_all.tex"
    if not p1.is_file():
        written.append(write_boxingvi_results_all_placeholder(paper_dir))
    p2 = paper_dir / "tables" / "baseline_comparison.tex"
    if not p2.is_file():
        written.append(write_baseline_comparison_placeholder(paper_dir))
    return written


def write_ablation_table_placeholder(paper_dir: Path) -> Path:
    path = paper_dir / "tables" / "ablation_selected_results.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ABLATION_TABLE_PLACEHOLDER, encoding="utf-8")
    return path


def write_quantitative_table_placeholder(paper_dir: Path) -> Path:
    path = paper_dir / "tables" / "quantitative_observations.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(QUANTITATIVE_TABLE_PLACEHOLDER, encoding="utf-8")
    return path


__all__ = [
    "BASELINE_COMPARISON_PLACEHOLDER",
    "BOXINGVI_RESULTS_ALL_PLACEHOLDER",
    "ensure_boxingvi_paper_table_fragments",
    "write_ablation_table_placeholder",
    "write_baseline_comparison_placeholder",
    "write_boxingvi_results_all_placeholder",
    "write_quantitative_table_placeholder",
]
