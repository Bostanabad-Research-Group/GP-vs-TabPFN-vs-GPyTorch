"""Render paper logscale comparison .tex table to PDF/PNG via matplotlib (no LaTeX needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from IDETC_create_tables import (  # noqa: E402
    NOISE_LEVELS,
    PAPER_MODEL_DISPLAY,
    PAPER_MODEL_ORDER,
    PAPER_ROW_LAYOUT,
    PROBLEM_DISPLAY,
    _display_center,
    _fmt_metric_value,
    collect_paper_logscale_sidebyside_stats,
)

OUT_DIR = EXPERIMENTS_DIR / "results_logscale_study" / "plots_gpplus_logscale_comparison_paper"


def _best_mask(pairs):
    disp = [_display_center(p[0]) for p in pairs]
    finite = [d for d in disp if d is not None]
    best = min(finite) if finite else None
    return [best is not None and d is not None and d == best for d in disp]


def _plain(v: str) -> str:
    return (
        v.replace(r" $\pm$ ", " ± ")
        .replace("$\\pm$", "±")
        .replace("\\textbf{", "")
        .replace("}", "")
    )


def main() -> None:
    stats = collect_paper_logscale_sidebyside_stats(center="median")
    models = list(PAPER_MODEL_ORDER)
    headers = (
        ["Problem", "N", "Dx", "Noise"]
        + [PAPER_MODEL_DISPLAY[m] for m in models]
        + [PAPER_MODEL_DISPLAY[m] for m in models]
    )

    rows = []
    bold = []
    prev_problem = None
    for problem_key, xdim, n_mult in PAPER_ROW_LAYOUT:
        name = PROBLEM_DISPLAY[problem_key]
        for ni, noise in enumerate(NOISE_LEVELS):
            rrmse = [
                stats.get((problem_key, xdim, n_mult, noise, m), {}).get("RRMSE", (None, None))
                for m in models
            ]
            nis = [
                stats.get((problem_key, xdim, n_mult, noise, m), {}).get("NIS", (None, None))
                for m in models
            ]
            show_problem = prev_problem != problem_key and ni == 0
            show_meta = ni == 0
            left = [
                name if show_problem else "",
                f"{n_mult}Dx" if show_meta else "",
                str(xdim) if show_meta else "",
                noise,
            ]
            vals = [_plain(_fmt_metric_value(*p)) for p in rrmse + nis]
            rows.append(left + vals)
            bold.append([False] * 4 + _best_mask(rrmse) + _best_mask(nis))
            if show_problem:
                prev_problem = problem_key

    fig, ax = plt.subplots(figsize=(18, 3.4))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    n_cols = len(headers)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold", fontsize=6.5)
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)

    for i, brow in enumerate(bold):
        for j, is_best in enumerate(brow):
            cell = table[i + 1, j]
            cell.set_edgecolor("#888888")
            cell.set_linewidth(0.4)
            if is_best:
                cell.set_text_props(weight="bold")
            if j == 10:  # first NIS column — thicker left edge as section split
                cell.set_linewidth(1.0)
                cell.set_edgecolor("black")

    ax.set_title(
        "RRMSE (left six models)  |  NIS (right six) — median ± std; bold = best in row/metric",
        fontsize=9,
        pad=10,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "regression_results_table.pdf"
    png_path = OUT_DIR / "regression_results_table.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
