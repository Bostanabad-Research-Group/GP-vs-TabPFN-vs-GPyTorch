"""
Generate RRMSE / NIS / NCRPS violin PDFs comparing original vs log-scale models.

Same format as IDETC_plot_gpplus_logscale_comparison.py (stacked N rows, per-metric PDFs),
but without GPyTorch columns.

Default data root: results_logscale_study/
  Original:   Gaussian, PE, LOO, PFN 2.5, PFN 2.0
  Log-scale:  Gaussian (log), PE (log), LOO (log)

Default problems: zakharov, buckling

Outputs (under results_logscale_study/):
  plots_gpplus_logscale_comparison2/       -> no LOO columns
  plots_gpplus_logscale_comparison_LOO2/   -> with LOO + LOO (log)

Run from experiments/:

  python experimental_utils/IDETC_plot_gpplus_logscale_comparison2.py
  python experimental_utils/IDETC_plot_gpplus_logscale_comparison2.py --only no_loo
  python experimental_utils/IDETC_plot_gpplus_logscale_comparison2.py --only loo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_violin_metrics import (  # noqa: E402
    collect_per_run_rows,
    parse_extra_named_gp_entries,
    remove_outliers_iqr,
)

DEFAULT_ROOT = EXPERIMENTS_DIR / "results_logscale_study"
DEFAULT_PROBLEMS = ("zakharov", "buckling")

# Subfolder names under DEFAULT_ROOT
DIR_GP = "10_runs_logging_full_Gaussian_orig"
DIR_PE = "10_runs_logging_full_PE_orig"
DIR_LOO = "10_runs_logging_full_Gaussian_LOO_orig"
DIR_GP_LOG = "10_runs_logging_full_Gaussian_logscale"
DIR_PE_LOG = "10_runs_logging_full_PE_logscale"
DIR_LOO_LOG = "10_runs_logging_full_Gaussian_LOO_logscale"
DIR_PFN25 = "10_runs_PFN_V2.5"
DIR_PFN20 = "10_runs_PFN_V2.0"

NOISE_COLOR_MAP = {
    "0.0": "#1f77b4",
    "0.002": "#ff7f0e",
    "0.08": "#2ca02c",
}
EXTRA_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

DISPLAY_NAMES = {
    "gpplus": "GP+",
    "GP+ (PE)": "GP+ (PE)",
    "GP+ (LOO)": "GP+ (LOO)",
    "GP+ (log)": "GP+ (log)",
    "GP+ (PE log)": "GP+ (PE log)",
    "GP+ (LOO log)": "GP+ (LOO log)",
    "tabpfn_v2.5": "PFN 2.5",
    "tabpfn_v2": "PFN 2.0",
}

# Filename stem used in original Overleaf assets (dixonprice, not dixon_price).
PROBLEM_FILE_STEM = {
    "dixon_price": "dixonprice",
    "dixonprice": "dixonprice",
}


def _problem_stem(problem: str) -> str:
    return PROBLEM_FILE_STEM.get(problem.lower(), problem.lower())


def _dim_sort_key(d: str) -> float:
    try:
        return float(d.rstrip("D")) if isinstance(d, str) and d.endswith("D") else float(d)
    except ValueError:
        return 0.0


def _display_label(model: str) -> str:
    """Prefer multiline labels for parenthetical suffixes so wide axes stay readable."""
    label = DISPLAY_NAMES.get(model, model)
    if " (" in label:
        return label.replace(" (", "\n(", 1)
    return label


def _plot_metric_axis(
    ax,
    dim_df: pd.DataFrame,
    metric: str,
    models: List[str],
    noise_levels: List[str],
    remove_outliers: bool,
) -> None:
    width = 0.8
    inner_width = width / max(len(noise_levels), 1) if len(noise_levels) > 1 else width * 0.7

    data_groups: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []

    for mi, model in enumerate(models):
        center = mi + 1
        start = center - width / 2 + inner_width / 2
        for ni, noise in enumerate(noise_levels):
            arr = (
                dim_df.loc[
                    (dim_df["model_family"] == model) & (dim_df["noise_label"] == noise),
                    metric,
                ]
                .dropna()
                .astype(float)
                .values
            )
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            if remove_outliers:
                arr = remove_outliers_iqr(arr)
            if arr.size == 0:
                continue
            xpos = start + ni * inner_width
            data_groups.append(arr)
            positions.append(xpos)
            colors.append(NOISE_COLOR_MAP.get(noise, EXTRA_COLORS[ni % len(EXTRA_COLORS)]))

    if data_groups:
        parts = ax.violinplot(
            data_groups,
            positions=positions,
            widths=inner_width * 0.9,
            showmeans=False,
            showmedians=False,
            showextrema=True,
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.7)

        for xpos, arr in zip(positions, data_groups):
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(mean_v, xpos - inner_width * 0.5, xpos + inner_width * 0.5, colors="#ff69b4", linewidth=2)
            ax.hlines(
                med_v,
                xpos - inner_width * 0.5,
                xpos + inner_width * 0.5,
                colors="black",
                linewidth=2,
                linestyles="--",
            )

    ax.set_xticks([mi + 1 for mi in range(len(models))])
    # Slightly smaller tick labels when many models share the axis.
    tick_fs = 18 if len(models) >= 7 else 22
    ax.set_xticklabels([_display_label(m) for m in models], fontsize=tick_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.locator_params(axis="y", nbins=4)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4, which="major", zorder=0)
    for boundary in [mi + 0.5 for mi in range(len(models) + 1)]:
        ax.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=0)

    # Thicker axes frame; keep below annotation overlays.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color("black")
        spine.set_zorder(1)
    ax.tick_params(width=1.8, length=5)


METRICS = ("RRMSE", "NIS", "NCRPS")


def _add_n_label(ax, dim_number: str) -> None:
    """Opaque N= label box, inset so it sits above grid and clear of the axes frame."""
    txt = ax.text(
        0.04,
        0.88,
        f"$N = {dim_number}D_x$",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=200,
        clip_on=False,
        color="black",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 2.0,
            "alpha": 1.0,
        },
    )
    bp = txt.get_bbox_patch()
    if bp is not None:
        bp.set_zorder(199)
        bp.set_facecolor("white")
        bp.set_alpha(1.0)
        bp.set_edgecolor("black")
        bp.set_linewidth(2.0)
        bp.set_fill(True)


def make_separate_metric_plots(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: List[str],
    *,
    metrics: tuple[str, ...] = METRICS,
    remove_outliers: bool = False,
    legend_on_problem: str = "buckling",
    fig_width_per_model: float = 2.4,
    fig_width_min: float = 14.0,
) -> None:
    """One PDF per (problem, xdim, metric), stacked N rows — matches orig_ref style."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    # Force visible axes frames regardless of seaborn theme.
    plt.rcParams.update({
        "axes.linewidth": 2.5,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
    })

    available = set(df["model_family"].dropna().unique())
    model_order = [m for m in model_order if m in available]
    if not model_order:
        raise SystemExit("No requested models present in collected data.")

    missing_cols = [m for m in metrics if m not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing metric columns in collected data: {missing_cols}")

    non_m2ax = df[df["problem"] != "m2ax"]
    n_saved = 0
    for (problem, xdim), prob_xdim_df in non_m2ax.groupby(["problem", "xdim"]):
        if xdim == "" or xdim == "all":
            continue

        dims = sorted(
            [d for d in prob_xdim_df["dim"].unique() if d not in ("", "all")],
            key=_dim_sort_key,
        )
        noise_levels = sorted(
            [n for n in prob_xdim_df["noise_label"].unique() if n != "unknown"],
            key=lambda v: float(v),
        )
        if not dims or not noise_levels:
            continue

        n_rows = len(dims)
        # Wider than the 5-model paper figures so original + log-scale columns fit.
        fig_width = max(fig_width_min, fig_width_per_model * len(model_order))
        fig_height = 3 * n_rows
        figs: dict[str, plt.Figure] = {}
        axes_by_metric: dict[str, np.ndarray] = {}
        for metric in metrics:
            fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
            if n_rows == 1:
                axes = np.array([axes])
            figs[metric] = fig
            axes_by_metric[metric] = axes

        show_legend = noise_levels and legend_on_problem.lower() in str(problem).lower()
        legend_elements = None
        if show_legend:
            legend_elements = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc=NOISE_COLOR_MAP.get(noise, EXTRA_COLORS[i % len(EXTRA_COLORS)]),
                    ec="black",
                    label=f"Noise {noise}",
                )
                for i, noise in enumerate(noise_levels)
            ]
            legend_elements.append(plt.Line2D([0, 1], [0, 1], color="#ff69b4", linewidth=2, label="Mean"))
            legend_elements.append(
                plt.Line2D([0, 1], [0, 1], color="black", linewidth=2, linestyle="--", label="Median")
            )

        for row_idx, dim in enumerate(dims):
            dim_df = prob_xdim_df[prob_xdim_df["dim"] == dim]
            models = [m for m in model_order if m in set(dim_df["model_family"].unique())]
            if not models:
                continue

            dim_number = dim.rstrip("D") if dim.endswith("D") else dim
            for metric in metrics:
                ax = axes_by_metric[metric][row_idx]
                _plot_metric_axis(ax, dim_df, metric, models, noise_levels, remove_outliers)
                _add_n_label(ax, dim_number)
                # Legend on upper panel only — compact stack flush to top-right.
                if show_legend and legend_elements is not None and row_idx == 0:
                    leg = ax.legend(
                        handles=legend_elements,
                        loc="upper right",
                        bbox_to_anchor=(0.995, 0.98),
                        borderaxespad=0.0,
                        ncol=1,
                        fontsize=11,
                        frameon=True,
                        fancybox=False,
                        edgecolor="black",
                        facecolor="white",
                        framealpha=1.0,
                        borderpad=0.4,
                        handlelength=1.5,
                        labelspacing=0.35,
                    )
                    leg.set_zorder(200)
                    frame = leg.get_frame()
                    frame.set_alpha(1.0)
                    frame.set_facecolor("white")
                    frame.set_edgecolor("black")
                    frame.set_linewidth(1.5)

        stem = _problem_stem(str(problem))
        suffix = "_no_outliers" if remove_outliers else ""
        for metric, fig in figs.items():
            # Re-assert spine thickness after layout (seaborn can thin them).
            for ax in fig.axes:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_color("black")
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            for ax in fig.axes:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.5)
                    spine.set_color("black")
                    spine.set_zorder(2)
            outfile = out_dir / f"{stem}_xdim{xdim}_gpplus_comparison_{metric}{suffix}.pdf"
            fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {outfile}")
        n_saved += 1

    if n_saved == 0:
        print(f"Warning: no plots written under {out_dir}")


# Backwards-compatible alias
def make_separate_rrmse_nis_plots(*args, **kwargs):
    return make_separate_metric_plots(*args, **kwargs)


def collect_logscale_df(
    *,
    gp_dir: Path,
    pe_dir: Path,
    loo_dir: Optional[Path],
    gp_log_dir: Path,
    pe_log_dir: Path,
    loo_log_dir: Optional[Path],
    pfn25_dir: Path,
    pfn20_dir: Path,
    include_loo: bool,
    problems: Optional[set[str]] = None,
    noise_levels: Optional[List[str]] = None,
) -> pd.DataFrame:
    extras = [
        f"GP+ (PE)={pe_dir}",
        f"GP+ (log)={gp_log_dir}",
        f"GP+ (PE log)={pe_log_dir}",
    ]
    if include_loo and loo_dir is not None:
        extras.append(f"GP+ (LOO)={loo_dir}")
    if include_loo and loo_log_dir is not None:
        extras.append(f"GP+ (LOO log)={loo_log_dir}")
    extra_named = parse_extra_named_gp_entries(extras)
    # Pass gp_dir as gpytorch_dir so collect_per_run_rows skips a separate GPyTorch tree.
    return collect_per_run_rows(
        gp_dir,
        gp_dir,
        None,
        pfn25_dir,
        pfn20_dir,
        problems,
        extra_named_gp=extra_named,
        noise_levels=noise_levels or ["0.002", "0.08"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orig vs log-scale RRMSE/NIS/NCRPS violin PDFs (zakharov, buckling)."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="results_logscale_study root")
    parser.add_argument("--gp_dir", type=Path, default=None)
    parser.add_argument("--pe_dir", type=Path, default=None)
    parser.add_argument("--loo_dir", type=Path, default=None)
    parser.add_argument("--gp_log_dir", type=Path, default=None)
    parser.add_argument("--pe_log_dir", type=Path, default=None)
    parser.add_argument("--loo_log_dir", type=Path, default=None)
    parser.add_argument("--tabpfn_v25_dir", type=Path, default=None)
    parser.add_argument("--tabpfn_v2_dir", type=Path, default=None)
    parser.add_argument("--out_no_loo", type=Path, default=None)
    parser.add_argument("--out_loo", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=("both", "no_loo", "loo"),
        default="both",
        help="Which plot sets to write (default: both).",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=list(DEFAULT_PROBLEMS),
        help=f"Problems to include (default: {' '.join(DEFAULT_PROBLEMS)}).",
    )
    parser.add_argument("--noise_levels", nargs="*", default=["0.002", "0.08"])
    parser.add_argument("--remove_outliers", action="store_true")
    args = parser.parse_args()

    root = args.root
    gp_dir = args.gp_dir or (root / DIR_GP)
    pe_dir = args.pe_dir or (root / DIR_PE)
    loo_dir = args.loo_dir or (root / DIR_LOO)
    gp_log_dir = args.gp_log_dir or (root / DIR_GP_LOG)
    pe_log_dir = args.pe_log_dir or (root / DIR_PE_LOG)
    loo_log_dir = args.loo_log_dir or (root / DIR_LOO_LOG)
    pfn25 = args.tabpfn_v25_dir or (root / DIR_PFN25)
    pfn20 = args.tabpfn_v2_dir or (root / DIR_PFN20)
    out_no_loo = args.out_no_loo or (root / "plots_gpplus_logscale_comparison2")
    out_loo = args.out_loo or (root / "plots_gpplus_logscale_comparison_LOO2")
    problems = set(args.problems) if args.problems else set(DEFAULT_PROBLEMS)

    # Pair each GP+ variant with its log-scale counterpart (no GPyTorch).
    order_no_loo = [
        "gpplus",
        "GP+ (log)",
        "GP+ (PE)",
        "GP+ (PE log)",
        "tabpfn_v2.5",
        "tabpfn_v2",
    ]
    order_loo = [
        "gpplus",
        "GP+ (log)",
        "GP+ (PE)",
        "GP+ (PE log)",
        "GP+ (LOO)",
        "GP+ (LOO log)",
        "tabpfn_v2.5",
        "tabpfn_v2",
    ]

    if args.only in ("both", "no_loo"):
        df = collect_logscale_df(
            gp_dir=gp_dir,
            pe_dir=pe_dir,
            loo_dir=None,
            gp_log_dir=gp_log_dir,
            pe_log_dir=pe_log_dir,
            loo_log_dir=None,
            pfn25_dir=pfn25,
            pfn20_dir=pfn20,
            include_loo=False,
            problems=problems,
            noise_levels=args.noise_levels,
        )
        if df.empty:
            raise SystemExit("No data for no-LOO plots.")
        print(f"[no LOO] rows={len(df)} models={sorted(df['model_family'].unique())}")
        print(f"[no LOO] problems={sorted(df['problem'].unique())}")
        make_separate_metric_plots(df, out_no_loo, order_no_loo, remove_outliers=args.remove_outliers)

    if args.only in ("both", "loo"):
        df = collect_logscale_df(
            gp_dir=gp_dir,
            pe_dir=pe_dir,
            loo_dir=loo_dir,
            gp_log_dir=gp_log_dir,
            pe_log_dir=pe_log_dir,
            loo_log_dir=loo_log_dir,
            pfn25_dir=pfn25,
            pfn20_dir=pfn20,
            include_loo=True,
            problems=problems,
            noise_levels=args.noise_levels,
        )
        if df.empty:
            raise SystemExit("No data for LOO plots.")
        print(f"[with LOO] rows={len(df)} models={sorted(df['model_family'].unique())}")
        print(f"[with LOO] problems={sorted(df['problem'].unique())}")
        make_separate_metric_plots(df, out_loo, order_loo, remove_outliers=args.remove_outliers)


if __name__ == "__main__":
    main()
