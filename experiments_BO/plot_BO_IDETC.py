"""
IDETC figures: Two multi-subplot figures (one per noise level) comparing BO models across 8 problems.
Uses same data sources and styling as plot_BO.py.
Creates both noisy-y and clean-y versions.

Run: python experiments_BO/plot_BO_IDETC.py

Output: experiments_BO/results_BO/BO_plots_IDETC/
  - BO_IDETC_noise0.002.png / BO_IDETC_noise0.002_clean.png
  - BO_IDETC_noise0.08.png / BO_IDETC_noise0.08_clean.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_BO import (
    RunTuple,
    _average_best_y_at_each_iteration,
    _darker_color,
    _get_palette,
    collect_runs,
    extract_dim_key,
    extract_noise_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results_BO" / "BO_plots_IDETC"

# Same model paths as plot_BO.py
DEFAULT_PATHS: list[tuple[str, str]] = [
    ("results_BO/GP+", "GP+"),
    # ("results_BO/GP+_TS", "GP+_TS"),
    # ("results_BO/GP+_no_AF_optimize", "GP+_no_AF_optimize"),
    ("results_BO/PFN_V2.0", "PFN 2.0"),
    # # ("results_BO/PFN_V2.0_GI", "PFN 2.0_GI"),
    # ("results_BO/PFN_V2.0_TS", "PFN 2.0_TS"),
    # # ("results_BO/PFN_V2.0_TS_GI", "PFN 2.0_TS_GI"),
    # ("results_BO/PFN_V2.5_BROKE", "PFN 2.5_old"),
    ("results_BO/PFN_V2.5", "PFN 2.5"),
    # ("results_BO/PFN_V2.5_TS", "PFN 2.5_TS"),

  ]
# 8 subplots in order: (folder, dim_key, subtitle)
# Desired order: Buckling, Borehole, Wing, Ackley 20D, Griewank 20D, Zakharov 20D, Ackley 40D, Dixon-Price 40D
IDETC_PROBLEMS: list[tuple[str, str, str]] = [
    ("B2_buckling", "", "Buckling"),
    ("B3_borehole", "", "Borehole"),
    ("B1_wing", "", "Wing"),
    ("B4_ackley", "20Dx", "Ackley 20D"),
    ("B8_griewank", "20Dx", "Griewank 20D"),
    ("B7_zakharov", "20Dx", "Zakharov 20D"),
    ("B4_ackley", "40Dx", "Ackley 40D"),
    ("B9_dixon_price", "40Dx", "Dixon-Price 40D"),
]

# Maximization problems: plot -y so trend is downward (consistent with minimization subplots)
MAXIMIZATION_PROBLEMS: frozenset[tuple[str, str]] = frozenset({
    ("B3_borehole", ""),
    ("B2_buckling", ""),
})

NOISE_LOW = "noiseTest0.002_noiseTrain0.002"
NOISE_HIGH = "noiseTest0.08_noiseTrain0.08"

# Darkening factor for average lines and legend (0.8 = slightly darker than base PLOT_COLORS)
COLOR_FACTOR = 0.8

# Fixed known optima (raw objective space) for SF benchmark variants.
# Keep these as constants (no recomputation during plotting):
# - Buckling s0 maximum: P = pi * E * I / (L*K)^2 at L=0.5, E=200, K=0.5, I=29.5
# - Borehole s0 maximum: Q at (rw,r,Tu,Hu,Tl,Hl,L,Kw)=(0.15,100,115600,1110,116,700,1120,12045)
# - Wing s0 minimum from bounded global solve (and monotonic structure check)
KNOWN_OPTIMA_RAW: dict[tuple[str, str], float] = {
    ("B2_buckling", ""): 296566.34649887646,
    ("B3_borehole", ""): 309.5755876604079,
    ("B1_wing", ""): 123.25367170091785,
    ("B4_ackley", "20Dx"): 0.0,
    ("B8_griewank", "20Dx"): 0.0,
    ("B7_zakharov", "20Dx"): 0.0,
    ("B4_ackley", "40Dx"): 0.0,
    ("B9_dixon_price", "40Dx"): 0.0,
}


def _is_max_problem(folder: str, dim_key: str) -> bool:
    return (folder, dim_key) in MAXIMIZATION_PROBLEMS


def _best_so_far_from_yhat(
    yhat_hist: list[float],
    maximize_problem: bool,
) -> list[float]:
    """Convert per-iteration y_hat(x_chosen) into best-so-far y_hat history."""
    if not yhat_hist:
        return []
    arr = np.asarray(yhat_hist, dtype=float)
    if maximize_problem:
        return np.maximum.accumulate(arr).tolist()
    return np.minimum.accumulate(arr).tolist()


def _load_yhat_runs_for_file(
    jf: Path,
    folder: str,
    dim_key: str,
) -> list[RunTuple]:
    """
    Load y_hat histories as best-so-far from one JSON file.
    Supports aggregated BO_full_info and per-run log files.
    """
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    maximize_problem = _is_max_problem(folder, dim_key)

    runs: list[RunTuple] = []

    if "BO_full_info" in data and isinstance(data["BO_full_info"], list):
        metrics = data.get("BO_metrics", [])
        for idx, entry in enumerate(data["BO_full_info"]):
            yhat_hist = entry.get("y_pred_mean_history")
            if not isinstance(yhat_hist, list) or not yhat_hist:
                continue
            best_hist = _best_so_far_from_yhat(yhat_hist, maximize_problem)
            total_time = None
            if isinstance(metrics, list) and idx < len(metrics):
                m = metrics[idx]
                if isinstance(m, dict) and "Total_Time" in m:
                    try:
                        total_time = float(m["Total_Time"])
                    except (TypeError, ValueError):
                        total_time = None
            run_id = entry.get("run", idx)
            runs.append((int(run_id), best_hist, total_time))
        return runs

    # Per-run file fallback
    yhat_hist = data.get("y_pred_mean_history")
    if isinstance(yhat_hist, list) and yhat_hist:
        best_hist = _best_so_far_from_yhat(yhat_hist, maximize_problem)
        runs.append((0, best_hist, None))
    return runs


def collect_runs_yhat(
    model_dir: Path,
) -> dict[tuple[str, str, str], list[RunTuple]]:
    """
    Collect best-y_hat histories from model directory.
    Key: (problem_folder, dim_key, noise_config) -> list[RunTuple].
    """
    result: dict[tuple[str, str, str], list[RunTuple]] = {}
    for prob_dir in sorted(model_dir.iterdir()):
        if not prob_dir.is_dir():
            continue
        folder = prob_dir.name
        run_dirs = sorted(
            [d for d in prob_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        )
        if run_dirs:
            for run_dir in run_dirs:
                for jf in run_dir.glob("*.json"):
                    noise = extract_noise_config(jf.name)
                    if not noise:
                        continue
                    dim_key = extract_dim_key(jf.stem) or ""
                    key = (folder, dim_key, noise)
                    loaded = _load_yhat_runs_for_file(jf, folder=folder, dim_key=dim_key)
                    if loaded:
                        result.setdefault(key, []).extend(loaded)
        else:
            for jf in prob_dir.glob("*.json"):
                noise = extract_noise_config(jf.name)
                if not noise:
                    continue
                dim_key = extract_dim_key(jf.stem) or ""
                key = (folder, dim_key, noise)
                loaded = _load_yhat_runs_for_file(jf, folder=folder, dim_key=dim_key)
                if loaded:
                    result.setdefault(key, []).extend(loaded)
    return result


def _draw_subplot(
    ax: plt.Axes,
    models_clean: list[tuple[str, list[RunTuple]]],
    title: str,
    models_yhat: list[tuple[str, list[RunTuple]]] | None = None,
    ylabel: str = "Best y",
    optimum_line_y: float | None = None,
) -> None:
    """Draw one problem comparison onto an existing axes (same style as plot_problem_comparison)."""
    palette = _get_palette(len(models_clean))
    # Compute single tick height from combined y range of all data (so all ticks same size)
    all_vals: list[float] = []
    for _, runs in models_clean:
        _, mean_vals = _average_best_y_at_each_iteration(runs)
        all_vals.extend(v for v in mean_vals if not np.isnan(v))
        for _, hist, _ in runs:
            all_vals.extend(hist)
    if models_yhat is not None:
        for _, runs in models_yhat:
            _, mean_vals = _average_best_y_at_each_iteration(runs)
            all_vals.extend(v for v in mean_vals if not np.isnan(v))
            for _, hist, _ in runs:
                all_vals.extend(hist)
    y_range = max(all_vals) - min(all_vals) if all_vals else 1.0
    tick_h = y_range * 0.11

    # 1. Draw all individual runs first (zorder=1, alpha so they show through each other)
    for idx, (_, runs) in enumerate(models_clean):
        color = palette[idx]
        for _, hist, _ in runs:
            iters = list(range(1, len(hist) + 1))
            ax.plot(iters, hist, color=color, alpha=0.75, linestyle="-", linewidth=0.4, zorder=1)

    # 2. Draw clean average lines on top.
    for idx, (label, runs) in enumerate(models_clean):
        color = palette[idx]
        avg_color = _darker_color(color, factor=COLOR_FACTOR)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            ax.plot(
                iters, mean_vals, color=avg_color, linestyle="--", linewidth=2.5, label=label,
                alpha=0.85, zorder=2
            )
            # Final average clean value dot
            ax.plot(
                [iters[-1]],
                [mean_vals[-1]],
                marker="o",
                markersize=4.0,
                linestyle="None",
                color=avg_color,
                zorder=4,
            )

    # 2b. Draw y_hat average lines (if requested).
    if models_yhat is not None:
        for idx, (_, runs) in enumerate(models_yhat):
            color = palette[idx]
            avg_color = _darker_color(color, factor=COLOR_FACTOR)
            iters_hat, mean_vals_hat = _average_best_y_at_each_iteration(runs)
            if iters_hat and not np.all(np.isnan(mean_vals_hat)):
                ax.plot(
                    iters_hat,
                    mean_vals_hat,
                    color=avg_color,
                    linestyle="-.",
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=2,
                )
                # Final average y_hat value dot
                ax.plot(
                    [iters_hat[-1]],
                    [mean_vals_hat[-1]],
                    marker="s",
                    markersize=4.2,
                    linestyle="None",
                    color=avg_color,
                    zorder=4,
                )

    # 3. Draw tick marks on top (zorder=3)
    for idx, (_, runs) in enumerate(models_clean):
        color = palette[idx]
        avg_color = _darker_color(color, factor=COLOR_FACTOR)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            avg_n_iters = np.mean([len(r[1]) for r in runs])
            idx_at_avg = int(round(avg_n_iters)) - 1
            if 0 <= idx_at_avg < len(mean_vals):
                x_tick = iters[idx_at_avg]
                y_tick = mean_vals[idx_at_avg]
                ax.plot([x_tick, x_tick], [y_tick - tick_h / 2, y_tick + tick_h / 2], color=avg_color, linewidth=2, linestyle=":", zorder=3)

    if optimum_line_y is not None:
        ax.axhline(
            optimum_line_y,
            color="0.2",
            linestyle=":",
            linewidth=1.8,
            alpha=0.95,
            zorder=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def plot_idetc_figures(
    models: list[tuple[Path, str]],
    out_dir: Path | None = None,
    use_clean_y: bool = False,
    include_yhat: bool = False,
) -> None:
    """Create two figures (noise 0.002 and 0.08), each with 8 problem subplots + 1 legend subplot (3x3 grid)."""
    use_out = out_dir if out_dir is not None else OUT_DIR
    use_out.mkdir(parents=True, exist_ok=True)

    all_data: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    for model_dir, _ in models:
        data, _ = collect_runs(model_dir, use_clean_y=use_clean_y)
        all_data.append(data)

    all_data_yhat: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    if include_yhat:
        for model_dir, _ in models:
            all_data_yhat.append(collect_runs_yhat(model_dir))

    if include_yhat:
        y_label = "Best y_hat"
        clean_suffix = "_yhat"
    else:
        y_label = "Best y (clean)" if use_clean_y else "Best y"
        clean_suffix = "_clean" if use_clean_y else ""

    for noise_config, noise_label in [(NOISE_LOW, "0.002"), (NOISE_HIGH, "0.08")]:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes_flat = axes.flatten()

        for i, (folder, dim_key, subtitle) in enumerate(IDETC_PROBLEMS):
            key = (folder, dim_key, noise_config)
            model_runs = []
            for j, (_, label) in enumerate(models):
                if include_yhat:
                    runs = all_data_yhat[j].get(key, [])
                else:
                    runs = all_data[j].get(key, [])
                model_runs.append((label, runs))

            # For maximization problems, plot -y so trend is downward (same as minimization)
            is_max = (folder, dim_key) in MAXIMIZATION_PROBLEMS
            if is_max:
                plot_runs_clean = [
                    (label, [(run_id, [-v for v in hist], t) for run_id, hist, t in runs])
                    for label, runs in model_runs
                ]
                if include_yhat:
                    model_runs_hat = None
                    subplot_ylabel = "Best (-y_hat)"
                else:
                    model_runs_hat = None
                    subplot_ylabel = ("Best (-y) (clean)" if use_clean_y else "Best (-y)")
            else:
                plot_runs_clean = model_runs
                model_runs_hat = None
                subplot_ylabel = y_label

            optimum_line_y = None
            true_opt = KNOWN_OPTIMA_RAW.get((folder, dim_key))
            if true_opt is not None:
                # Maximization panels are plotted in transformed (-y) space.
                optimum_line_y = -true_opt if is_max else true_opt

            ax = axes_flat[i]
            _draw_subplot(
                ax,
                plot_runs_clean,
                models_yhat=model_runs_hat,
                title=subtitle,
                ylabel=subplot_ylabel,
                optimum_line_y=optimum_line_y,
            )

        # Last subplot: legend - use actual line handles from first subplot for exact color match
        ax_legend = axes_flat[8]
        ax_legend.axis("off")
        # Build legend entries with styles that explicitly match the average lines:
        # dashed lines in the same colors as the subplot averages.
        legend_elements = []
        palette = _get_palette(len(models))
        for i, (_, label) in enumerate(models):
            base_color = palette[i]
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=_darker_color(base_color, factor=COLOR_FACTOR),
                    linestyle="--",
                    linewidth=2.5,
                    label=label,
                )
            )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="|",
                markersize=18,
                linestyle="None",
                markeredgewidth=2.5,
                color="0.3",
                label="Avg. iterations",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                linestyle=":",
                linewidth=1.8,
                label="Known true optimum",
            )
        )
        if include_yhat:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="0.25",
                    label="Final best y_hat (dot)",
                )
            )
        ax_legend.legend(handles=legend_elements, loc="center", fontsize=10, title=f"Noise = {noise_label} * y_std")
        fig.tight_layout()
        out_file = use_out / f"BO_IDETC_noise{noise_label}{clean_suffix}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_file}")


def main() -> None:
    root = SCRIPT_DIR
    models = [(root / p, label) for p, label in DEFAULT_PATHS]
    plot_idetc_figures(models, use_clean_y=False)  # Noisy y
    plot_idetc_figures(models, use_clean_y=True)   # Clean y
    plot_idetc_figures(models, use_clean_y=True, include_yhat=True)  # y_hat only
    print("Done.")


if __name__ == "__main__":
    main()
