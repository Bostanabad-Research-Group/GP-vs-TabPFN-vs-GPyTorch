"""
IDETC figures for Overleaf: create individual per-problem BO plots
and a LaTeX snippet that arranges them as subfigures (3 per row).

Uses only noisy best-observed y from JSON ``best_y_history`` (no clean-y or y_hat curves).

Run: python experiments_BO/plot_BO_IDETC_overleaf.py

Output
------
experiments_BO/results_BO/gpVpfn_BO_plots/
  - Individual PNGs for each (problem, noise) pair using sampled best_y
  - gpVpfn_BO_figures_noise{low,high}.tex (two figure* environments)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_BO import RunTuple, _average_best_y_at_each_iteration, _darker_color, _get_palette, collect_runs
from plot_BO_IDETC import (
    COLOR_FACTOR,
    IDETC_PROBLEMS,
    MAXIMIZATION_PROBLEMS,
    NOISE_HIGH,
    NOISE_LOW,
)
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results_BO" / "gpVpfn_BO_plots"

# Matplotlib figure size (inches).
# Slightly wider and marginally shorter so rows take a bit less vertical space.
SUBPLOT_FIGSIZE = (4.2, 2.25)
LEGEND_FIGSIZE = (3.0, 1.65)

# Fixed margins make the axes region land in the same place across PNGs.
# This avoids `tight_layout()` shifting the left/bottom padding based on how
# wide the y tick labels are (e.g., 1 digit vs 3 digits / scientific notation).
SUBPLOT_LEFT = 0.16
SUBPLOT_RIGHT = 0.98
SUBPLOT_BOTTOM = 0.18
SUBPLOT_TOP = 0.95

# Local model configuration for this Overleaf script.
# You can edit this list to control WHICH result folders are plotted
# and how they are NAMED in the legend, without touching plot_BO_IDETC.py.
#
# Each tuple: (relative_results_folder, legend_label)
# Paths are resolved relative to SCRIPT_DIR.
OVERLEAF_MODELS: list[tuple[str, str]] = [
    ("results_BO/GP+_", "GP+_Powell"),
    ("results_BO/GP+", "GP+_LBFGS"),
    ("results_BO/GP+_no_AF_optimize", "GP+_no_AF_optimize"),
    # ("results_BO/PFN_V2.5", "PFN 2.5"),
    # ("results_BO/PFN_V2.0", "PFN 2.0"),
    # Example of adding another variant:
    # ("results_BO/PFN_V2.5_homePC", "PFN 2.5 (home)"),
]


@dataclass(frozen=True)
class ProblemTexMeta:
    folder: str
    dim_key: str
    noise_key: str  # NOISE_LOW / NOISE_HIGH
    letter: str
    name: str  # e.g. "Buckling"
    dim_caption: str  # e.g. "D_x=4"
    slug: str  # used for filenames, e.g. "buckling_D4"


PROBLEM_DIM_INFO = {
    ("B2_buckling", ""): r"D$_x$=4",
    ("B3_borehole", ""): r"D$_x$=8",
    ("B1_wing", ""): r"D$_x$=10",
    ("B4_ackley", "20Dx"): r"D$_x$=20",
    ("B8_griewank", "20Dx"): r"D$_x$=20",
    ("B7_zakharov", "20Dx"): r"D$_x$=20",
    ("B4_ackley", "40Dx"): r"D$_x$=40",
    ("B9_dixon_price", "40Dx"): r"D$_x$=40",
}

# Optional overrides for nicer problem names in subcaptions
PROBLEM_NAME_OVERRIDES = {
    ("B2_buckling", ""): "Buckling",
    ("B3_borehole", ""): "Borehole",
    ("B1_wing", ""): "Wing Weight",
    ("B4_ackley", "20Dx"): "Ackley",
    ("B8_griewank", "20Dx"): "Griewank",
    ("B7_zakharov", "20Dx"): "Zakharov",
    ("B4_ackley", "40Dx"): "Ackley",
    ("B9_dixon_price", "40Dx"): "Dixon-Price",
}

PROBLEM_SLUGS = {
    ("B2_buckling", ""): "buckling_D4",
    ("B3_borehole", ""): "borehole_D8",
    ("B1_wing", ""): "wing_D10",
    ("B4_ackley", "20Dx"): "ackley_D20",
    ("B8_griewank", "20Dx"): "griewank_D20",
    ("B7_zakharov", "20Dx"): "zakharov_D20",
    ("B4_ackley", "40Dx"): "ackley_D40",
    ("B9_dixon_price", "40Dx"): "dixon_price_D40",
}

PROBLEM_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]


def _build_problem_tex_meta(noise_key: str) -> list[ProblemTexMeta]:
    """Create ordered metadata matching IDETC_PROBLEMS and the lettering in the paper."""
    meta_list: list[ProblemTexMeta] = []
    for idx, (folder, dim_key, subtitle) in enumerate(IDETC_PROBLEMS):
        letter = PROBLEM_LETTERS[idx]
        dim_caption = PROBLEM_DIM_INFO[(folder, dim_key)]
        slug = PROBLEM_SLUGS[(folder, dim_key)]
        display_name = PROBLEM_NAME_OVERRIDES.get((folder, dim_key), subtitle)
        meta_list.append(
            ProblemTexMeta(
                folder=folder,
                dim_key=dim_key,
                noise_key=noise_key,
                letter=letter,
                name=display_name,
                dim_caption=dim_caption,
                slug=slug,
            )
        )
    return meta_list


def _draw_single_subplot(
    ax: plt.Axes,
    models: list[tuple[str, list[RunTuple]]],
    ylabel: str,
    split_mode: str | None = None,
) -> None:
    """Copy of plot_BO_IDETC._draw_subplot but kept local to avoid circular imports."""
    palette = _get_palette(len(models))
    all_vals: list[float] = []
    for _, runs in models:
        _, mean_vals = _average_best_y_at_each_iteration(runs)
        all_vals.extend(v for v in mean_vals if not np.isnan(v))
        for _, hist, _ in runs:
            all_vals.extend(hist)
    y_range = max(all_vals) - min(all_vals) if all_vals else 1.0
    tick_h = y_range * 0.11

    # Individual runs
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        for run_id, hist, _ in runs:
            iters = list(range(1, len(hist) + 1))
            ax.plot(iters, hist, color=color, alpha=0.75, linestyle="-", linewidth=0.4, zorder=1)

    # Average lines (all dashed) with more separated dash *phases* so overlaps are visible
    n_models = max(len(models), 1)
    # Larger dash offsets but same dashed pattern
    dash_styles = [(offset, (4, 3)) for offset in np.linspace(0, 8, n_models)]

    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        avg_color = _darker_color(color, factor=COLOR_FACTOR)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            ax.plot(
                iters,
                mean_vals,
                color=avg_color,
                linestyle=dash_styles[idx],
                linewidth=2.25,
                label=label,
                alpha=0.85,
                zorder=2,
            )

    # Tick marks at average iteration.
    if split_mode is not None:
        # Special split cases for certain problems/noise configs.
        # split_mode: "pfn20_top" or "pfn25_top"
        label_to_idx = {label: i for i, (label, _) in enumerate(models)}
        idx_gp = label_to_idx.get("GP+")
        idx_pfn20 = label_to_idx.get("PFN 2.0")
        idx_pfn25 = label_to_idx.get("PFN 2.5")

        def _avg_point(model_idx: int | None) -> tuple[float, float] | None:
            if model_idx is None:
                return None
            _, runs = models[model_idx]
            iters, mean_vals = _average_best_y_at_each_iteration(runs)
            if not iters or np.all(np.isnan(mean_vals)):
                return None
            # Use the true mean iteration (no rounding) for the x-position.
            avg_n_iters = float(np.mean([len(r[1]) for r in runs]))
            # Guard against going past the available curve length.
            max_iter = float(len(mean_vals))
            x_tick = min(max(avg_n_iters, 1.0), max_iter)
            # Interpolate y at this (possibly fractional) iteration index.
            idx_low = int(np.floor(x_tick)) - 1
            idx_high = min(int(np.ceil(x_tick)) - 1, len(mean_vals) - 1)
            if idx_low < 0:
                idx_low = 0
            if idx_high < 0:
                idx_high = 0
            if idx_low == idx_high:
                y_tick = float(mean_vals[idx_low])
            else:
                t = (x_tick - (idx_low + 1)) / ((idx_high + 1) - (idx_low + 1))
                y_tick = float((1 - t) * mean_vals[idx_low] + t * mean_vals[idx_high])
            return x_tick, y_tick

        base_half = tick_h / 2.0

        # GP+ full bar
        pt_gp = _avg_point(idx_gp)
        if pt_gp is not None:
            x_tick, y_tick = pt_gp
            color = palette[idx_gp]
            avg_color = _darker_color(color, factor=COLOR_FACTOR)
            ax.plot(
                [x_tick, x_tick],
                [y_tick - base_half, y_tick + base_half],
                color=avg_color,
                linewidth=1.3,
                linestyle="-",
                zorder=3,
            )

        # Decide which PFN is top vs bottom based on split_mode
        top_idx, bot_idx = (idx_pfn20, idx_pfn25) if split_mode == "pfn20_top" else (idx_pfn25, idx_pfn20)

        # Top half
        pt_top = _avg_point(top_idx)
        if pt_top is not None:
            x_tick, y_tick = pt_top
            color = palette[top_idx]
            avg_color = _darker_color(color, factor=COLOR_FACTOR)
            ax.plot(
                [x_tick, x_tick],
                [y_tick, y_tick + base_half],
                color=avg_color,
                linewidth=1.3,
                linestyle="-",
                zorder=4,
            )

        # Bottom half
        pt_bot = _avg_point(bot_idx)
        if pt_bot is not None:
            x_tick, y_tick = pt_bot
            color = palette[bot_idx]
            avg_color = _darker_color(color, factor=COLOR_FACTOR)
            ax.plot(
                [x_tick, x_tick],
                [y_tick - base_half, y_tick],
                color=avg_color,
                linewidth=1.3,
                linestyle="-",
                zorder=5,
            )
    else:
        # Default: same-height ticks for all models.
        half_h = tick_h / 2.0
        for idx, (label, runs) in enumerate(models):
            color = palette[idx]
            avg_color = _darker_color(color, factor=COLOR_FACTOR)
            iters, mean_vals = _average_best_y_at_each_iteration(runs)
            if iters and not np.all(np.isnan(mean_vals)):
                # Use true mean iteration (no rounding) for tick x-position.
                avg_n_iters = float(np.mean([len(r[1]) for r in runs]))
                max_iter = float(len(mean_vals))
                x_tick = min(max(avg_n_iters, 1.0), max_iter)
                idx_low = int(np.floor(x_tick)) - 1
                idx_high = min(int(np.ceil(x_tick)) - 1, len(mean_vals) - 1)
                if idx_low < 0:
                    idx_low = 0
                if idx_high < 0:
                    idx_high = 0
                if idx_low == idx_high:
                    y_tick = float(mean_vals[idx_low])
                else:
                    t = (x_tick - (idx_low + 1)) / ((idx_high + 1) - (idx_low + 1))
                    y_tick = float((1 - t) * mean_vals[idx_low] + t * mean_vals[idx_high])
                ax.plot(
                    [x_tick, x_tick],
                    [y_tick - half_h, y_tick + half_h],
                    color=avg_color,
                    linewidth=1.3,
                    linestyle="-",
                    zorder=3,
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    # No title; names and dimensions are added in LaTeX subcaptions.
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def create_overleaf_plots_and_tex() -> None:
    """Generate per-problem PNGs (sampled best_y) and LaTeX snippets:

    - gpVpfn_BO_figures_noise{low,high}.tex: two figure* environments (one per noise level)
    - gpVpfn_BO_tables_noise{low,high}.tex: tables of median/std of final best y per problem/model (noisy only)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only use uncommented OVERLEAF_MODELS entries defined in this file
    models = [(SCRIPT_DIR / p, label) for p, label in OVERLEAF_MODELS]

    # Preload all runs: noisy best_y_history only (never best_y_clean_history or model y_hat).
    all_data: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    for model_dir, _ in models:
        data, _ = collect_runs(model_dir, use_clean_y=False)
        all_data.append(data)

    # Helper to compute final-best-y stats (noisy only) per (problem, noise, model),
    # plus iteration statistics (mean / std of number of iterations).
    def _final_stats(
        runs: list[RunTuple], is_max: bool
    ) -> tuple[float, float, float, float] | None:
        """Return (mean_y, std_y, mean_iters, std_iters) across runs."""
        vals: list[float] = []
        it_lens: list[int] = []
        for _run_id, hist, _t in runs:
            if not hist:
                continue
            it_lens.append(len(hist))
            v = float(hist[-1])
            if is_max:
                v = -v
            vals.append(v)
        if not vals or not it_lens:
            return None
        arr = np.asarray(vals, dtype=float)
        it_arr = np.asarray(it_lens, dtype=float)
        return (
            float(np.mean(arr)),
            float(np.std(arr, ddof=0)),
            float(np.mean(it_arr)),
            float(np.std(it_arr, ddof=0)),
        )

    def _fmt_mean_y(x: float) -> str:
        """Format best-y means.

        - Scientific for very large/small values, with 3 significant digits.
        - Plain with 2 decimal digits for typical scales (gives ~3 sig figs around 10).
        """
        ax = abs(x)
        if ax != 0 and (ax >= 1e4 or ax < 1e-3):
            s = f"{x:.2e}"  # 3 sig figs
            mantissa, exp = s.split("e")
            exp_i = int(exp)
            return f"{mantissa}e{exp_i}"
        if 1e-2 <= ax < 1e2:
            return f"{x:.2f}"
        return f"{x:.3g}"

    def _fmt_std_y(x: float) -> str:
        """Format std of best y.

        - Scientific: 2 significant digits.
        - Plain:
          * for very large values (>=100): no decimals,
          * for moderate values (10–100): one decimal place,
          * otherwise: two decimal places.
        """
        ax = abs(x)
        if ax != 0 and (ax >= 1e4 or ax < 1e-3):
            s = f"{x:.1e}"  # 2 sig figs
            mantissa, exp = s.split("e")
            exp_i = int(exp)
            return f"{mantissa}e{exp_i}"
        if ax >= 100.0:
            return f"{x:.0f}"
        if ax >= 10.0:
            return f"{x:.1f}"
        return f"{x:.2f}"

    def _fmt_mean_iter(x: float) -> str:
        """Format iteration means with one decimal place (always show .0 for integers)."""
        return f"{x:.1f}"

    def _fmt_std_iter(x: float) -> str:
        """Format iteration standard deviations with 1 decimal place."""
        return f"{x:.1f}"

    y_label = "Best y"

    # We create metadata lists separately for low and high noise
    noise_configs = [
        (NOISE_LOW, "0.002", "low"),
        (NOISE_HIGH, "0.08", "high"),
    ]

    # Create a shared legend PNG (independent of noise level).
    # Use a smaller canvas so that, when stretched to \linewidth in the minipage,
    # the legend box visually occupies a similar area as the other subplots.
    # Slightly larger legend canvas so the text is more legible in the paper.
    legend_fig, legend_ax = plt.subplots(figsize=LEGEND_FIGSIZE)
    legend_ax.axis("off")
    legend_elements = []
    palette = _get_palette(len(models))

    # In the legend, show simple dashed lines (no phase offset) for clarity.
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
            color="0.2",
            linestyle="None",
            linewidth=0.0,
            marker="|",  # vertical tick to match plot markers
            markersize=18,
            markeredgewidth=2.0,
            label="Avg. iterations",
        )
    )

    legend_ax.legend(
        handles=legend_elements,
        loc="center",
        # Nudge the legend slightly further upward so there's a bit more padding below.
        bbox_to_anchor=(0.5, 0.62),
        fontsize=12,
        frameon=True,
    )
    # Use a slightly tighter layout to reduce surrounding padding.
    legend_fig.tight_layout(pad=0.3)
    legend_path = OUT_DIR / "bo_legend_gpVpfn.png"
    legend_fig.savefig(legend_path, dpi=150)
    plt.close(legend_fig)
    print(f"Saved legend: {legend_path}")

    for noise_key, noise_label, noise_name in noise_configs:
        # Fresh table lines per noise so each noise level gets its own .tex file.
        tables_lines: list[str] = []
        tables_lines.append(r"% Auto-generated by plot_BO_IDETC_overleaf.py")
        tables_lines.append(
            rf"% Summary tables: median/std of final best $y$ (noisy only; lower is better), "
            rf"noise = {noise_label} * y_std."
        )
        tables_lines.append("")

        tex_lines: list[str] = []

        metas = _build_problem_tex_meta(noise_key)

        # 1. Make individual PNGs for this noise level
        for meta in metas:
            key = (meta.folder, meta.dim_key, meta.noise_key)
            model_runs: list[tuple[str, list[RunTuple]]] = []
            for j, (_, label) in enumerate(models):
                runs = all_data[j].get(key, [])
                model_runs.append((label, runs))

            is_max = (meta.folder, meta.dim_key) in MAXIMIZATION_PROBLEMS
            if is_max:
                plot_runs = [
                    (label, [(run_id, [-v for v in hist], t) for run_id, hist, t in runs])
                    for label, runs in model_runs
                ]
                subplot_ylabel = "Best (-y)"
            else:
                plot_runs = model_runs
                subplot_ylabel = y_label

            fig, ax = plt.subplots(figsize=SUBPLOT_FIGSIZE)
            # Special split modes:
            # - Borehole 0.002: PFN 2.0 top, PFN 2.5 bottom
            if meta.noise_key == NOISE_LOW:
                if meta.folder == "B3_borehole":
                    split_mode = "pfn20_top"
                else:
                    split_mode = None
            else:
                split_mode = None

            _draw_single_subplot(ax, plot_runs, ylabel=subplot_ylabel, split_mode=split_mode)

            # For some problems the raw y values are very large in magnitude, which
            # triggers Matplotlib's "offset" scientific notation and also makes the
            # subplot padding inconsistent.
            #
            # For these we:
            #   1) Manually rescale by 10^k so the tick labels are O(1).
            #   2) Use fixed tick spacing / decimals where requested.
            #   3) Append a mathtext scale factor to the y-label, e.g. "×10^5" with
            #      a proper superscript.
            if meta.folder in {"B2_buckling", "B3_borehole", "B7_zakharov", "B9_dixon_price"}:
                import math
                from matplotlib.ticker import FuncFormatter, MultipleLocator

                # Problem‑specific tick configuration in the *scaled* coordinates.
                # (step, decimal_places)
                tick_cfg: dict[tuple[str, str], tuple[float, int]] = {
                    # Buckling: ticks every 1, show as 0, 1, 2, ...
                    ("B2_buckling", ""): (1.0, 0),
                    # Borehole: ticks every 0.5, show as 0.0, 0.5, 1.0, ...
                    ("B3_borehole", ""): (0.5, 1),
                    # Zakharov: default spacing for low-noise (overridden for high-noise below).
                    ("B7_zakharov", "20Dx"): (2.0, 0),
                    # Dixon-Price: ticks every 1, show as whole numbers.
                    ("B9_dixon_price", "40Dx"): (1.0, 0),
                }

                cfg_key = (meta.folder, meta.dim_key)
                preferred_step, decimals = tick_cfg.get(cfg_key, (0.5, 1))
                # Noise-dependent tick rules (Zakharov request).
                if cfg_key == ("B7_zakharov", "20Dx"):
                    if meta.noise_key == NOISE_LOW:
                        preferred_step, decimals = (2.0, 0)
                    elif meta.noise_key == NOISE_HIGH:
                        preferred_step, decimals = (1.0, 0)

                # Determine exponent from the actual plotted data (not from ticks),
                # so we avoid Matplotlib's auto-offset behavior entirely.
                ydata: list[float] = []
                for line in ax.lines:
                    yd = line.get_ydata(orig=False)
                    if yd is None:
                        continue
                    ydata.extend([float(v) for v in np.asarray(yd).ravel().tolist()])
                finite = [v for v in ydata if np.isfinite(v) and v != 0.0]
                if finite:
                    max_abs = max(abs(v) for v in finite)
                    scale_exp = int(math.floor(math.log10(max_abs)))
                    scale = 10.0 ** scale_exp

                    # Keep the axis limits based on the data (consistent padding like other plots).
                    # Setting explicit tick locations can force view limits to the tick extremes,
                    # so we use a locator+formatter instead.
                    # Choose a step in *scaled* units that yields a reasonable number of ticks.
                    # We *prefer* the configured step, but if it would produce too few ticks
                    # for the current y-span (e.g. Zakharov noise=0.08), we halve until it fits.
                    y0, y1 = ax.get_ylim()
                    scaled_span = abs((y1 - y0) / scale) if scale != 0 else 0.0
                    step = float(preferred_step)
                    # If we're formatting as whole numbers, never allow sub-integer steps,
                    # otherwise different ticks collapse to the same label (e.g. -2.5, -2.0, -1.5 -> "-2").
                    min_step = 1.0 if decimals == 0 else 0.25
                    # Aim for at least ~3 intervals across the visible span.
                    while step > 0 and scaled_span / step < 3.0 and step > min_step:
                        step /= 2.0
                    if step < min_step:
                        step = min_step

                    ax.yaxis.set_major_locator(MultipleLocator(step * scale))

                    fmt = f"{{:.{decimals}f}}"

                    def _scaled_fmt(val: float, _pos: int) -> str:
                        return fmt.format(val / scale)

                    ax.yaxis.set_major_formatter(FuncFormatter(_scaled_fmt))
                    ax.yaxis.get_offset_text().set_visible(False)

                    # Put the scale factor in square brackets with proper superscript.
                    current_ylabel = ax.get_ylabel()
                    ax.set_ylabel(rf"{current_ylabel} [$\times 10^{{{scale_exp}}}$]")

            # Fixed margins for consistent axis placement across all subplots.
            fig.subplots_adjust(
                left=SUBPLOT_LEFT,
                right=SUBPLOT_RIGHT,
                bottom=SUBPLOT_BOTTOM,
                top=SUBPLOT_TOP,
            )

            out_file = OUT_DIR / f"{meta.slug}_noise{noise_label}.png"
            fig.savefig(out_file, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_file}")

        # 2. Append LaTeX figure* environment for this noise level (3 subfigures per row)
        tex_lines.append(r"\begin{figure*}[htb]")
        tex_lines.append(r"    \centering")
        tex_lines.append(r"    \captionsetup{font=footnotesize}")

        # Eight problem subfigures
        for idx, meta in enumerate(metas):
            # 3 subfigures per row: after every 3rd, avoid trailing \hfill to help spacing
            tex_lines.append(r"    \begin{minipage}[b]{0.33\linewidth}")
            tex_lines.append(r"        \centering")
            tex_lines.append(
                rf"        \includegraphics[width=\linewidth, trim = 0 3pt 0 0, clip]{{gpVpfn_BO_plots/{meta.slug}_noise{noise_label}.png}}"
            )
            # Let the subcaption package handle lettering ((a), (b), ...), so we
            # only include the descriptive text here.
            tex_lines.append(
                rf"        \subcaption{{{meta.name}: {meta.dim_caption}}}"
            )
            tex_lines.append(
                rf"        \label{{fig:bo_idetc_{meta.slug}_noise{noise_name}}}"
            )
            tex_lines.append(r"    \end{minipage}")
            # Horizontal spacing within a row.
            if (idx + 1) % 3 != 0:
                tex_lines.append(r"    \hfill")
            # Add a bit of vertical padding between rows so subcaptions from one
            # row do not visually collide with the plots in the next row.
            if (idx + 1) % 3 == 0 and (idx + 1) < len(metas):
                tex_lines.append(r"")
                tex_lines.append(r"    \vspace{0.5em}")
                tex_lines.append(r"")

        # Ninth subfigure: legend
        tex_lines.append(r"    \begin{minipage}[b]{0.33\linewidth}")
        tex_lines.append(r"        \centering")
        tex_lines.append(
            r"        \includegraphics[width=\linewidth, trim = 0 8pt 20pt 0, clip]{gpVpfn_BO_plots/bo_legend_gpVpfn.png}"
        )
        # Add an invisible subcaption so this minipage has similar height to the
        # others (which include subcaptions). This shifts the legend upwards
        # when using bottom-aligned minipages.
        tex_lines.append(r"        \subcaption*{\phantom{X}}")
        tex_lines.append(r"    \end{minipage}")

        if noise_label == "0.002":
            noise_caption = r"$\varepsilon \sim \mathcal{N}(0,(0.002\times c)^2)$"
        else:
            noise_caption = r"$\varepsilon \sim \mathcal{N}(0,(0.08\times c)^2)$"

        tex_lines.append(
            rf"    \caption{{Bayesian optimization performance for GP+ and TabPFN across benchmark problems "
            rf"with observation noise {noise_caption}. Buckling and Borehole are maximization problems, "
            rf"hence best negative y values are reported.}}"
        )
        tex_lines.append(rf"    \label{{fig:bo_idetc_gpVpfn_noise{noise_name}}}")
        tex_lines.append(r"\end{figure*}")
        tex_lines.append("")

        tex_path = OUT_DIR / f"gpVpfn_BO_figures_noise{noise_name}.tex"
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        print(f"Wrote LaTeX snippet to: {tex_path}")

        # 3. Append a single wide LaTeX table* for this noise level, similar to the timing table style.
        #    One row per problem, columns are GP+, PFN 2.0, PFN 2.5 (median ± std of final best y).

        # Collect rows: (problem name, dim_caption, {label: (median_y, std_y, mean_iters, std_iters)})
        summary_rows: list[tuple[ProblemTexMeta, dict[str, tuple[float, float, float, float]]]] = []
        for meta in metas:
            key = (meta.folder, meta.dim_key, meta.noise_key)
            is_max = (meta.folder, meta.dim_key) in MAXIMIZATION_PROBLEMS

            stats_by_model: dict[str, tuple[float, float, float, float]] = {}
            for j, (_, label) in enumerate(models):
                runs = all_data[j].get(key, [])
                stats = _final_stats(runs, is_max=is_max)
                if stats is None:
                    continue
                stats_by_model[label] = stats

            if not stats_by_model:
                continue
            summary_rows.append((meta, stats_by_model))

        if summary_rows:
            tables_lines.append(r"\begin{table*}[!h]")
            tables_lines.append(r"    \centering")
            # Slightly wider columns for readability, but still within page margins.
            tables_lines.append(r"    {\extracolsep{0pt}")
            tables_lines.append(r"    \setlength{\tabcolsep}{4.0pt}")
            tables_lines.append(r"    \renewcommand{\arraystretch}{0.95}")
            if noise_label == "0.002":
                noise_caption = r"$\varepsilon \sim \mathcal{N}(0,(0.002\times c)^2)$"
            else:
                noise_caption = r"$\varepsilon \sim \mathcal{N}(0,(0.08\times c)^2)$"

            tables_lines.append(
                rf"    \caption{{Final best $y$ (mean $\pm$ std; lower is better) "
                rf"for GP+, PFN 2.0, and PFN 2.5 across benchmark problems with observation noise "
                rf"{noise_caption}.}}"
            )
            tables_lines.append(
                r"    \begin{tabular}{>{\centering\arraybackslash}p{2.3cm}"
                r">{\centering\arraybackslash}p{1.0cm}"
                r">{\centering\arraybackslash}p{0.9cm}|c|c|c|c|c|c}"
            )
            tables_lines.append(r"    \toprule")
            tables_lines.append(
                r"    \multicolumn{3}{c|}{} & \multicolumn{2}{c|}{\textbf{GP+}} "
                r"& \multicolumn{2}{c|}{\textbf{PFN 2.0}} & \multicolumn{2}{c}{\textbf{PFN 2.5}} \\"
            )
            tables_lines.append(r"    \midrule")
            tables_lines.append(
                r"    \textbf{Problem} & \textbf{D$_x$} & \textbf{Noise} & "
                r"\textbf{Best $y$} & \textbf{Iters} & "
                r"\textbf{Best $y$} & \textbf{Iters} & "
                r"\textbf{Best $y$} & \textbf{Iters} \\"
            )
            tables_lines.append(r"    \midrule")

            for meta, stats_by_model in summary_rows:
                # All rows in this file share the same noise value.
                noise_str = noise_label
                prob_name = meta.name
                dx_str = meta.dim_caption

                # Determine best (lowest) median among available models for bolding.
                medians = [stats_by_model[lbl][0] for lbl in ("GP+", "PFN 2.0", "PFN 2.5") if lbl in stats_by_model]
                best_med = min(medians) if medians else None
                eps_best = 1e-12

                def _cells(label: str) -> tuple[str, str]:
                    if label not in stats_by_model:
                        return r"--", r"--"
                    med, std, it_mean, it_std = stats_by_model[label]
                    besty = rf"{_fmt_mean_y(med)} $\pm$ {_fmt_std_y(std)}"
                    iters = rf"{_fmt_mean_iter(it_mean)} $\pm$ {_fmt_std_iter(it_std)}"
                    if best_med is not None and abs(med - best_med) <= eps_best:
                        return rf"\textbf{{{besty}}}", rf"\textbf{{{iters}}}"
                    return besty, iters

                gp_y, gp_it = _cells("GP+")
                p20_y, p20_it = _cells("PFN 2.0")
                p25_y, p25_it = _cells("PFN 2.5")
                tables_lines.append(
                    rf"    \textbf{{{prob_name}}} & \textbf{{{dx_str}}} & {noise_str} & "
                    rf"{gp_y} & {gp_it} & {p20_y} & {p20_it} & {p25_y} & {p25_it} \\[2pt]"
                )

            tables_lines.append(r"    \bottomrule")
            tables_lines.append(r"    \end{tabular}")
            tables_lines.append(r"    }")
            tables_lines.append(r"\end{table*}")
            tables_lines.append("")

        # Write per-noise tables LaTeX file.
        tables_path = OUT_DIR / f"gpVpfn_BO_tables_noise{noise_name}.tex"
        tables_path.write_text("\n".join(tables_lines), encoding="utf-8")
        print(f"Wrote LaTeX tables to: {tables_path}")

    # 4. Combined-noise table: both 0.002 and 0.08 rows for each problem.
    combined_lines: list[str] = []
    combined_lines.append(r"% Auto-generated by plot_BO_IDETC_overleaf.py")
    combined_lines.append(
        r"% Summary table: final best y (mean ± std) and iterations (mean ± std) for both noise levels."
    )
    combined_lines.append("")
    combined_lines.append(r"\begin{table*}[!h]")
    combined_lines.append(r"    \centering")
    combined_lines.append(r"    {\extracolsep{0pt}")
    combined_lines.append(r"    \setlength{\tabcolsep}{4.0pt}")
    combined_lines.append(r"    \renewcommand{\arraystretch}{0.95}")
    combined_lines.append(
        r"    \caption{Final best $y$ (mean $\pm$ std; lower is better) and iterations (mean $\pm$ std) "
        r"for GP+, PFN 2.0, and PFN 2.5 across benchmark problems for both noise levels.}"
    )
    combined_lines.append(
        r"    \begin{tabular}{>{\centering\arraybackslash}p{2.3cm}"
        r">{\centering\arraybackslash}p{1.0cm}"
        r">{\centering\arraybackslash}p{0.9cm}|c|c|c|c|c|c}"
    )
    combined_lines.append(r"    \toprule")
    combined_lines.append(
        r"    \multicolumn{3}{c|}{} & \multicolumn{2}{c|}{\textbf{GP+}} "
        r"& \multicolumn{2}{c|}{\textbf{PFN 2.0}} & \multicolumn{2}{c}{\textbf{PFN 2.5}} \\"
    )
    combined_lines.append(r"    \midrule")
    combined_lines.append(
        r"    \textbf{Problem} & \textbf{D$_x$} & \textbf{Noise} & "
        r"\textbf{Best $y$} & \textbf{Iters} & "
        r"\textbf{Best $y$} & \textbf{Iters} & "
        r"\textbf{Best $y$} & \textbf{Iters} \\"
    )
    combined_lines.append(r"    \midrule")

    for folder, dim_key, subtitle in IDETC_PROBLEMS:
        # Compute stats for both noise levels for this problem.
        per_noise_stats: dict[str, dict[str, tuple[str, str]]] = {}

        for noise_key, noise_label, noise_name in noise_configs:
            key = (folder, dim_key, noise_key)
            is_max = (folder, dim_key) in MAXIMIZATION_PROBLEMS

            stats_by_model_raw: dict[str, tuple[float, float, float, float]] = {}
            for j, (_, label) in enumerate(models):
                runs = all_data[j].get(key, [])
                stats = _final_stats(runs, is_max=is_max)
                if stats is None:
                    continue
                stats_by_model_raw[label] = stats

            if not stats_by_model_raw:
                continue

            # Determine best mean best-y among available models for bolding.
            medians = [stats_by_model_raw[lbl][0] for lbl in ("GP+", "PFN 2.0", "PFN 2.5") if lbl in stats_by_model_raw]
            best_med = min(medians) if medians else None
            eps_best = 1e-12

            def _cells(label: str) -> tuple[str, str]:
                if label not in stats_by_model_raw:
                    return r"--", r"--"
                med, std, it_mean, it_std = stats_by_model_raw[label]
                besty = rf"{_fmt_mean_y(med)} $\pm$ {_fmt_std_y(std)}"
                iters = rf"{_fmt_mean_iter(it_mean)} $\pm$ {_fmt_std_iter(it_std)}"
                if best_med is not None and abs(med - best_med) <= eps_best:
                    return rf"\textbf{{{besty}}}", rf"\textbf{{{iters}}}"
                return besty, iters

            per_noise_stats[noise_label] = {
                "GP+": _cells("GP+"),
                "PFN 2.0": _cells("PFN 2.0"),
                "PFN 2.5": _cells("PFN 2.5"),
            }

        if not per_noise_stats:
            continue

        # Display name and dimension caption (same for both noise rows).
        prob_name = PROBLEM_NAME_OVERRIDES.get((folder, dim_key), subtitle)
        dx_str = PROBLEM_DIM_INFO[(folder, dim_key)]

        # First row: low noise prints problem + Dx, second row leaves them blank.
        for row_idx, (noise_key, noise_label, noise_name) in enumerate(noise_configs):
            if noise_label not in per_noise_stats:
                continue
            cells = per_noise_stats[noise_label]
            gp_y, gp_it = cells["GP+"]
            p20_y, p20_it = cells["PFN 2.0"]
            p25_y, p25_it = cells["PFN 2.5"]
            if row_idx == 0:
                combined_lines.append(
                    rf"    \textbf{{{prob_name}}} & \textbf{{{dx_str}}} & {noise_label} & "
                    rf"{gp_y} & {gp_it} & {p20_y} & {p20_it} & {p25_y} & {p25_it} \\[1pt]"
                )
            else:
                combined_lines.append(
                    rf"    & & {noise_label} & "
                    rf"{gp_y} & {gp_it} & {p20_y} & {p20_it} & {p25_y} & {p25_it} \\[4pt]"
                )

    combined_lines.append(r"    \bottomrule")
    combined_lines.append(r"    \end{tabular}")
    combined_lines.append(r"    }")
    combined_lines.append(r"\end{table*}")
    combined_lines.append("")

    combined_path = OUT_DIR / "gpVpfn_BO_tables.tex"
    combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
    print(f"Wrote combined-noise BO tables to: {combined_path}")


def main() -> None:
    create_overleaf_plots_and_tex()


if __name__ == "__main__":
    main()

