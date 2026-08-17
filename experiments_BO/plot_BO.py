"""
Compare best y vs iterations across multiple BO models (GP+, PFN variants, etc.).
- Creates per-problem plots in subfolders (folder name = problem title)
- Creates an "all runs" plot with best run (by final y) solid, others dashed
- Matches problems across models by folder name and noise config

How to run:
  - With no args: uses DEFAULT_PATHS below (edit to add folders and labels).
        python experiments_BO/compare_BO.py
  - With args: model1_dir model2_dir [model3_dir ...] [--output output_dir] [--labels L1,L2,...]
        python experiments_BO/compare_BO.py results/GP+ results/PFN_V2.0_no_GI results/PFN_V2.0_GI --labels "GP+,PFN 2.0,PFN 2.0_GI"

Output folder: experiments_BO/results/compare_BO/
  - One subfolder per problem (folder name = problem title, e.g. B1_wing, B3_borehole)
  - One comparison plot per (problem, noise): compare_<full_title>.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paths (overridden by CLI or DEFAULT_PATHS when set)
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results_BO" / "BO_plots"

# Custom base colors for better distinguishability
PLOT_COLORS = ["#42A5F5", "#E53935", "#43A047", "#9C27B0"]  # blue, red, green, purple


def _get_palette(n: int) -> list[str]:
    """
    Return a list of n visually distinct colors.

    - For n <= len(PLOT_COLORS), just use the base list.
    - For larger n, use a matplotlib qualitative colormap (tab20) to generate n colors.
    """
    if n <= len(PLOT_COLORS):
        return PLOT_COLORS[:n]

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap("tab20", n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def extract_noise_config(filename: str) -> str | None:
    """Extract noise config from filename, e.g. noiseTest0.08_noiseTrain0.08."""
    m = re.search(r"noiseTest([\d.]+)_noiseTrain([\d.]+)", filename)
    return m.group(0) if m else None


def extract_dim_key(filename: str) -> str | None:
    """
    Extract dimension descriptor from filename stem, e.g. '_20Dx_5Dn_' -> '20Dx'.

    We group problems by (folder, dim_key, noise) so that 5D vs 20D variants
    are treated as distinct problems even if they live in the same folder and
    share the same noise configuration. For problems without an explicit Dx
    tag in the filename (e.g. B1–B3), we fall back to a dimension-less key.
    """
    m = re.search(r"_(\d+)Dx_", filename)
    if not m:
        return None
    return f"{m.group(1)}Dx"


def load_result_file(
    path: Path, use_clean_y: bool = False
) -> list[tuple[int, list[float], float | None]] | None:
    """
    Load result file (GP or PFN format). Returns [(run_id, best_y_history, total_time_s), ...].
    Supports: (1) per-run files with top-level best_y_history; (2) aggregated BO_full_info + BO_metrics.
    When use_clean_y=True, uses best_y_clean_history (fallback to best_y_history if absent).
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None

    hist_key = "best_y_clean_history" if use_clean_y else "best_y_history"
    fallback_key = "best_y_history" if use_clean_y else None

    # Per-run file: best_y_history at top level, no Total_Time
    if "best_y_history" in data and "BO_full_info" not in data:
        hist = data.get(hist_key) or (data.get(fallback_key) if fallback_key else None)
        if isinstance(hist, list) and hist:
            return [(0, hist, None)]
        return None

    # Aggregated file: BO_full_info + BO_metrics for Total_Time
    if "BO_full_info" in data and "BO_metrics" in data:
        full_info = data["BO_full_info"]
        metrics = data["BO_metrics"]
        runs = []
        for idx, entry in enumerate(full_info):
            hist = entry.get(hist_key) if use_clean_y else entry.get("best_y_history")
            if use_clean_y and (not hist or not isinstance(hist, list)):
                hist = entry.get("best_y_history")
            if not hist or not isinstance(hist, list) or not hist:
                continue
            total_time = None
            if idx < len(metrics) and "Total_Time" in metrics[idx]:
                total_time = float(metrics[idx]["Total_Time"])
            run_id = entry.get("run", idx)
            runs.append((run_id, hist, total_time))
        return runs if runs else None
    return None


def _stem_prefix_from_name(stem: str) -> str:
    """Infer stem prefix to strip (gp_, pfn_, etc.) from filename stem."""
    for p in ("gp_", "pfn_"):
        if stem.startswith(p):
            return p
    return ""


def _format_times_text(times: list[float | None]) -> str:
    """Format total times for display: 'run 0: 12.3 s; run 1: 15.2 s; ...' or 'N/A'."""
    parts = []
    for i, t in enumerate(times):
        if t is not None:
            parts.append(f"run {i}: {t:.1f} s")
        else:
            parts.append(f"run {i}: N/A")
    return "; ".join(parts) if parts else "N/A"


def _format_times_summary(times: list[float | None]) -> str:
    """Compact summary for many runs: 'N runs; min X s, max Y s, mean Z s' (or N/A)."""
    valid = [t for t in times if t is not None]
    if not valid:
        return "N/A"
    return f"{len(valid)} runs; min {min(valid):.1f} s, max {max(valid):.1f} s, mean {np.mean(valid):.1f} s"


def _format_times_mean_std(times: list[float | None]) -> str:
    """Format for all-runs plots: 'mean ± std s' (or N/A)."""
    valid = [t for t in times if t is not None]
    if not valid:
        return "N/A"
    mean_t = float(np.mean(valid))
    std_t = float(np.std(valid))
    return f"{mean_t:.1f} ± {std_t:.1f} s"


RunTuple = tuple[int, list[float], float | None]  # (run_id, best_y_history, total_time_s)


def _stem_to_full_title(stem: str, prefix: str) -> str:
    """Strip leading gp_ or pfn_ from filename stem to get full run descriptor for titles/filenames."""
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def collect_runs(
    model_dir: Path, use_clean_y: bool = False
) -> tuple[dict[tuple[str, str, str], list[RunTuple]], dict[tuple[str, str, str], str]]:
    """
    Collect runs from any model folder: (problem_folder, dim_key, noise_config) -> [(run_id, best_y_history, total_time_s), ...]
    Also returns (folder, dim_key, noise) -> full_title for plot/filename use.
    Supports both per-run (run_*) subfolders and aggregated JSONs in problem folder.
    When use_clean_y=True, uses best_y_clean_history instead of best_y_history.
    """
    result: dict[tuple[str, str, str], list[RunTuple]] = {}
    stems: dict[tuple[str, str, str], str] = {}

    for prob_dir in sorted(model_dir.iterdir()):
        if not prob_dir.is_dir():
            continue
        folder = prob_dir.name

        # Check for run_N subfolders (per-run structure)
        run_dirs = sorted([d for d in prob_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if run_dirs:
            for run_dir in run_dirs:
                m = re.match(r"run_(\d+)", run_dir.name)
                run_id = int(m.group(1)) if m else 0
                for jf in run_dir.glob("*.json"):
                    noise = extract_noise_config(jf.name)
                    if not noise:
                        continue
                    dim_key = extract_dim_key(jf.stem) or ""
                    key = (folder, dim_key, noise)
                    if key not in stems:
                        prefix = _stem_prefix_from_name(jf.stem)
                        stems[key] = _stem_to_full_title(jf.stem, prefix)
                    loaded = load_result_file(jf, use_clean_y=use_clean_y)
                    if loaded:
                        for _, hist, total_time in loaded:
                            if hist:
                                result.setdefault(key, []).append((run_id, hist, total_time))
        else:
            # Aggregated structure: JSONs directly in problem folder
            for jf in prob_dir.glob("*.json"):
                noise = extract_noise_config(jf.name)
                if not noise:
                    continue
                dim_key = extract_dim_key(jf.stem) or ""
                key = (folder, dim_key, noise)
                if key not in stems:
                    prefix = _stem_prefix_from_name(jf.stem)
                    stems[key] = _stem_to_full_title(jf.stem, prefix)
                loaded = load_result_file(jf, use_clean_y=use_clean_y)
                if loaded:
                    for run_id_inner, hist, total_time in loaded:
                        if hist:
                            result.setdefault(key, []).append((run_id_inner, hist, total_time))

    return result, stems


def _average_best_y_at_each_iteration(runs: list[RunTuple]) -> tuple[list[int], list[float]]:
    """
    At each iteration, compute mean best_y over all runs. Runs that end early have
    their final best_y carried forward to later iterations so the average never
    increases (best_y is "best so far", so holding the last value is correct).
    """
    if not runs:
        return [], []
    max_len = max(len(r[1]) for r in runs)
    if max_len == 0:
        return [], []
    iters_out = list(range(1, max_len + 1))
    means = []
    for k in range(max_len):
        # For each run: use value at iteration k if available, else carry forward last value
        vals = [
            (r[1][k] if len(r[1]) > k else r[1][-1])
            for r in runs
            if r[1]
        ]
        means.append(float(np.mean(vals)) if vals else np.nan)
    return iters_out, means


def _darker_color(color: str, factor: float = 0.7) -> tuple[float, float, float, float]:
    """Return a slightly darker/richer version of a matplotlib color for the average line."""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgba(color)
    darker = (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor, 1.0)
    return darker


def plot_problem_comparison(
    models: list[tuple[str, list[RunTuple]]],
    title: str,
    out_path: Path,
) -> None:
    """Plot best y vs iteration for one problem across N models. Solid = individual runs; dashed = average best_y."""
    fig, ax = plt.subplots(figsize=(8, 5))

    time_parts: list[str] = []
    palette = _get_palette(len(models))

    # 1. Draw all individual runs first (zorder=1, alpha so they show through each other)
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        for run_id, hist, _ in runs:
            iters = list(range(1, len(hist) + 1))
            ax.plot(iters, hist, color=color, alpha=0.75, linestyle="-", linewidth=0.6, zorder=1)

    # 2. Draw all average lines on top (zorder=2, alpha so they show through each other)
    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        avg_color = _darker_color(color, factor=0.6)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            ax.plot(
                iters, mean_vals, color=avg_color, linestyle="-", linewidth=2.5, label=label,
                alpha=0.85, zorder=2
            )

    # Vertical tick at average iterations for each model (single tick height from full y range)
    all_vals: list[float] = []
    for _, runs in models:
        _, mean_vals = _average_best_y_at_each_iteration(runs)
        all_vals.extend(v for v in mean_vals if not np.isnan(v))
        for _, hist, _ in runs:
            all_vals.extend(hist)
    y_range = max(all_vals) - min(all_vals) if all_vals else 1.0
    tick_h = y_range * 0.11

    for idx, (label, runs) in enumerate(models):
        color = palette[idx]
        avg_color = _darker_color(color, factor=0.6)
        iters, mean_vals = _average_best_y_at_each_iteration(runs)
        if iters and not np.all(np.isnan(mean_vals)):
            avg_n_iters = np.mean([len(r[1]) for r in runs])
            idx_at_avg = int(round(avg_n_iters)) - 1  # iters is 1-indexed
            if 0 <= idx_at_avg < len(mean_vals):
                x_tick = iters[idx_at_avg]
                y_tick = mean_vals[idx_at_avg]
                ax.plot([x_tick, x_tick], [y_tick - tick_h / 2, y_tick + tick_h / 2], color=avg_color, linewidth=2, linestyle=":")

    for idx, (label, runs) in enumerate(models):

        times = [r[2] for r in runs]
        time_parts.append(f"{label} total time (s): {_format_times_mean_std(times)}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best y")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    time_str = "\n".join(time_parts)
    fig.text(0.5, 0.02, time_str, ha="center", fontsize=8, family="monospace")
    bottom = max(0.16, 0.10 + 0.05 * len(models))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.92, bottom=bottom)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_run_comparison(
    models: list[tuple[str, RunTuple]],
    title: str,
    out_path: Path,
    run_idx: int,
) -> None:
    """Plot one run: N lines (one per model for run i)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    time_parts: list[str] = []
    palette = _get_palette(len(models))

    for idx, (label, run_tuple) in enumerate(models):
        _, hist, total_time = run_tuple
        color = palette[idx]
        iters = list(range(1, len(hist) + 1))
        ax.plot(iters, hist, color=color, linestyle="-", label=label)
        tstr = f"{total_time:.1f} s" if total_time is not None else "N/A"
        time_parts.append(f"{label}: {tstr}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best y")
    ax.set_title(f"{title} — run {run_idx}", fontsize=9)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.text(0.5, 0.02, "  |  ".join(time_parts), ha="center", fontsize=8, family="monospace")
    fig.subplots_adjust(left=0.1, right=0.96, top=0.92, bottom=0.16)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    models: list[tuple[Path, str]],
    out_dir: Path | None = None,
) -> None:
    """Run comparison across multiple models. models = [(results_dir, label), ...]."""
    use_out = out_dir if out_dir is not None else OUT_DIR
    use_out.mkdir(parents=True, exist_ok=True)

    if len(models) < 1:
        print("Need at least one model. Provide (dir, label) pairs.")
        return

    # Collect runs from each model
    all_data: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    all_stems: list[dict[tuple[str, str, str], str]] = []
    for model_dir, _ in models:
        data, stems = collect_runs(model_dir)
        all_data.append(data)
        all_stems.append(stems)

    # Find keys present in ALL models
    common_keys = set(all_data[0].keys())
    for d in all_data[1:]:
        common_keys &= set(d.keys())

    if not common_keys:
        print("No matching (problem, noise) across all models. Check paths and filenames.")
        return

    # Use first model's stems for full_title
    stems_ref = all_stems[0]

    for (folder, dim_key, noise) in sorted(common_keys):
        model_runs = [(label, all_data[i][(folder, dim_key, noise)])
                      for i, (_, label) in enumerate(models)]
        full_title = stems_ref.get((folder, dim_key, noise)) or folder

        subfolder = use_out / folder
        subfolder.mkdir(parents=True, exist_ok=True)
        out_file = subfolder / f"compare_{full_title}.png"
        plot_problem_comparison(model_runs, full_title, out_file)
        print(f"Saved: {out_file}")

        dim_prefix = f"{dim_key}_" if dim_key else ""
        runs_subfolder = subfolder / f"runs_{dim_prefix}{noise}"
        runs_subfolder.mkdir(parents=True, exist_ok=True)
        n_paired = min(len(runs) for _, runs in model_runs)
        for i in range(n_paired):
            run_plot_path = runs_subfolder / f"run_{i}.png"
            single_run_models = [(label, runs[i]) for label, runs in model_runs]
            plot_single_run_comparison(
                single_run_models,
                title=full_title,
                out_path=run_plot_path,
                run_idx=i,
            )
            print(f"Saved: {run_plot_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# "Just run this file": set default model folders and labels when no CLI args.
# DEFAULT_PATHS = [(folder, label), ...] — add as many as you want.
# Example (relative to experiments_BO/):
#   DEFAULT_PATHS = [
#       ("results/GP+", "GP+"),
#       ("results/PFN_V2.0", "PFN 2.0"),
#       ("results/PFN_V2.0_GI", "PFN 2.0_GI"),
#       ("results/PFN_V2.5", "PFN 2.5"),
#       ("results/PFN_V2.5_GI", "PFN 2.5_GI"),
#   ]
# ---------------------------------------------------------------------------
DEFAULT_PATHS: list[tuple[str, str]] = [
      ("results_BO/GP+", "GP+"),
      ("results_BO/GP+_no_AF_optimize", "GP+_no_AF_optimize"),
    #   ("results_BO/PFN_V2.0", "PFN 2.0"),
    #   ("results_BO/PFN_V2.0_GI", "PFN 2.0_GI"),
    #   ("results_BO/PFN_V2.0_TS", "PFN 2.0_TS"),
    #   ("results_BO/PFN_V2.0_TS_GI", "PFN 2.0_TS_GI"),
    #   ("results_BO/PFN_V2.5", "PFN 2.5"),
    #   ("results_BO/PFN_V2.5_TS", "PFN 2.5_TS"),

  ]

def _parse_args() -> tuple[list[tuple[Path, str]], Path | None]:
    """Parse CLI: model1_dir model2_dir [...] [--output out_dir] [--labels L1,L2,...]"""
    import sys
    root = Path(__file__).resolve().parent

    args = list(sys.argv[1:])
    if not args:
        if not DEFAULT_PATHS:
            print("No args and DEFAULT_PATHS is empty. Provide model folders.")
            sys.exit(1)
        models = [(root / p, label) for p, label in DEFAULT_PATHS]
        return models, None

    out_dir: Path | None = None
    labels_arg: str | None = None
    dirs: list[Path] = []
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            out_dir = Path(args[i + 1]).resolve()
            i += 2
            continue
        if args[i] == "--labels" and i + 1 < len(args):
            labels_arg = args[i + 1]
            i += 2
            continue
        if not args[i].startswith("--"):
            dirs.append(Path(args[i]).resolve())
        i += 1
    if not dirs:
        print("Provide at least one model directory.")
        sys.exit(1)

    if labels_arg:
        label_list = [s.strip() for s in labels_arg.split(",")]
        if len(label_list) != len(dirs):
            print(f"--labels has {len(label_list)} entries but {len(dirs)} directories. They must match.")
            sys.exit(1)
        models = list(zip(dirs, label_list))
    else:
        models = [(d, d.name) for d in dirs]

    return models, out_dir


if __name__ == "__main__":
    models_arg, out_dir_arg = _parse_args()
    main(models=models_arg, out_dir=out_dir_arg)
