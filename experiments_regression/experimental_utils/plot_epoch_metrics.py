"""
Plot epoch vs training metrics from GP_Trainer_Analysis.json files.

Reads epoch_metrics from each run (list of {epoch, loss, NLL, NIS, LOO_NLL, KF, MSE, R2})
and plots one subplot per metric: epoch (x) vs metric value (y).
All inits as thin grey lines; chosen init (lowest loss per run) in green; best RRMSE init in red.

Usage:
  python plot_epoch_metrics.py path/to/gpVpfn_*_GP_Trainer_Analysis.json   # single file
  python plot_epoch_metrics.py results_v2.0/NLL3                          # all trainer_analysis JSONs under NLL3
  python plot_epoch_metrics.py                                            # default: results_v2.0/NLL3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_trainer_analysis_hyperparams import (
    extract_runs_and_chosen,
    find_trainer_analysis_jsons,
    load_trainer_analysis,
)

# Metrics to plot (keys in each epoch_metrics / lbfgs_inner_metrics entry)
# We skip "NLL" here because the loss is already NLL.
# Older JSONs may have "Residual_MSE" or "RRMSE" — fallback for "MSE" below.
EPOCH_METRIC_KEYS = ["loss", "NIS", "LOO_NLL", "KF", "MSE", "R2"]
ITER_METRIC_KEYS = EPOCH_METRIC_KEYS


DELTA_LOG10_LOSS_KEY = "__delta_log10_loss__"


def _compute_delta_log10(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-step improvement in log10-space:
      delta[t] = log10(y[t-1]) - log10(y[t])
    Positive values indicate improvement (decreasing metric).
    """
    if len(xs) < 2 or len(ys) < 2:
        return np.array([]), np.array([])
    prev_y = ys[:-1]
    next_y = ys[1:]
    mask = (
        np.isfinite(prev_y)
        & np.isfinite(next_y)
        & np.isfinite(xs[1:])
        & (prev_y > 0.0)
        & (next_y > 0.0)
    )
    if not np.any(mask):
        return np.array([]), np.array([])
    delta_x = xs[1:][mask]
    delta_y = np.log10(prev_y[mask]) - np.log10(next_y[mask])
    return delta_x, delta_y


def _get_metric_arrays(
    run: dict,
    entry_key: str,
    x_field: str,
    metric_keys: list[str],
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Generic helper: for a single run, extract (x, values) per metric from a list of dicts
    stored under run[entry_key]. Returns dict[metric_key] = (xs, ys).
    """
    em = run.get(entry_key) or []
    if not em:
        return {}
    out: dict[str, tuple[list[float], list[float]]] = {}
    for key in metric_keys:
        xs: list[float] = []
        ys: list[float] = []
        for e in em:
            x = e.get(x_field)
            v = e.get(key)
            # Backwards-compat: older logs used "Residual_MSE" or "RRMSE" for residual metric
            if v is None and key == "MSE":
                v = e.get("Residual_MSE")
            if x is not None and v is not None:
                try:
                    xs.append(float(x))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    pass
        if xs and ys:
            out[key] = (xs, ys)
    return out


def _get_epoch_metric_arrays(run: dict) -> dict[str, tuple[list[float], list[float]]]:
    """Epoch-level metrics: use 'epoch_metrics' and x field 'epoch'."""
    return _get_metric_arrays(run, entry_key="epoch_metrics", x_field="epoch", metric_keys=EPOCH_METRIC_KEYS)


def _get_iter_metric_arrays(run: dict) -> dict[str, tuple[list[float], list[float]]]:
    """Inner-iteration metrics: use 'lbfgs_inner_metrics' and x field 'lbfgs_iter'."""
    return _get_metric_arrays(run, entry_key="lbfgs_inner_metrics", x_field="lbfgs_iter", metric_keys=ITER_METRIC_KEYS)


def _group_key_to_suffix(group_key: str) -> str:
    """Convert _group_key (e.g. run_1) to a short suffix for filenames: run1."""
    if not group_key:
        return ""
    if group_key.startswith("run_"):
        try:
            return f"run{group_key.replace('run_', '')}"
        except ValueError:
            return group_key
    if group_key.startswith("fold_"):
        try:
            return f"run{group_key.replace('fold_', '')}"  # legacy: treat fold_N as run N
        except ValueError:
            return group_key
    return group_key.replace("_", "")


def _split_title_and_run_suffix(title: str) -> tuple[str, str]:
    """
    Split a title like 'wing_SF_10Dn_..._run1' (or legacy '..._fold1') into:
    - base title: 'wing_SF_10Dn_...'
    - suffix: 'run1'
    If there is no _run or _fold suffix, return (title, '').
    """
    if "_run" in title and title[-1].isdigit():
        base, run_part = title.rsplit("_run", 1)
        if run_part.isdigit():
            return base, f"run{run_part}"
    if "_fold" in title:
        base, fold_part = title.rsplit("_fold", 1)
        fold_part = fold_part.strip()
        if fold_part:
            return base, f"run{fold_part}"  # legacy: foldN -> runN
    return title, ""


def _plot_iter_metrics_for_runs(
    runs: list[dict],
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """Plot LBFGS inner-iteration metrics for a given list of runs."""
    runs_with_metrics = [r for r in runs if (r.get("lbfgs_inner_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with lbfgs_inner_metrics")

    metric_keys: list[str] = []
    for key in ITER_METRIC_KEYS:
        if any(_get_iter_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No inner-iteration metric data found")

    include_delta_log10 = any(
        len((_get_iter_metric_arrays(r).get("loss", ([], []))[0])) >= 2
        for r in runs_with_metrics
    )
    if include_delta_log10:
        metric_keys.append(DELTA_LOG10_LOSS_KEY)

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        for r in runs:
            arrs = _get_iter_metric_arrays(r)
            source_key = "loss" if key == DELTA_LOG10_LOSS_KEY else key
            it_v, val_v = arrs.get(source_key, ([], []))
            if not it_v or not val_v:
                continue
            it_v = np.array(it_v)
            val_v = np.array(val_v)
            if key == DELTA_LOG10_LOSS_KEY:
                it_v, val_v = _compute_delta_log10(it_v, val_v)
                if len(it_v) == 0 or len(val_v) == 0:
                    continue
            if r.get("_best_rrmse"):
                ax.plot(
                    it_v,
                    val_v,
                    c="red",
                    alpha=0.95,
                    linewidth=2.5,
                    zorder=6,
                    label="Best RRMSE (init)" if not labeled_best else None,
                )
                labeled_best = True
            elif r.get("_chosen"):
                ax.plot(
                    it_v,
                    val_v,
                    c="C2",
                    alpha=0.85,
                    linewidth=2,
                    zorder=5,
                    label="Chosen init" if not labeled_chosen else None,
                )
                labeled_chosen = True
            else:
                ax.plot(
                    it_v,
                    val_v,
                    c="gray",
                    alpha=0.45,
                    linewidth=0.9,
                    zorder=1,
                    label="All inits" if not labeled_all else None,
                )
                labeled_all = True
        ax.set_xlabel("LBFGS iteration")
        if key == DELTA_LOG10_LOSS_KEY:
            ax.set_ylabel("Delta log10 loss")
            ax.set_title("Delta log10 loss")
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5, linestyle="--")
        else:
            ax.set_ylabel(key.replace("_", " "))
            ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Iteration vs metrics — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    # Save iter-metrics plots into a per-experiment folder:
    #   <out_dir>/iter_metrics_<experiment>/iter_metrics[_runN].png
    base_title, run_suffix = _split_title_and_run_suffix(title)
    iter_dir = Path(out_dir) / f"iter_metrics_{base_title}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = f"_{run_suffix}" if run_suffix else ""
    save_path = iter_dir / f"iter_metrics{file_suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_iter_metrics_all_from_data(
    data: dict,
    out_dir: Path,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
    title: str | None = None,
) -> Path:
    """
    Plot LBFGS inner-iteration metrics for all inits (all runs combined) on a single figure.
    Saved alongside the trainer_hyperparams PNG in the same plots directory.
    """
    all_runs, _ = extract_runs_and_chosen(data)
    inits_with_metrics = [r for r in all_runs if (r.get("lbfgs_inner_metrics") or [])]
    if not inits_with_metrics:
        raise ValueError("No inits with lbfgs_inner_metrics")

    metric_keys: list[str] = []
    for key in ITER_METRIC_KEYS:
        if any(_get_iter_metric_arrays(r).get(key) for r in inits_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No inner-iteration metric data found")

    include_delta_log10 = any(
        len((_get_iter_metric_arrays(r).get("loss", ([], []))[0])) >= 2
        for r in inits_with_metrics
    )
    if include_delta_log10:
        metric_keys.append(DELTA_LOG10_LOSS_KEY)

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        chosen_vals: list[float] = []
        for r in all_runs:
            arrs = _get_iter_metric_arrays(r)
            source_key = "loss" if key == DELTA_LOG10_LOSS_KEY else key
            it_v, val_v = arrs.get(source_key, ([], []))
            if not it_v or not val_v:
                continue
            it_v = np.array(it_v)
            val_v = np.array(val_v)
            if key == DELTA_LOG10_LOSS_KEY:
                it_v, val_v = _compute_delta_log10(it_v, val_v)
                if len(it_v) == 0 or len(val_v) == 0:
                    continue
            if r.get("_best_rrmse"):
                ax.plot(
                    it_v,
                    val_v,
                    c="red",
                    alpha=0.95,
                    linewidth=2.5,
                    zorder=6,
                    label="Best RRMSE (init)" if not labeled_best else None,
                )
                labeled_best = True
                chosen_vals.extend(val_v.tolist())
            elif r.get("_chosen"):
                ax.plot(
                    it_v,
                    val_v,
                    c="C2",
                    alpha=0.85,
                    linewidth=2,
                    zorder=5,
                    label="Chosen init" if not labeled_chosen else None,
                )
                labeled_chosen = True
                chosen_vals.extend(val_v.tolist())
            else:
                ax.plot(
                    it_v,
                    val_v,
                    c="gray",
                    alpha=0.45,
                    linewidth=0.9,
                    zorder=1,
                    label="All inits" if not labeled_all else None,
                )
                labeled_all = True
        ax.set_xlabel("LBFGS iteration")
        if key == DELTA_LOG10_LOSS_KEY:
            ax.set_ylabel("Delta log10 loss")
            ax.set_title("Delta log10 loss")
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5, linestyle="--")
        else:
            ax.set_ylabel(key.replace("_", " "))
            ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # iter2 style: scale y-axis to chosen + best RRMSE inits so convergence is visible
        if chosen_vals:
            arr = np.array(chosen_vals)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                y_min, y_max = float(np.min(arr)), float(np.max(arr))
                pad = max((y_max - y_min) * 0.08, 1e-12)
                ax.set_ylim(y_min - pad, y_max + pad)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])

    title = title or data.get("title", "trainer_analysis")
    fig.suptitle(f"Iteration vs metrics (y: chosen inits) — {title} (all inits)", fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(out_dir)
    base_title, _ = _split_title_and_run_suffix(title)
    iter_dir = out_dir / f"iter_metrics_{base_title}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    save_path = iter_dir / "iter_metrics_all.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_epoch_metrics_from_data(
    data: dict,
    out_dir: Path,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
    title: str | None = None,
) -> Path:
    """
    Plot epoch vs training metrics from in-memory trainer_info data (same structure as saved JSON).
    Use this right after saving the trainer_analysis JSON so epoch-metrics plots are created automatically.
    Returns the path where the figure was saved. Raises ValueError if no runs have epoch_metrics.
    """
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("epoch_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with epoch_metrics in trainer_info data")

    title = title or data.get("title", "trainer_analysis")
    return _plot_epoch_metrics_impl(data, out_dir, title, figsize_per_subplot)


def _plot_epoch_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """Shared implementation: data already loaded, title and out_dir set."""
    all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("epoch_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with epoch_metrics")

    metric_keys = []
    for key in EPOCH_METRIC_KEYS:
        if any(_get_epoch_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No epoch metric data found")

    include_delta_log10 = any(
        len((_get_iter_metric_arrays(r).get("loss", ([], []))[0])) >= 2
        for r in runs_with_metrics
    )
    if include_delta_log10:
        metric_keys.append(DELTA_LOG10_LOSS_KEY)

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        for r in all_runs:
            arrs = _get_epoch_metric_arrays(r)
            ep_v, val_v = arrs.get(key, ([], []))
            if not ep_v or not val_v:
                continue
            ep_v = np.array(ep_v)
            val_v = np.array(val_v)
            if r.get("_best_rrmse"):
                ax.plot(ep_v, val_v, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (init)" if not labeled_best else None)
                labeled_best = True
            elif r.get("_chosen"):
                ax.plot(ep_v, val_v, c="C2", alpha=0.85, linewidth=2, zorder=5,
                        label="Chosen init" if not labeled_chosen else None)
                labeled_chosen = True
            else:
                ax.plot(ep_v, val_v, c="gray", alpha=0.45, linewidth=0.9, zorder=1,
                        label="All inits" if not labeled_all else None)
                labeled_all = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Epoch vs metrics — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"epoch_metrics_{title}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_epoch_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Load a GP_Trainer_Analysis.json, extract epoch_metrics from runs, and plot epoch vs each metric.
    Returns the path where the figure was saved.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    return _plot_epoch_metrics_impl(data, Path(out_dir), title, figsize_per_subplot)


def _plot_iter_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """Shared implementation for inner-iteration (lbfgs_iter) metrics."""
    all_runs, _ = extract_runs_and_chosen(data)
    return _plot_iter_metrics_for_runs(all_runs, out_dir, title, figsize_per_subplot)


def plot_iter_metrics_from_data(
    data: dict,
    out_dir: Path,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
    title: str | None = None,
) -> Path:
    """
    Plot iteration vs metrics from in-memory trainer_info data (same structure as saved JSON).
    Produces both iter_metrics_*.png (full y-range) and iter2_metrics_*.png (y scaled to chosen inits).
    Use this right after saving the trainer_analysis JSON. Returns the path to the iter_metrics figure.
    Raises ValueError if no runs have lbfgs_inner_metrics.
    """
    title = title or data.get("title", "trainer_analysis")
    out_dir = Path(out_dir)
    all_runs, _ = extract_runs_and_chosen(data)
    if not all_runs:
        raise ValueError("No runs in trainer_info data")

    # Group by _group_key (run_1, run_2, ...) so we get one plot per run with correct naming
    group_keys = sorted(
        {r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") for r in all_runs},
        key=lambda k: (0 if k.startswith("run_") else 1, int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0),
    )
    first_path: Path | None = None
    for group_key in group_keys:
        runs_group = [r for r in all_runs if r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") == group_key]
        if not any(r.get("lbfgs_inner_metrics") for r in runs_group):
            continue
        suffix = _group_key_to_suffix(group_key)
        group_title = f"{title}_{suffix}" if suffix else title
        path_iter = _plot_iter_metrics_for_runs(runs_group, out_dir, group_title, figsize_per_subplot)
        _plot_iter2_metrics_impl(
            {"trainer_info": None, "title": group_title, "all_runs_override": runs_group},
            out_dir,
            group_title,
            figsize_per_subplot,
        )
        if first_path is None:
            first_path = path_iter

    # Also create a single figure with all runs combined in one plot,
    # saved alongside the trainer_hyperparams PNG in the same plots directory.
    try:
        plot_iter_metrics_all_from_data(
            data=data,
            out_dir=out_dir,
            figsize_per_subplot=figsize_per_subplot,
            title=title,
        )
    except ValueError:
        # If for some reason there are no lbfgs metrics, don't fail the per-run plots.
        pass

    if first_path is None:
        raise ValueError("No runs with lbfgs_inner_metrics in trainer_info data")
    return first_path


def plot_iter_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Load a GP_Trainer_Analysis.json, extract lbfgs_inner_metrics from runs, and
    plot iteration vs each metric.
    Returns the path where the figure was saved.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    out_dir = Path(out_dir)
    all_runs, _ = extract_runs_and_chosen(data)
    if not all_runs:
        raise ValueError("No runs in trainer_info data")

    group_keys = sorted(
        {r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") for r in all_runs},
        key=lambda k: (0 if k.startswith("run_") else 1, int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0),
    )
    first_path: Path | None = None
    for group_key in group_keys:
        runs_group = [r for r in all_runs if r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") == group_key]
        if not any(r.get("lbfgs_inner_metrics") for r in runs_group):
            continue
        suffix = _group_key_to_suffix(group_key)
        group_title = f"{title}_{suffix}" if suffix else title
        path_iter = _plot_iter_metrics_for_runs(runs_group, out_dir, group_title, figsize_per_subplot)
        if first_path is None:
            first_path = path_iter

    if first_path is None:
        raise ValueError("No runs with lbfgs_inner_metrics")
    return first_path


def _plot_iter2_metrics_impl(
    data: dict,
    out_dir: Path,
    title: str,
    figsize_per_subplot: tuple[float, float],
) -> Path:
    """
    Same as iter_metrics but y-axis scaled to min/max of chosen inits (and best RRMSE init)
    so convergence is visible instead of squashed by high-loss runs.
    """
    # Allow callers to override the inits list (used for per-run plots).
    if "all_runs_override" in data:
        all_runs = data["all_runs_override"]
    else:
        all_runs, _ = extract_runs_and_chosen(data)
    runs_with_metrics = [r for r in all_runs if (r.get("lbfgs_inner_metrics") or [])]
    if not runs_with_metrics:
        raise ValueError("No runs with lbfgs_inner_metrics")

    metric_keys: list[str] = []
    for key in ITER_METRIC_KEYS:
        if any(_get_iter_metric_arrays(r).get(key) for r in runs_with_metrics):
            metric_keys.append(key)
    if not metric_keys:
        raise ValueError("No inner-iteration metric data found")

    n = len(metric_keys)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flat

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        labeled_all = False
        labeled_chosen = False
        labeled_best = False
        chosen_vals: list[float] = []
        for r in all_runs:
            arrs = _get_iter_metric_arrays(r)
            source_key = "loss" if key == DELTA_LOG10_LOSS_KEY else key
            it_v, val_v = arrs.get(source_key, ([], []))
            if not it_v or not val_v:
                continue
            it_v = np.array(it_v)
            val_v = np.array(val_v)
            if key == DELTA_LOG10_LOSS_KEY:
                it_v, val_v = _compute_delta_log10(it_v, val_v)
                if len(it_v) == 0 or len(val_v) == 0:
                    continue
            if r.get("_best_rrmse"):
                ax.plot(it_v, val_v, c="red", alpha=0.95, linewidth=2.5, zorder=6,
                        label="Best RRMSE (init)" if not labeled_best else None)
                labeled_best = True
                chosen_vals.extend(val_v.tolist())
            elif r.get("_chosen"):
                ax.plot(it_v, val_v, c="C2", alpha=0.85, linewidth=2, zorder=5,
                        label="Chosen init" if not labeled_chosen else None)
                labeled_chosen = True
                chosen_vals.extend(val_v.tolist())
            else:
                ax.plot(it_v, val_v, c="gray", alpha=0.45, linewidth=0.9, zorder=1,
                        label="All inits" if not labeled_all else None)
                labeled_all = True
        ax.set_xlabel("LBFGS iteration")
        if key == DELTA_LOG10_LOSS_KEY:
            ax.set_ylabel("Delta log10 loss")
            ax.set_title("Delta log10 loss")
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5, linestyle="--")
        else:
            ax.set_ylabel(key.replace("_", " "))
            ax.set_title(key.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Scale y-axis to chosen (and best RRMSE) runs only
        if chosen_vals:
            arr = np.array(chosen_vals)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                y_min, y_max = float(np.min(arr)), float(np.max(arr))
                pad = max((y_max - y_min) * 0.08, 1e-12)
                ax.set_ylim(y_min - pad, y_max + pad)

    for j in range(len(metric_keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Iteration vs metrics (y: chosen inits) — {title}", fontsize=11, y=1.02)
    fig.tight_layout()

    # Save iter2-metrics plots into the same per-experiment folder as iter-metrics:
    #   <out_dir>/iter_metrics_<experiment>/iter2_metrics[_runN].png
    base_title, run_suffix = _split_title_and_run_suffix(title)
    iter_dir = Path(out_dir) / f"iter_metrics_{base_title}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = f"_{run_suffix}" if run_suffix else ""
    save_path = iter_dir / f"iter2_metrics{file_suffix}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_iter2_metrics(
    json_path: Path,
    out_dir: Path | None = None,
    figsize_per_subplot: tuple[float, float] = (5, 3.5),
) -> Path:
    """
    Same as plot_iter_metrics but y-axis is scaled to min/max of chosen inits (and best RRMSE init)
    so loss convergence is visible.
    """
    data = load_trainer_analysis(json_path)
    if out_dir is None:
        out_dir = json_path.parent / "plots"
    title = data.get("title", json_path.stem.replace("_GP_Trainer_Analysis", ""))
    out_dir = Path(out_dir)
    all_runs, _ = extract_runs_and_chosen(data)
    if not all_runs:
        raise ValueError("No runs in trainer_info data")

    group_keys = sorted(
        {r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") for r in all_runs},
        key=lambda k: (0 if k.startswith("run_") else 1, int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0),
    )
    first_path: Path | None = None
    for group_key in group_keys:
        runs_group = [r for r in all_runs if r.get("_group_key", f"run_{r.get('_run_idx', 0) + 1}") == group_key]
        if not any(r.get("lbfgs_inner_metrics") for r in runs_group):
            continue
        suffix = _group_key_to_suffix(group_key)
        group_title = f"{title}_{suffix}" if suffix else title
        path_iter2 = _plot_iter2_metrics_impl(
            {"trainer_info": None, "title": group_title, "all_runs_override": runs_group},
            out_dir,
            group_title,
            figsize_per_subplot,
        )
        if first_path is None:
            first_path = path_iter2

    if first_path is None:
        raise ValueError("No runs with lbfgs_inner_metrics")
    return first_path


def main() -> None:
    import argparse
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "results_v2.0" / "NLL3"
    parser = argparse.ArgumentParser(
        description="Plot epoch vs training metrics (loss, NIS, LOO_NLL, KF, MSE, R2) from trainer_analysis JSON.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=f"Path to a *_GP_Trainer_Analysis.json file or a directory to search (default: {default_root}).",
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=None,
        help="Directory to save the figure (single-file mode only). Default: <json_dir>/plots",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve() if args.path else default_root
    if not path.exists():
        print(f"Path not found: {path}")
        return

    if path.is_dir():
        jsons = find_trainer_analysis_jsons(path)
        if not jsons:
            print(f"No *_GP_Trainer_Analysis.json files found under {path}")
            return
        print(f"Found {len(jsons)} trainer analysis JSON(s) under {path}\n")
        for json_path in sorted(jsons):
            out_dir = json_path.parent / "plots"
            try:
                save_path = plot_epoch_metrics(json_path, out_dir=out_dir)
                print(f"  Saved: {save_path}")
            except ValueError as e:
                print(f"  Skip {json_path.name}: {e}")
        print("\nDone.")
    else:
        if not path.is_file():
            print(f"Not a file: {path}")
            return
        out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        save_path = plot_epoch_metrics(path, out_dir=out_dir)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
