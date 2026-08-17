import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

# from sklearn.metrics import mean_absolute_error, r2_score


def compute_metrics(y_true, y_hat, output_std=None, start_time=None, training_time=None, prediction_time=None):
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)

    Returns:
        dict: Dictionary with computed metrics including time information
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Handle time metrics
    if training_time is not None and prediction_time is not None:
        # New approach: separate training and prediction times
        total_time = training_time + prediction_time
        metrics = {
            "Total_Time": total_time,
            "Training_Time": training_time,
            "Prediction_Time": prediction_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    elif start_time is not None:
        # Legacy approach: single start_time
        elapsed_time = time.time() - start_time
        metrics = {
            "Time": elapsed_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    else:
        # No time information
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics

    return metrics


def analyze_metrics(metrics_list, print_summary: bool = False, label: str = None, title: str = None):
    """
    Summarize metrics across seeds for RRMSE and NIS.

    Args:
        metrics_list: list of dicts, each containing metric values for a seed

    Returns:
        dict with per-metric summary: {metric: {mean, std, median, min, max}}
    """
    import numpy as np
    import pandas as pd

    if metrics_list is None or len(metrics_list) == 0:
        return {}

    df = pd.DataFrame(metrics_list)

    # Mean/std for all available metric columns
    # Detailed stats for specific metrics
    detailed = {}
    for m in ["RRMSE", "NIS", "Time", "Total_Time", "Training_Time", "Prediction_Time"]:
        if m in df.columns:
            vals = df[m].dropna().values
            if len(vals) == 0:
                continue
            vals = vals.astype(float)
            detailed[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": int(len(vals)),
            }

    if print_summary and len(detailed) > 0:
        header = f"{label} summary" if label else "Summary"
        label_print = label or "Summary"

        if title:
            print(f"\n{label_print} over {len(metrics_list)} seeds for {title} (RRMSE, NIS):")
        else:
            print(f"\n{label_print} over {len(metrics_list)} seeds (RRMSE, NIS):")

        for m, s in detailed.items():
            print(
                f"  {m}: median={s['median']:.6f} | min={s['min']:.6f} | max={s['max']:.6f} | "
                f"mean={s['mean']:.6f} ± {s['std']:.6f} (n={s['count']})"
            )

    return detailed


def plot_metrics(*args, labels: list = None, title: str = None, save_path: str = None):
    """
    Plot per-seed metric VALUES (not aggregates) for multiple runs as violin plots.

    Args:
        metrics_lists: list of lists, where each inner list is a metrics_list
                       (the same structure you pass to analyze_metrics), i.e.,
                       a list of dicts with keys like 'RRMSE', 'NIS', 'Time'.
        labels: optional list of names for each metrics_list; defaults to
                ["run_0", ...].

    Produces two figures (RRMSE, NIS) with one violin per input label.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Normalize inputs: allow plot_metric_values(run1, run2, ...) or plot_metric_values([run1, run2, ...])
    if len(args) == 1 and isinstance(args[0], list) and (len(args[0]) == 0 or isinstance(args[0][0], (dict, list))):
        metrics_lists = args[0]
    else:
        metrics_lists = list(args)

    if labels is None:
        labels = [f"run_{i}" for i in range(len(metrics_lists))]

    def extract(vals_list, key):
        out = []
        for ml in metrics_lists:
            # ml should be list[dict]
            if not isinstance(ml, list):
                out.append(np.array([], dtype=float))
                continue
            arr = [d[key] for d in ml if isinstance(d, dict) and key in d and d[key] is not None]
            out.append(np.array(arr, dtype=float) if len(arr) > 0 else np.array([], dtype=float))
        return out

    # Determine a representative seed count (use min across lists to be safe)
    seed_counts = []
    for ml in metrics_lists:
        seed_counts.append(len(ml) if isinstance(ml, list) else 0)
    n_seeds = min(seed_counts) if len(seed_counts) > 0 else 0

    for metric in ["RRMSE", "NIS"]:
        data = extract(metrics_lists, metric)
        fig, ax = plt.subplots(figsize=(7, 4))
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#888888")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        # Overlay mean (blue) and median (red) lines
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(mean_v, i - 0.25, i + 0.25, colors="blue", linewidth=2)
            ax.hlines(med_v, i - 0.25, i + 0.25, colors="red", linewidth=2)

        # Legend: blue = mean, red = median
        try:
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], color="blue", lw=2, label="Mean"),
                Line2D([0], [0], color="red", lw=2, label="Median"),
            ]
            ax.legend(handles=legend_handles, loc="upper right", frameon=False)
        except Exception:
            pass

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        if title:
            try:
                fig.suptitle(title)
                ax.set_title(f"{metric} per-seed distribution (n={n_seeds})")
            except Exception:
                ax.set_title(f"{metric} per-seed distribution - {title} (n={n_seeds})")
        else:
            ax.set_title(f"{metric} per-seed distribution (n={n_seeds})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()

        # Save if a directory is provided
        if save_path is not None:
            from pathlib import Path

            p = Path(save_path)
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            fname = f"{metric.lower()}" + (f"_{title}" if title else "") + ".png"
            try:
                fig.savefig(str(p / fname), dpi=300, bbox_inches="tight")
            except Exception:
                pass
        else:
            plt.show()


def compute_per_source_metrics(
    y_true, y_hat, output_std, X_test, source_columns, start_time=None, training_time=None, prediction_time=None
):
    """
    Compute metrics for each source separately.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        X_test: Test features (2D array) containing source information
        source_columns: Either a single column index (int) or list of column indices for source identification
        start_time: Start time for timing (optional, deprecated - use training_time and prediction_time instead)
        training_time: Training time in seconds (optional)
        prediction_time: Prediction time in seconds (optional)

    Returns:
        dict: Dictionary with overall metrics and per-source metrics including time information
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Store the overall standard deviation for consistent normalization
    overall_std = y_true.std()

    # Handle source_columns parameter
    if isinstance(source_columns, int):
        # Single column case - filter by value in that column
        source_values = np.unique(X_test[:, source_columns])
        num_sources = len(source_values)
        source_indices = {}
        for i, val in enumerate(source_values):
            source_indices[f"source_{i}"] = X_test[:, source_columns] == val
    else:
        # Multiple columns case - one-hot encoded sources
        # Convert to list if it's a numpy array
        if isinstance(source_columns, np.ndarray):
            source_columns = source_columns.tolist()

        num_sources = len(source_columns)
        source_indices = {}
        for i in range(num_sources):
            source_indices[f"source_{i}"] = X_test[:, source_columns[i]] == 1

    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_hat, output_std, start_time, training_time, prediction_time)
    # Add sample size to overall metrics (as integer)
    overall_metrics["num_samples"] = int(len(y_true))

    # Compute per-source metrics
    per_source_metrics = {}
    for source_name, source_mask in source_indices.items():
        if np.sum(source_mask) > 0:  # Only compute if source has data
            source_y_true = y_true[source_mask]
            source_y_hat = y_hat[source_mask]
            source_output_std = output_std[source_mask] if output_std is not None else None

            # Compute source metrics with consistent normalization
            source_rmse = np.sqrt(mean_squared_error(source_y_true, source_y_hat))
            source_rrmse = source_rmse / overall_std  # Use overall std for consistency

            # Handle time metrics for per-source
            if training_time is not None and prediction_time is not None:
                # New approach: separate training and prediction times
                total_time = training_time + prediction_time
                source_metrics = {
                    "Total_Time": total_time,
                    "Training_Time": training_time,
                    "Prediction_Time": prediction_time,
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            elif start_time is not None:
                # Legacy approach: single start_time
                elapsed_time = time.time() - start_time
                source_metrics = {
                    "Time": elapsed_time,
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }
            else:
                # No time information
                source_metrics = {
                    "RRMSE": source_rrmse,
                    "RMSE": source_rmse,
                    "MSE": mean_squared_error(source_y_true, source_y_hat),
                    "num_samples": int(len(source_y_true)),  # Number of predictions for this source
                }

            # Add NIS if output_std is provided
            if source_output_std is not None:
                z = 1.96
                L = source_y_hat - z * source_output_std
                U = source_y_hat + z * source_output_std
                width = U - L
                below = (L - source_y_true) * (source_y_true < L)
                above = (source_y_true - U) * (source_y_true > U)
                interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
                source_nis = interval_score.mean() / source_y_true.std()  # Use per-source target std for NIS
                source_metrics["NIS"] = source_nis

            per_source_metrics[source_name] = source_metrics

    # Combine overall and per-source metrics
    all_metrics = {"overall": overall_metrics, "per_source": per_source_metrics, "num_sources": num_sources}

    return all_metrics
