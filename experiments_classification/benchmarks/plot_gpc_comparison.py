# Plots the *_summary.csv files written by the run scripts
# Run after the run scripts

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Non-interactive backend: these scripts only ever write files.
matplotlib.use("Agg")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from result_paths import new_results

_RESULTS = str(new_results("classification"))

# Missing files are skipped
SUMMARY_CSVS = [
    (os.path.join(_RESULTS, "electrical_grid", "electrical_grid_summary.csv"), "electrical_grid"),
    (os.path.join(_RESULTS, "truss_6d", "truss_6d_summary.csv"), "truss_6d"),
    (os.path.join(_RESULTS, "stellar", "stellar_summary.csv"), "stellar"),
    (os.path.join(_RESULTS, "steel_plates", "steel_plates_summary.csv"), "steel_plates"),
]

OUTPUT_DIR = os.path.join(_RESULTS, "plots_per_dataset")

# Set False to skip the per-dataset figure files and keep only the combined grid.
WRITE_PER_DATASET = True

DATASET_NAMES = {
    "electrical_grid": "Electrical Grid Stability",
    "truss_6d":        "Truss 6D",
    "stellar":         "Stellar",
    "steel_plates":    "Steel Plates Faults",
}

# None plots everything
INCLUDE_SUBSTRINGS = None
EXCLUDE_SUBSTRINGS = None

# Shaded mean +/- std bands around each line
SHOW_VARIANCE_BAND = False

# Multi-panel figure for the paper: metrics as rows, datasets as columns
PAPER_EXPORT = True
PAPER_DIR = os.path.join(_RESULTS, "summary")
PAPER_DATASETS = ["electrical_grid", "truss_6d", "stellar", "steel_plates"]
PAPER_PANEL_W, PAPER_PANEL_H = 3.4, 2.0
PAPER_TICK_FS, PAPER_LABEL_FS, PAPER_LEGEND_FS = 13, 13, 12

# (mean column, std column, full name, short axis label, filename suffix, log y)
METRICS = [
    ("accuracy_mean", "accuracy_std",
     "Accuracy", "Acc", "accuracy", False),
    ("ece_mean", "ece_std",
     "ECE", "ECE", "ece", False),
    ("final_train_nll_mean", "final_train_nll_std",
     "Final training NLL", r"$\mathcal{L}$", "train_loss", False),
    ("train_time_mean", "train_time_std",
     "Training time (s)", "Time (s)", "training_time", True),
]

# Row order in the paper grid.
PAPER_METRICS = ["train_loss", "accuracy", "ece"]

CONFIG_COLUMNS = [
    "kernel", "alpha_epsilon", "num_inits", "num_mc_samples",
    "adam_lr", "adam_epochs", "val_fraction", "val_patience", "val_check_every",
]

AXIS_ABBREV = {
    "kernel":          "",
    "alpha_epsilon":   "a=",
    "num_inits":       "inits=",
    "num_mc_samples":  "mc=",
    "adam_lr":         "lr=",
    "adam_epochs":     "ep=",
    "val_fraction":    "vf=",
    "val_patience":    "pat=",
    "val_check_every": "chk=",
}

KERNEL_DISPLAY = {
    "gaussian":   "RBF",
    "matern_0.5": "Matérn 0.5",
    "powerexp":   "PowerExp",
}

BASELINE_STYLES = {
    "TabPFN v2.0": ("#d62728", "D", "--"),
    "TabPFN v2.5": ("#9467bd", "v", "--"),
}

COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#8c564b", "#17becf",
          "#e377c2", "#7f7f7f", "#bcbd22"]
MARKERS = ["^", "o", "s", "P", "X", "*", "d"]
LINESTYLES = ["-", "--", "-.", ":"]


def load_summary(path, dataset):
    if not os.path.exists(path):
        print(f"[skip] {dataset}: {path} not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[skip] {dataset}: {path} is empty")
        return None
    for col in CONFIG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["dataset"] = dataset
    return df


# Baseline rows leave the config columns blank, so they are excluded before
# counting.
def active_axes(df):
    gp = df[~df["model"].isin(BASELINE_STYLES)]
    if gp.empty:
        return []
    axes = []
    for col in CONFIG_COLUMNS:
        values = gp[col].dropna().unique()
        if len(values) > 1:
            axes.append(col)
    return axes


def _format_value(col, value):
    if col == "kernel":
        return KERNEL_DISPLAY.get(value, str(value))
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{AXIS_ABBREV.get(col, col + '=')}{value}"


def make_label(row, axes):
    if row["model"] in BASELINE_STYLES:
        return row["model"]
    if not axes:
        return "GP+"
    parts = [_format_value(col, row[col]) for col in axes]
    return "GP+ (" + ", ".join(parts) + ")"


def _value_index(values):
    values = list(values)
    try:
        ordered = sorted(values, key=float)
    except (TypeError, ValueError):
        ordered = sorted(values, key=str)
    return {v: i for i, v in enumerate(ordered)}


def assign_styles(df, axes):
    styles = {}
    for name, style in BASELINE_STYLES.items():
        styles[name] = style

    gp = df[~df["model"].isin(BASELINE_STYLES)]
    if gp.empty:
        return styles

    channels = [_value_index(gp[col].dropna().unique()) for col in axes[:3]]

    for label, grp in gp.groupby("label"):
        row = grp.iloc[0]
        color, marker, linestyle = COLORS[0], MARKERS[0], LINESTYLES[0]
        if len(channels) >= 1:
            color = COLORS[channels[0].get(row[axes[0]], 0) % len(COLORS)]
        if len(channels) >= 2:
            marker = MARKERS[channels[1].get(row[axes[1]], 0) % len(MARKERS)]
        if len(channels) >= 3:
            linestyle = LINESTYLES[channels[2].get(row[axes[2]], 0) % len(LINESTYLES)]
        styles[label] = (color, marker, linestyle)
    return styles


def apply_filters(df):
    if INCLUDE_SUBSTRINGS:
        keep = df["label"].apply(lambda s: any(sub in s for sub in INCLUDE_SUBSTRINGS))
        df = df[keep]
    if EXCLUDE_SUBSTRINGS:
        drop = df["label"].apply(lambda s: any(sub in s for sub in EXCLUDE_SUBSTRINGS))
        df = df[~drop]
    return df


def _legend_order(df):
    labels = list(dict.fromkeys(df["label"]))
    gp = sorted(lbl for lbl in labels if lbl not in BASELINE_STYLES)
    base = [lbl for lbl in labels if lbl in BASELINE_STYLES]
    return gp + base


def draw_metric(ax, df, styles, mean_col, std_col, log_y):
    handles = []
    for label in _legend_order(df):
        grp = df[df["label"] == label].sort_values("train_size")
        if mean_col not in grp or grp[mean_col].isna().all():
            continue
        color, marker, linestyle = styles.get(label, (COLORS[0], "o", "-"))
        line, = ax.plot(grp["train_size"], grp[mean_col],
                        color=color, marker=marker, linestyle=linestyle,
                        markersize=5, linewidth=1.6, label=label)
        handles.append(line)
        if SHOW_VARIANCE_BAND and std_col in grp:
            lo = grp[mean_col] - grp[std_col]
            hi = grp[mean_col] + grp[std_col]
            ax.fill_between(grp["train_size"], lo, hi, color=color, alpha=0.12,
                            linewidth=0)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, linewidth=0.6)
    return handles


def plot_dataset(df, dataset, styles):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    written = []
    for mean_col, std_col, full_name, _short, suffix, log_y in METRICS:
        if mean_col not in df.columns or df[mean_col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        handles = draw_metric(ax, df, styles, mean_col, std_col, log_y)
        if not handles:
            plt.close(fig)
            continue
        ax.set_xlabel("Train size $N$")
        ax.set_ylabel(full_name)
        ax.set_title(f"{DATASET_NAMES.get(dataset, dataset)} — {full_name}")
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=9)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{dataset}_{suffix}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_paper_grid(frames, styles):
    datasets = [d for d in PAPER_DATASETS if d in frames]
    metrics = [m for m in METRICS if m[4] in PAPER_METRICS]
    metrics.sort(key=lambda m: PAPER_METRICS.index(m[4]))
    if not datasets or not metrics:
        print("[paper] nothing to draw")
        return

    os.makedirs(PAPER_DIR, exist_ok=True)
    n_rows, n_cols = len(metrics), len(datasets)
    fig, axarr = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(PAPER_PANEL_W * n_cols, PAPER_PANEL_H * n_rows))

    all_handles = {}
    for r, (mean_col, std_col, _full, short, _suffix, log_y) in enumerate(metrics):
        for c, dataset in enumerate(datasets):
            ax = axarr[r][c]
            handles = draw_metric(ax, frames[dataset], styles, mean_col, std_col, log_y)
            for h in handles:
                all_handles.setdefault(h.get_label(), h)
            ax.tick_params(labelsize=PAPER_TICK_FS)
            if c == 0:
                ax.set_ylabel(short, fontsize=PAPER_LABEL_FS)
            if r == n_rows - 1:
                ax.set_xlabel("$N$", fontsize=PAPER_LABEL_FS)

    fig.tight_layout()
    grid_path = os.path.join(PAPER_DIR, "classification_grid.pdf")
    fig.savefig(grid_path, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER_DIR, "classification_final.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ordered = [all_handles[k] for k in sorted(all_handles,
                                              key=lambda k: (k in BASELINE_STYLES, k))]
    leg_fig = plt.figure(figsize=(PAPER_PANEL_W * n_cols, 0.6))
    leg_fig.legend(handles=ordered, loc="center", ncol=min(len(ordered), 5),
                   frameon=False, fontsize=PAPER_LEGEND_FS)
    legend_path = os.path.join(PAPER_DIR, "classification_legend.pdf")
    leg_fig.savefig(legend_path, bbox_inches="tight")
    plt.close(leg_fig)

    print(f"\nPaper grid saved to {grid_path}")
    print(f"Paper legend saved to {legend_path}")


def main():
    loaded = {}
    for path, dataset in SUMMARY_CSVS:
        df = load_summary(path, dataset)
        if df is not None:
            loaded[dataset] = df

    if not loaded:
        print("No summary CSVs found — run the run_*.py scripts first.")
        return

    # Pooled so a model keeps the same color and marker in every figure
    pooled = pd.concat(loaded.values(), ignore_index=True)
    axes = active_axes(pooled)
    pooled["label"] = pooled.apply(lambda r: make_label(r, axes), axis=1)
    pooled = apply_filters(pooled)
    styles = assign_styles(pooled, axes)

    if axes:
        print(f"\nVarying config axes: {', '.join(axes)}")
    else:
        print("\nNo config axes vary — one line per model.")

    n_lines = pooled["label"].nunique()
    if n_lines > 12:
        print(f"[note] {n_lines} distinct lines per axes. Color, marker and "
              f"linestyle carry the first three axes; beyond that the legend is "
              f"the only channel left. Set INCLUDE_SUBSTRINGS to isolate a "
              f"slice (e.g. [\"RBF\"]) if the figures are too dense.")

    frames = {}
    for dataset in loaded:
        sub = pooled[pooled["dataset"] == dataset]
        if sub.empty:
            continue
        frames[dataset] = sub
        if WRITE_PER_DATASET:
            for path in plot_dataset(sub, dataset, styles):
                print(f"Saved {path}")

    if PAPER_EXPORT:
        plot_paper_grid(frames, styles)


if __name__ == "__main__":
    main()
