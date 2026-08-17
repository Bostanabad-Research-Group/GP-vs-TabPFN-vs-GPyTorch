"""
Create violin plots for RRMSE and NIS across GP+, TabPFN v2.5, TabPFN v2, and GPytorch runs.

Inputs (defaults to your current dirs):
- Results directory: C:/Users/forty/tyler_gpplus/gp-private/results_final/1_11

TabPFN version detection:
- TabPFN v2.5: Files WITHOUT "gpytorch" in the filename
- TabPFN v2: Files WITH "gpytorch" in the filename

Special handling for TabPFN v2.5 files run separately:
- Files starting with "pfn" (case-insensitive) are treated as TabPFN v2.5 results
  that were run separately. These files are processed and their TabPFN data is
  extracted and placed in the appropriate plots based on Dx (dimensions), Dn
  (train size), and noise levels parsed from the filename.

Per-run data is extracted from the `metrics` arrays in each JSON. Filenames
are parsed to extract noise levels (noiseTest / noiseTrain) and dimension.

Output:
- One violin plot per problem for RRMSE and NIS, saved to an output directory
  (default: plots_violin). X-axis = noiseTrain (falls back to noiseTest if
  missing), hue = model family (gpplus, tabpfn_v2.5, tabpfn_v2, gpytorch).
- Only two noise levels are included: 0.005 and 0.05
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _infer_problem_from_stem(stem: str) -> str:
    """Infer problem name from a result filename stem (e.g. gp_Zakharov_20Dx_...)."""
    stem_l = stem.lower()
    # Longest-first so e.g. dixon_price wins over price-like fragments.
    candidates = (
        "dixon_price",
        "dixonprice",
        "rosenbrock",
        "rastrigin",
        "griewank",
        "zakharov",
        "borehole",
        "buckling",
        "ackley",
        "wing",
        "m2ax",
    )
    for name in candidates:
        if name in stem_l:
            return "dixon_price" if name == "dixonprice" else name
    return ""


def iter_json_files(base_dir: Optional[Path], problems: Optional[set[str]]) -> Iterable[Path]:
    if base_dir is None or not base_dir.exists():
        return
    problems_lower = {p.lower() for p in problems} if problems else None
    for path in base_dir.rglob("*.json"):
        # Skip trainer-analysis dumps (no per-run gp_data/tabpfn_data metrics).
        if "trainer_analysis" in path.parts:
            continue
        rel = path.relative_to(base_dir)
        if problems_lower:
            if not rel.parts:
                continue
            # Case-insensitive matching on problem folder, with flat-dir fallback
            # (JSONs live directly under the result root, problem only in filename).
            problem_lower = rel.parts[0].lower()
            if problem_lower.endswith(".json"):
                inferred = _infer_problem_from_stem(path.stem)
                if inferred not in problems_lower:
                    continue
            elif problem_lower not in problems_lower:
                continue
        yield path


def _try_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


def same_resolved_dir(a: Optional[Path], b: Optional[Path]) -> bool:
    if a is None or b is None:
        return False
    return _try_resolve(a) == _try_resolve(b)


def path_is_under(path: Path, base: Path) -> bool:
    try:
        _try_resolve(path).relative_to(_try_resolve(base))
        return True
    except ValueError:
        return False


def _skip_gp_pass_for_tabpfn_tree(
    path: Path,
    tabpfn_dir: Optional[Path],
    gp_dir: Path,
    gpytorch_dir: Path,
) -> bool:
    """Skip GP pass only when JSONs live in a TabPFN tree separate from both GP roots."""
    if tabpfn_dir is None:
        return False
    if same_resolved_dir(tabpfn_dir, gp_dir) or same_resolved_dir(tabpfn_dir, gpytorch_dir):
        return False
    return path_is_under(path, tabpfn_dir)


def _base_dir_for_meta(path: Path, gp_dir: Path, gpytorch_dir: Path) -> Path:
    """Pick gp_dir vs gpytorch_dir so relative_to matches on-disk layout (handles rel vs abs paths)."""
    if same_resolved_dir(gp_dir, gpytorch_dir):
        return gp_dir
    if path_is_under(path, gpytorch_dir):
        return gpytorch_dir
    if path_is_under(path, gp_dir):
        return gp_dir
    return gp_dir


def parse_meta_from_filename(path: Path, base_dir: Path) -> Dict[str, str]:
    rel_parts = path.relative_to(base_dir).parts
    problem = rel_parts[0].lower() if rel_parts else ""  # Normalize to lowercase
    # Flat result dirs (e.g. results/zakharov_log_scale/*.json): problem is in the stem.
    if not problem or problem.endswith(".json"):
        problem = _infer_problem_from_stem(path.stem)
    date = rel_parts[1] if len(rel_parts) > 1 and not rel_parts[0].lower().endswith(".json") else ""
    stem = path.stem

    # Try standard noise pattern first
    noise_match = re.search(r"noiseTest(?P<noiseTest>[0-9.]+)_noiseTrain(?P<noiseTrain>[0-9.]+)", stem)
    # Look for dimension pattern: {number}Dn (e.g., 80Dn means N = 80Dx)
    dim_match = re.search(r"([0-9]+)Dn", stem)
    # Also try the old pattern for backward compatibility
    if not dim_match:
        dim_match = re.search(r"_([0-9]+D)_", stem)
    # Extract xDim from pattern like "5Dx" or "40Dx" (capital D, lowercase x)
    xdim_match = re.search(r"([0-9]+)Dx", stem)
    # Also try the old pattern for backward compatibility
    if not xdim_match:
        xdim_match = re.search(r"([0-9]+)xdim", stem)
    
    # For M2AX and similar problems, try to extract test_size as noise label
    test_size_match = re.search(r"([0-9.]+)tsize", stem)
    
    noise_test = noise_match.group("noiseTest") if noise_match else ""
    noise_train = noise_match.group("noiseTrain") if noise_match else ""
    
    # If no noise pattern found but test_size found, use test_size as noise label
    if not noise_test and not noise_train and test_size_match:
        noise_test = test_size_match.group(1)
        noise_train = test_size_match.group(1)

    # Extract xdim from filename, or use problem-specific defaults
    xdim = xdim_match.group(1) if xdim_match else ""
    
    # If xdim not found in filename, use problem-specific defaults
    if not xdim:
        if problem == "wing":
            xdim = "10"
        elif problem == "buckling":
            xdim = "4"
        elif problem == "m2ax":
            xdim = "3"
        elif problem == "borehole":
            xdim = "8"

    # Extract dimension: if pattern is {number}Dn, extract just the number and add "D"
    dim = ""
    if dim_match:
        dim_value = dim_match.group(1)
        # If it's the new pattern (numberDn), use it as-is but format as "numberD"
        # If it's the old pattern (numberD), use it as-is
        if "Dn" in stem[dim_match.start():dim_match.end()]:
            dim = f"{dim_value}D"
        else:
            dim = dim_match.group(1)
    
    return {
        "problem": problem,
        "date": date,
        "dim": dim,
        "xdim": xdim,
        "noiseTest": noise_test,
        "noiseTrain": noise_train,
    }


def extract_rows_from_section(
    data: Dict,
    section: str,
    model_family: str,
    meta: Dict[str, str],
) -> List[Dict]:
    rows: List[Dict] = []
    section_data = data.get(section, {})
    metrics_list = section_data.get("metrics")
    if not isinstance(metrics_list, list):
        return rows
    for m in metrics_list:
        if not isinstance(m, dict):
            continue
        # Fallbacks for TabPFN JSONs that store inference time under "Time"
        pred_time = m.get("Prediction_Time")
        if pred_time is None and model_family.startswith("tabpfn"):
            pred_time = m.get("Time")

        row = {
            "model_family": model_family,
            "problem": meta["problem"],
            "date": meta["date"],
            "dim": meta["dim"],
            "xdim": meta.get("xdim", ""),
            "noiseTest": meta["noiseTest"],
            "noiseTrain": meta["noiseTrain"],
            "RRMSE": m.get("RRMSE"),
            "RMSE": m.get("RMSE"),
            "NIS": m.get("NIS"),
            "NCRPS": m.get("NCRPS"),
            "Training_Time": m.get("Training_Time"),
            "Total_Time": m.get("Total_Time"),
            "Prediction_Time": pred_time,
            "source_file": str(meta.get("source_file", "")),
        }
        rows.append(row)
    return rows


def collect_per_run_rows(
    gp_dir: Path,
    gpytorch_dir: Path,
    seek_dir: Optional[Path],
    tabpfn_v25_dir: Optional[Path],
    tabpfn_v2_dir: Optional[Path],
    problems: Optional[set[str]],
    extra_named_gp: Optional[List[Tuple[str, Path]]] = None,
    noise_levels: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    gpytorch_dir = gpytorch_dir or gp_dir

    # Process all files from the results directory
    # Files are processed once, and we determine model types based on filename
    all_paths = set(iter_json_files(gp_dir, problems))
    if not same_resolved_dir(gpytorch_dir, gp_dir):
        all_paths.update(iter_json_files(gpytorch_dir, problems))
    
    # Process regular files (GP+ and GPytorch only, no TabPFN from these)
    # TabPFN metrics are read in a second pass from tabpfn_*_dir trees. When a tree
    # matches gp_dir or gpytorch_dir (same folder holds GP and TabPFN), do not skip
    # the GP pass here — and use resolved paths so rel vs abs roots still match.
    for path in all_paths:
        if _skip_gp_pass_for_tabpfn_tree(path, tabpfn_v25_dir, gp_dir, gpytorch_dir):
            continue
        if _skip_gp_pass_for_tabpfn_tree(path, tabpfn_v2_dir, gp_dir, gpytorch_dir):
            continue
            
        with path.open("r") as f:
            data = json.load(f)
        base_dir = _base_dir_for_meta(path, gp_dir, gpytorch_dir)
        meta = parse_meta_from_filename(path, base_dir)
        meta["source_file"] = str(path)
        
        # Determine model types based on filename
        path_str_lower = str(path).lower()
        has_gpytorch_in_name = "gpytorch" in path_str_lower
        
        # Extract GP data only (no TabPFN from regular files - TabPFN comes from dedicated directories only)
        if has_gpytorch_in_name:
            # For files with gpytorch in name, gp_data section contains GPytorch results
            rows.extend(extract_rows_from_section(data, "gp_data", "gpytorch", meta))
        else:
            # For files without gpytorch in name, gp_data section contains GP+ results
            rows.extend(extract_rows_from_section(data, "gp_data", "gpplus", meta))
    
    # Process TabPFN v2.5 files from dedicated directory
    if tabpfn_v25_dir and tabpfn_v25_dir.exists():
        for path in iter_json_files(tabpfn_v25_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, tabpfn_v25_dir)
            meta["source_file"] = str(path)
            # Extract TabPFN v2.5 data
            rows.extend(extract_rows_from_section(data, "tabpfn_data", "tabpfn_v2.5", meta))
    
    # Process TabPFN v2 files from dedicated directory
    if tabpfn_v2_dir and tabpfn_v2_dir.exists():
        for path in iter_json_files(tabpfn_v2_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, tabpfn_v2_dir)
            meta["source_file"] = str(path)
            # Extract TabPFN v2 data
            rows.extend(extract_rows_from_section(data, "tabpfn_data", "tabpfn_v2", meta))

    # Optional extra named result trees.
    # Treat each extra entry as its own model_family label and ingest whichever
    # sections are present in those JSONs (gp_data and/or tabpfn_data).
    for model_name, edir in extra_named_gp or []:
        if not edir.exists():
            print(f"Warning: --extra_gp path does not exist, skipping: {model_name} -> {edir}")
            continue
        for path in iter_json_files(edir, problems):
            if _skip_gp_pass_for_tabpfn_tree(path, tabpfn_v25_dir, gp_dir, gpytorch_dir):
                continue
            if _skip_gp_pass_for_tabpfn_tree(path, tabpfn_v2_dir, gp_dir, gpytorch_dir):
                continue
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, edir)
            meta["source_file"] = str(path)
            rows.extend(extract_rows_from_section(data, "gp_data", model_name, meta))
            rows.extend(extract_rows_from_section(data, "tabpfn_data", model_name, meta))

    # Seek
    if seek_dir:
        for path in iter_json_files(seek_dir, problems):
            with path.open("r") as f:
                data = json.load(f)
            meta = parse_meta_from_filename(path, seek_dir)
            meta["source_file"] = str(path)
            rows.extend(extract_rows_from_section(data, "gp_data", "seek", meta))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Normalize noise label for plotting
    noise_col = df["noiseTrain"].fillna("")
    fallback = df["noiseTest"].fillna("")
    df["noise_label"] = noise_col.where(noise_col != "", other=fallback)
    # For problems without dimension (like M2AX), use "all" instead of empty
    df["dim"] = df["dim"].replace("", "all")
    # For problems without xdim, use "all" instead of empty
    df["xdim"] = df["xdim"].replace("", "all")
    # M2AX uses test_size as grouping parameter (not noise), so keep the test_size value
    # (it was already extracted from filename in parse_meta_from_filename)
    df["noise_label"] = df["noise_label"].replace("", "unknown")

    # Real datasets often have no noise tags in filenames; if a whole problem has only
    # unknown noise labels, collapse it to a single explicit bucket so plotting creates
    # one consolidated figure instead of expecting noise splits.
    for problem, idx in df.groupby("problem").groups.items():
        if (df.loc[idx, "noise_label"] == "unknown").all():
            df.loc[idx, "noise_label"] = "all"
    
    # Optional noise filtering. Keep selected levels, but for problems that do not
    # use noise labels (e.g. real datasets yielding "all"), keep those rows
    # so entire problems are not silently dropped.
    if noise_levels:
        requested = {str(v) for v in noise_levels}
        filtered = df[df["noise_label"].isin(requested)]
        present_problems = set(filtered["problem"].dropna().unique())
        all_problems = set(df["problem"].dropna().unique())
        missing_problems = all_problems - present_problems
        if missing_problems:
            filtered = pd.concat(
                [filtered, df[df["problem"].isin(missing_problems)]],
                ignore_index=True,
            )
        df = filtered
    
    return df


RESERVED_GP_MODEL_FAMILY_NAMES = frozenset(
    {"gpplus", "gpytorch", "seek", "tabpfn_v2.5", "tabpfn_v2"}
)


def parse_extra_named_gp_entries(entries: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """Parse repeated ``NAME=PATH`` CLI entries into (model_family, directory) pairs."""
    if not entries:
        return []
    out: List[Tuple[str, Path]] = []
    seen_names: set[str] = set()
    reserved_lower = {x.lower() for x in RESERVED_GP_MODEL_FAMILY_NAMES}
    for raw in entries:
        s = raw.strip()
        if not s:
            continue
        if "=" not in s:
            raise SystemExit(f"Invalid --extra_gp {raw!r}: expected NAME=PATH")
        name, path_str = s.split("=", 1)
        name = name.strip()
        path_str = path_str.strip()
        if not name:
            raise SystemExit(f"Invalid --extra_gp {raw!r}: name before '=' cannot be empty")
        if not path_str:
            raise SystemExit(f"Invalid --extra_gp {raw!r}: path after '=' cannot be empty")
        if name.lower() in reserved_lower:
            raise SystemExit(
                f"Invalid extra model name {name!r}: conflicts with a built-in model id "
                f"(reserved: {', '.join(sorted(RESERVED_GP_MODEL_FAMILY_NAMES))})."
            )
        if name in seen_names:
            raise SystemExit(f"Duplicate --extra_gp name {name!r}")
        seen_names.add(name)
        out.append((name, Path(path_str)))
    return out


def display_label_for_model_family(m: str) -> str:
    """Axis / legend / table title label for a model_family string."""
    if m == "gpplus":
        return "GP+"
    if m == "gpytorch":
        return "GPyTorch"
    if m == "seek":
        return "Seek"
    if m == "tabpfn_v2.5":
        return "PFN 2.5"
    if m == "tabpfn_v2":
        return "PFN 2.0"
    label = m.replace("_", " ")
    # Keep long custom model names readable in legends/axes.
    # Prefer breaking before parenthetical suffix, e.g. "GP+\n(PE log)".
    if " (" in label:
        return label.replace(" (", "\n(", 1)
    # Fallback: split long labels into two lines at nearest middle space.
    if len(label) > 18 and " " in label:
        mid = len(label) // 2
        spaces = [i for i, ch in enumerate(label) if ch == " "]
        split_idx = min(spaces, key=lambda i: abs(i - mid))
        return f"{label[:split_idx]}\n{label[split_idx + 1:]}"
    return label


def get_model_order(
    gpytorch_dir: Optional[Path] = None,
    tabpfn_v25_dir: Optional[Path] = None,
    tabpfn_v2_dir: Optional[Path] = None,
    seek_dir: Optional[Path] = None,
    extra_named_gp: Optional[List[Tuple[str, Path]]] = None,
) -> list[str]:
    """Get model order based on directories explicitly provided."""
    base_models = ["gpplus"]
    for name, _ in extra_named_gp or []:
        base_models.append(name)
    if tabpfn_v25_dir is not None:
        base_models.append("tabpfn_v2.5")
    if tabpfn_v2_dir is not None:
        base_models.append("tabpfn_v2")
    if gpytorch_dir is not None:
        base_models.append("gpytorch")
    if seek_dir is not None:
        base_models.append("seek")
    return base_models


def format_problem_title(problem: str) -> str:
    """Format problem name for display, with special handling for wing -> Wing Weight."""
    if problem.lower() == "wing":
        return "Wing Weight"
    return problem.title()


def format_number_latex(value: float, metric: str) -> str:
    """Format numbers for LaTeX tables with maximum 5 digits total."""
    if metric == "RRMSE" or metric == "NIS":
        # Use compact formatting to fit in cells
        if abs(value) >= 100:
            return f"{value:.1f}"  # 1 decimal place for large numbers
        elif abs(value) >= 10:
            return f"{value:.2f}"  # 2 decimal places
        else:
            return f"{value:.3f}"  # 3 decimal places for small numbers
    elif "Time" in metric:
        # Time metrics: compact formatting
        if abs(value) >= 1000:
            return f"{value:.1e}"  # Compact scientific
        else:
            return f"{value:.1f}"  # 1 decimal place
    else:
        # Default formatting
        return f"{value:.2f}"


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers from data using the Interquartile Range (IQR) method.
    
    Args:
        data: Array of values
        factor: Multiplier for IQR (default 1.5, standard for box plots)
    
    Returns:
        Filtered array with outliers removed
    """
    if len(data) == 0:
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        # If IQR is 0, all values are the same, return as-is
        return data
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    # Filter out outliers
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask]


def _save_fig(outfile: Path, fig, dpi: int = 200, **kwargs) -> None:
    """Save figure to outfile; on PermissionError (e.g. PDF open elsewhere) save as PNG instead."""
    try:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", **kwargs)
    except PermissionError:
        png_file = outfile.with_suffix(".png")
        fig.savefig(png_file, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"  (PDF locked, saved as {png_file})")
    else:
        print(f"Saved {outfile}")


def create_violin_plot(
    ax,
    data_arrays,
    metric: str,
    labels: list[str],
    title: str,
    add_mean_median_legend: bool = True,
):
    """Mimic metrics_functions.plot_metrics styling: gray violins with mean/median lines."""
    # Filter out empty arrays to avoid matplotlib errors
    non_empty = [(arr, lab) for arr, lab in zip(data_arrays, labels) if len(arr) > 0]
    if not non_empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_title(title)
        return

    data_arrays, labels = zip(*non_empty)

    parts = ax.violinplot(data_arrays, showmeans=False, showmedians=False, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#888888")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    # Overlay mean (pink) and median (dashed black) lines spanning full violin width
    for i, arr in enumerate(data_arrays, start=1):
        if len(arr) == 0:
            continue
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        # Default matplotlib violin width is ~0.5 around the center, so use +/-0.5
        ax.hlines(mean_v, i - 0.5, i + 0.5, colors="#ff69b4", linewidth=2)
        ax.hlines(med_v, i - 0.5, i + 0.5, colors="black", linewidth=2, linestyles="--")

    # Legend for mean/median (optionally disabled for multi-panel layouts)
    if add_mean_median_legend:
        try:
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], color="#ff69b4", lw=2, label="Mean"),
                Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                frameon=False,
            )
        except Exception:
            pass

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel(metric, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def create_stacked_plot(rrmse_arrays, nis_arrays, labels: list[str], title: str):
    """Create a stacked (RRMSE over NIS) figure with shared styling."""
    # Skip if both are empty
    has_rrmse = any(len(a) > 0 for a in rrmse_arrays)
    has_nis = any(len(a) > 0 for a in nis_arrays)
    if not (has_rrmse or has_nis):
        return None

    # 16:9-ish aspect
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)

    if has_rrmse:
        create_violin_plot(ax_top, rrmse_arrays, "RRMSE", labels, f"{title} - RRMSE")
    else:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_top.set_xticks([])
        ax_top.set_title(f"{title} - RRMSE")

    if has_nis:
        create_violin_plot(ax_bottom, nis_arrays, "NIS", labels, f"{title} - NIS")
    else:
        ax_bottom.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_bottom.set_xticks([])
        ax_bottom.set_title(f"{title} - NIS")

    plt.tight_layout()
    return fig


def create_table_in_ax(ax, table_data, stats, metrics, method_name: str):
    """Create a professional table in the given axes."""
    ax.axis("off")
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric"] + stats,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    # Professional styling
    # Header row
    for i in range(len(stats) + 1):
        table[(0, i)].set_facecolor("#2E75B6")  # Professional blue
        table[(0, i)].set_text_props(weight="bold", color="white", size=10)
        table[(0, i)].set_height(0.08)
    
    # Data rows with alternating colors
    for i, metric in enumerate(metrics):
        row_idx = i + 1
        # Metric column
        table[(row_idx, 0)].set_facecolor("#D0DEEF")
        table[(row_idx, 0)].set_text_props(weight="bold", size=9)
        
        # Statistics columns
        if i % 2 == 0:
            bg_color = "#F2F2F2"  # Light gray
        else:
            bg_color = "#FFFFFF"  # White
        
        for j in range(1, len(stats) + 1):
            table[(row_idx, j)].set_facecolor(bg_color)
            table[(row_idx, j)].set_text_props(size=8)
    
    # Set column widths
    table.auto_set_column_width([0, 1, 2, 3, 4])
    table.scale(1, 2.0)
    
    # Add method name as subtitle
    ax.set_title(display_label_for_model_family(method_name), fontsize=11, fontweight="bold", pad=10)


def _create_simple_table(ax, table_body, col_labels, title: str) -> None:
    """Render a simple table with consistent styling. Expects body rows only."""
    ax.axis("off")
    table = ax.table(
        cellText=table_body,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Header styling - maximize font size
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#2E75B6")
        table[(0, i)].set_text_props(weight="bold", color="white", size=20)
        table[(0, i)].set_height(0.03)  # Smaller header row height (blue boxes)

    # Body styling - maximize font size
    for r in range(1, len(table_body) + 1):
        bg_color = "#F2F2F2" if (r % 2) else "#FFFFFF"
        for c in range(len(col_labels)):
            table[(r, c)].set_facecolor(bg_color)
            table[(r, c)].set_text_props(size=24)  # Large font for visibility when shrunk

    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.2)  # Reduced vertical scaling to minimize padding
    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)  # Reduced padding


def create_comprehensive_tables(df: pd.DataFrame, out_dir: Path, model_order: list[str]) -> None:
    """
    Create comprehensive tables with median and std for all metrics across all configurations.
    
    Table structure:
    - Main rows: (xdim, dim) combinations
    - Sub-rows: Noise levels (0.005, 0.05) nested under each (xdim, dim)
    - Main columns: Models (gpplus, tabpfn_v2.5, tabpfn_v2, gpytorch)
    - Sub-columns: For each metric (RRMSE, NIS, Training_Time, Prediction_Time), show Median and Std
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ["RRMSE", "NIS", "Training_Time", "Prediction_Time"]
    model_order_filtered = [m for m in model_order if m in ["gpplus", "tabpfn_v2.5", "tabpfn_v2", "gpytorch"]]
    
    # Group by problem
    for problem, prob_df in df.groupby("problem"):
        # Skip problems with missing xdim or dim
        prob_df = prob_df[(prob_df["xdim"] != "") & (prob_df["xdim"] != "all") & 
                          (prob_df["dim"] != "") & (prob_df["dim"] != "all")]
        if prob_df.empty:
            continue
        
        # Get unique combinations
        xdims = sorted([x for x in prob_df["xdim"].unique()], key=lambda v: float(v))
        dims = sorted([d for d in prob_df["dim"].unique()], 
                     key=lambda v: float(v.rstrip("D")) if isinstance(v, str) and v.endswith("D") else float(v))
        noise_levels = sorted([n for n in prob_df["noise_label"].unique() if n in ["0.005", "0.05"]])
        
        if not xdims or not dims or not noise_levels:
            continue
        
        # Build table data structure
        # Calculate statistics for each combination
        stats_dict = {}
        for xdim in xdims:
            for dim in dims:
                for noise in noise_levels:
                    key = (xdim, dim, noise)
                    stats_dict[key] = {}
                    for model in model_order_filtered:
                        subset = prob_df[
                            (prob_df["xdim"] == xdim) &
                            (prob_df["dim"] == dim) &
                            (prob_df["noise_label"] == noise) &
                            (prob_df["model_family"] == model)
                        ]
                        stats_dict[key][model] = {}
                        for metric in metrics:
                            # Skip Training_Time for TabPFN models
                            if metric == "Training_Time" and model.startswith("tabpfn"):
                                stats_dict[key][model][metric] = {"median": None, "std": None}
                            else:
                                vals = subset[metric].dropna().astype(float).values
                                if len(vals) == 0:
                                    stats_dict[key][model][metric] = {"median": None, "std": None}
                                else:
                                    stats_dict[key][model][metric] = {
                                        "median": float(np.median(vals)),
                                        "std": float(np.std(vals))
                                    }
        
        # Build column headers with simple multi-level structure (original design)
        header_row1 = [""]  # Top level: Methods
        header_row2 = [""]  # Second level: Metrics
        header_row3 = [""]  # Third level: Statistics (Median, Std)
        
        for model in model_order_filtered:
            # Format model names: gpplus -> GP+, gpytorch -> GPyTorch
            if model == "gpplus":
                model_display = "GP+"
            elif model == "gpytorch":
                model_display = "GPyTorch"
            else:
                model_display = model.replace("_", " ").title()
            
            for metric in metrics:
                # Skip Training_Time for TabPFN
                if metric == "Training_Time" and model.startswith("tabpfn"):
                    continue
                
                # Format: "Model\nMetric\nMedian" and "Model\nMetric\nStd"
                header_row1.append(f"{model_display}\n{metric}\nMedian")
                header_row2.append("")  # Not used in simple design
                header_row3.append("")  # Not used in simple design
                
                header_row1.append(f"{model_display}\n{metric}\nStd")
                header_row2.append("")  # Not used in simple design
                header_row3.append("")  # Not used in simple design
        
        # Build table body with hierarchical rows
        table_body = []
        row_labels = []
        row_is_main = []  # Track which rows are main rows vs sub-rows
        
        for xdim in xdims:
            for dim in dims:
                # Main row label
                dim_clean = dim.rstrip("D") if dim.endswith("D") else dim
                main_row_label = f"Dx={xdim}, N={dim_clean}Dx"
                
                # Add rows for each noise level
                for noise_idx, noise in enumerate(noise_levels):
                    row_data = []
                    key = (xdim, dim, noise)
                    
                    for model in model_order_filtered:
                        for metric in metrics:
                            if metric == "Training_Time" and model.startswith("tabpfn"):
                                continue
                            stats = stats_dict[key][model][metric]
                            if stats["median"] is not None:
                                if metric == "RRMSE":
                                    row_data.append(f"{stats['median']:.4f}")
                                    row_data.append(f"{stats['std']:.4f}")
                                elif "Time" in metric:
                                    row_data.append(f"{stats['median']:.2f}")
                                    row_data.append(f"{stats['std']:.2f}")
                                else:  # NIS
                                    row_data.append(f"{stats['median']:.6f}")
                                    row_data.append(f"{stats['std']:.6f}")
                            else:
                                row_data.append("N/A")
                                row_data.append("N/A")
                    
                    # First noise level gets the main row label, others are sub-rows
                    if noise_idx == 0:
                        row_labels.append(main_row_label)
                        row_is_main.append(True)
                    else:
                        row_labels.append(f"  noise={noise}")  # Indented sub-row
                        row_is_main.append(False)
                    
                    table_body.append(row_data)
        
        # Create figure with landscape orientation for wide tables
        n_cols = len(header_row3) - 1  # Subtract 1 for row label column
        n_rows = len(table_body)
        
        # Calculate figure size (landscape, wide enough for all columns)
        fig_width = max(20, n_cols * 0.8)
        fig_height = max(8, n_rows * 0.4 + 3)  # Extra height for 3 header rows
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        
        # Create table with simple combined headers (original design)
        col_labels_simple = [""] + [h for h in header_row1[1:]]  # Skip first empty element
        
        table = ax.table(
            cellText=[[label] + row for label, row in zip(row_labels, table_body)],
            colLabels=col_labels_simple,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        
        # Style header (simple single row with combined text)
        for col_idx in range(len(col_labels_simple)):
            table[(0, col_idx)].set_facecolor("#2E75B6")
            table[(0, col_idx)].set_text_props(weight="bold", color="white", size=8)
            table[(0, col_idx)].set_height(0.06)  # Taller for multi-line headers
        
        # Style data rows with hierarchical structure
        # Data rows start at index 1 (after 1 header row)
        main_row_count = 0
        for r in range(1, len(table_body) + 1):
            data_row_idx = r - 1
            is_main = row_is_main[data_row_idx]
            
            if is_main:
                main_row_count += 1
                # Main rows: alternating colors, bold labels
                bg_color = "#F2F2F2" if (main_row_count % 2 == 0) else "#FFFFFF"
                label_weight = "bold"
                label_size = 8
            else:
                # Sub-rows: lighter gray background
                bg_color = "#E8E8E8"
                label_weight = "normal"
                label_size = 7
            
            for c in range(len(col_labels_simple)):
                table[(r, c)].set_facecolor(bg_color)
                table[(r, c)].set_text_props(size=7)
            
            # Style row label column
            table[(r, 0)].set_text_props(weight=label_weight, size=label_size)
        
        table.auto_set_column_width(list(range(len(col_labels_simple))))
        table.scale(1, 1.5)
        
        # Set title
        title = f"{format_problem_title(str(problem))} - Comprehensive Metrics Table"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        
        plt.tight_layout()
        
        # Save
        outfile = out_dir / f"{problem}_comprehensive_table.pdf"
        fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {outfile}")


def create_comprehensive_tables_latex(df: pd.DataFrame, out_dir: Path, model_order: list[str]) -> None:
    """
    Create LaTeX tables with proper merged cells for comprehensive metrics.
    Creates two separate tables per problem:
    1. RRMSE and NIS metrics
    2. Training Time and Prediction Time metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = ["RRMSE", "NIS", "Training_Time", "Prediction_Time"]
    model_order_filtered = [m for m in model_order if m in ["gpplus", "tabpfn_v2.5", "tabpfn_v2", "gpytorch"]]
    
    # Group by problem
    for problem, prob_df in df.groupby("problem"):
        # Skip problems with missing xdim or dim
        prob_df = prob_df[(prob_df["xdim"] != "") & (prob_df["xdim"] != "all") & 
                          (prob_df["dim"] != "") & (prob_df["dim"] != "all")]
        if prob_df.empty:
            continue
        
        # Get existing (xdim, dim) combinations from the actual data
        existing_combinations = set()
        for _, row in prob_df.iterrows():
            existing_combinations.add((str(row["xdim"]), str(row["dim"]), str(row["noise_label"])))
        
        # Get unique combinations from existing data only
        valid_xdims = sorted({xdim for xdim, dim, _ in existing_combinations}, key=lambda v: float(v))
        valid_dims = sorted({dim for xdim, dim, _ in existing_combinations})
        noise_levels = sorted({noise for _, _, noise in existing_combinations if noise in ["0.005", "0.05"]})
        
        if not valid_xdims or not valid_dims or not noise_levels:
            continue
        
        # Build table data structure - iterate directly over existing combinations only
        stats_dict = {}
        for xdim, dim, noise in existing_combinations:
            key = (xdim, dim, noise)
            
            # Process this combination since we know it exists in data
            subset = prob_df[
                (prob_df["xdim"] == xdim) &
                (prob_df["dim"] == dim) &
                (prob_df["noise_label"] == noise)
            ]
            
            stats_dict[key] = {}
            for model in model_order_filtered:
                model_subset = subset[subset["model_family"] == model]
                stats_dict[key][model] = {}
                for metric in all_metrics:
                    # Skip Training_Time for TabPFN models
                    if metric == "Training_Time" and model.startswith("tabpfn"):
                        stats_dict[key][model][metric] = {"median": None, "std": None}
                    else:
                        vals = model_subset[metric].dropna().astype(float).values
                        if len(vals) == 0:
                            stats_dict[key][model][metric] = {"median": None, "std": None}
                        else:
                            stats_dict[key][model][metric] = {
                                "median": float(np.median(vals)),
                                "std": float(np.std(vals))
                            }
        
        # Create table for RRMSE and NIS
        _create_metric_table_latex(
            problem, stats_dict, valid_xdims, valid_dims, noise_levels, 
            model_order_filtered, ["RRMSE", "NIS"], 
            out_dir, f"{problem}_metrics_table.tex", 
            f"{format_problem_title(str(problem))} - RRMSE and NIS Metrics"
        )
        
        # Create table for Training Time and Prediction Time
        _create_metric_table_latex(
            problem, stats_dict, valid_xdims, valid_dims, noise_levels,
            model_order_filtered, ["Training_Time", "Prediction_Time"],
            out_dir, f"{problem}_time_table.tex",
            f"{format_problem_title(str(problem))} - Training and Prediction Time"
        )


def _create_metric_table_latex(
    problem: str, stats_dict: dict, xdims: list, dims: list, noise_levels: list,
    model_order_filtered: list, metrics: list, out_dir: Path, 
    filename: str, caption: str
) -> None:
    """Helper function to create a LaTeX table for specific metrics."""
    def _get_stats_entry(key, model: str, metric: str) -> dict:
        """Return stats entry or an empty placeholder when combination is missing."""
        return (
            stats_dict.get(key, {})
            .get(model, {})
            .get(metric, {"median": None, "std": None})
        )

    # Calculate columns per model for LaTeX multicolumn
    model_col_counts = {}
    for model in model_order_filtered:
        count = 0
        for metric in metrics:
            if metric == "Training_Time" and model.startswith("tabpfn"):
                continue
            count += 2  # Median and Std
        model_col_counts[model] = count
    
    # Build LaTeX table
    latex_lines = []
    latex_lines.append("% Requires: \\usepackage{booktabs, multirow, makecell, tabularx, array, xcolor, colortbl, hhline}")
    latex_lines.append("% Use \\hhline to prevent color bleeding between cells")
    latex_lines.append("\\begin{table*}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append("\\label{tab:" + problem.lower().replace(" ", "_") + "}")
    
    # Data columns (Dx/N merged + Noise) + data columns
    # Use tabularx with X columns to fit page width, minimize label column widths
    total_data_cols = sum(model_col_counts.values())
    # Use X columns that expand to fill width, with proper spacing to prevent color bleeding
    # Add small spacing between columns to prevent color overlap
    col_sep = "@{\\hspace{0.5pt}}"
    # Make Data, Noise columns narrower, give more space to metric columns
    col_spec = f"|{col_sep}p{{1.2cm}}{col_sep}|{col_sep}p{{0.6cm}}{col_sep}|" + f"{col_sep}>{{\\centering\\arraybackslash}}X{col_sep}|" * total_data_cols
    latex_lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Header row 1: Methods (merged) - bold
    header_row1 = "\\multicolumn{1}{|c|}{\\bfseries Data} & \\multicolumn{1}{p{0.6cm}|}{\\bfseries Noise}"
    for model in model_order_filtered:
        # Format model names: GPPLUS -> GP+, TabPFN versions, GPYTORCH -> GPyTorch
        if model == "gpplus":
            model_display = "GP+"
        elif model == "tabpfn_v2.5":
            model_display = "TabPFN v2.5"
        elif model == "tabpfn_v2":
            model_display = "TabPFN v2.0"
        elif model == "gpytorch":
            model_display = "GPyTorch"
        else:
            model_display = model.replace("_", " ").title()
        col_span = model_col_counts[model]
        header_row1 += f" & \\multicolumn{{{col_span}}}{{c|}}{{\\bfseries {model_display}}}"
    header_row1 += " \\\\ \\midrule"
    latex_lines.append(header_row1)
    
    # Header row 2: Metrics - bold
    header_row2 = "\\multicolumn{1}{|c|}{} & \\multicolumn{1}{p{0.6cm}|}{}"
    for model in model_order_filtered:
        for metric in metrics:
            if metric == "Training_Time" and model.startswith("tabpfn"):
                continue
            # Format metric name for header
            if metric == "Training_Time":
                metric_display = "\\makecell{Training\\\\Time}"
            elif metric == "Prediction_Time":
                metric_display = "\\makecell{Prediction\\\\Time}"
            else:
                metric_display = metric.replace("_", " ")
            header_row2 += f" & \\multicolumn{{2}}{{c|}}{{\\bfseries {metric_display}}}"
    header_row2 += " \\\\"
    latex_lines.append(header_row2)
    # Add cmidrule for metrics row (spans all data columns)
    latex_lines.append(f"\\cmidrule(lr){{3-{total_data_cols + 2}}}")
    
    # Header row 3: Statistics - bold
    header_row3 = "\\multicolumn{1}{|c|}{} & \\multicolumn{1}{p{0.6cm}|}{}"
    for model in model_order_filtered:
        for metric in metrics:
            if metric == "Training_Time" and model.startswith("tabpfn"):
                continue
            header_row3 += " & \\bfseries Median & \\bfseries Std"
    header_row3 += " \\\\ \\midrule"
    latex_lines.append(header_row3)
    
    # Data rows: Dx/N merged across noise levels, noise in separate column
    num_noise_levels = len(noise_levels)
    row_counter = 0  # Track row number for alternating colors
    
    # Only iterate over (xdim, dim) combinations that exist in stats_dict
    existing_xdim_dim_combinations = set()
    for key in stats_dict.keys():
        xdim, dim, _ = key
        existing_xdim_dim_combinations.add((xdim, dim))
    
    for xdim in xdims:
        for dim in dims:
            # Skip if this (xdim, dim) combination doesn't exist in stats_dict
            if (xdim, dim) not in existing_xdim_dim_combinations:
                continue
                
            dim_clean = dim.rstrip("D") if dim.endswith("D") else dim
            # Dx and N on two lines using \makecell, constrained to narrow width
            # Use D$_x$ for subscript x
            dim_label = f"\\makebox[0.12cm][c]{{\\makecell{{D$_x$={xdim}\\\\N={dim_clean}D$_x$}}}}"
            
            # First noise level - main row with Dx/N merged
            if noise_levels:
                noise = noise_levels[0]
                key = (xdim, dim, noise)
                # Dx/N column merged across all noise levels for this (xdim, dim)
                # Use multicolumn with |c| to ensure full borders around the cell
                # Only use multicolumn for merged cell
                row_data = f"\\multicolumn{{1}}{{|p{{1.2cm}}|}}{{\\multirow{{{num_noise_levels}}}{{*}}{{{dim_label}}}}} & \\makebox[0.5cm][c]{{{noise}}} & "
                row_counter += 1
                
                for model in model_order_filtered:
                    for metric in metrics:
                        if metric == "Training_Time" and model.startswith("tabpfn"):
                            continue
                        stats = _get_stats_entry(key, model, metric)
                        if stats["median"] is not None:
                            median_val = format_number_latex(stats['median'], metric)
                            std_val = format_number_latex(stats['std'], metric)
                            row_data += f"{median_val} & {std_val} & "
                        else:
                            na_val = "N/A"
                            row_data += f"{na_val} & {na_val} & "
                
                # Remove trailing " & " and add line break (no hline between merged rows)
                row_data = row_data.rstrip(" & ") + " \\\\"
                latex_lines.append(row_data)
            
            # Remaining noise levels - sub-rows with empty Dx/N (merged)
            for noise_idx, noise in enumerate(noise_levels[1:], start=1):
                key = (xdim, dim, noise)
                # Empty Dx/N cell (merged above), noise in second column
                # Use multicolumn to preserve left border for empty merged cell
                # Continue for sub-rows
                row_data = f"\\multicolumn{{1}}{{|p{{1.2cm}}|}}{{}} & \\makebox[0.5cm][c]{{{noise}}} & "
                
                for model in model_order_filtered:
                    for metric in metrics:
                        if metric == "Training_Time" and model.startswith("tabpfn"):
                            continue
                        stats = _get_stats_entry(key, model, metric)
                        if stats["median"] is not None:
                            median_val = format_number_latex(stats['median'], metric)
                            std_val = format_number_latex(stats['std'], metric)
                            row_data += f"{median_val} & {std_val} & "
                        else:
                            na_val = "N/A"
                            row_data += f"{na_val} & {na_val} & "
                
                # Remove trailing " & " and add line break
                # Only add rule after the last noise level for this Dx/N group
                if noise_idx == len(noise_levels) - 1:
                    row_data = row_data.rstrip(" & ") + " \\\\ \\midrule"
                else:
                    row_data = row_data.rstrip(" & ") + " \\\\"
                latex_lines.append(row_data)
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabularx}")
    latex_lines.append("\\end{table*}")
    
    # Save LaTeX file
    outfile = out_dir / filename
    with outfile.open("w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {outfile}")


def create_time_tables(df: pd.DataFrame, out_dir: Path, model_order: list[str]) -> None:
    """
    Create median time tables:
      - Training_Time: GPs only (gpplus, gpytorch)
      - Prediction_Time: all methods
    Columns: dataset size (N = Dx).
    One PDF per problem and noise level.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Methods for each table
    training_methods = [m for m in model_order if m in ["gpplus", "gpytorch"]]
    inference_methods = [m for m in model_order if m in ["gpplus", "gpytorch", "tabpfn_v2.5", "tabpfn_v2"]]

    for problem, prob_df in df.groupby("problem"):
        for noise_label, noise_df in prob_df.groupby("noise_label"):
            xdims = sorted([x for x in noise_df["xdim"].unique() if x not in ("", "all")], key=lambda v: float(v))
            if not xdims:
                continue

            for xd in xdims:
                subset = noise_df[noise_df["xdim"] == xd]
                dims = sorted([d for d in subset["dim"].unique() if d not in ("", "all")], key=lambda v: float(v.rstrip("D")) if isinstance(v, str) and v.endswith("D") else float(v))
                if not dims:
                    continue

                # Columns = methods, rows = N (train size)
                train_cols = ["N"] + training_methods
                train_rows = []
                for dim in dims:
                    # Format as N = 10D_x with subscript x
                    dim_clean = dim.rstrip("D") if dim.endswith("D") else dim
                    row = [f"N = {dim_clean}D$_x$"]
                    for method in training_methods:
                        vals = subset[
                            (subset["model_family"] == method)
                            & (subset["dim"] == dim)
                        ]["Training_Time"].dropna().astype(float)
                        med = vals.median() if len(vals) else None
                        row.append(f"{med:.2f}" if med is not None else "N/A")
                    train_rows.append(row)

                infer_cols = ["N"] + inference_methods
                infer_rows = []
                for dim in dims:
                    # Format as N = 10D_x with subscript x
                    dim_clean = dim.rstrip("D") if dim.endswith("D") else dim
                    row_inf = [f"N = {dim_clean}D$_x$"]
                    for method in inference_methods:
                        vals = subset[
                            (subset["model_family"] == method)
                            & (subset["dim"] == dim)
                        ]["Prediction_Time"].dropna().astype(float)
                        med = vals.median() if len(vals) else None
                        row_inf.append(f"{med:.2f}" if med is not None else "N/A")
                    infer_rows.append(row_inf)

                # Skip if both tables have no data
                has_train_data = any(any(cell != "N/A" for cell in row[1:]) for row in train_rows)
                has_infer_data = any(any(cell != "N/A" for cell in row[1:]) for row in infer_rows)
                if not has_train_data and not has_infer_data:
                    continue

                # Use gridspec to create unequal column widths
                from matplotlib import gridspec
                fig = plt.figure(figsize=(18, 8))  # Larger figure to accommodate bigger fonts
                gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5])  # Training table narrower, inference wider
                axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
                _create_simple_table(axes[0], train_rows, train_cols, "Training Time (s) – median")
                _create_simple_table(axes[1], infer_rows, infer_cols, "Inference Time (s) – median")

                title_txt = f"{format_problem_title(str(problem))} - Noise {noise_label} - Dx={xd}"
                fig.suptitle(title_txt, fontsize=20, fontweight="bold", y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reduced top margin to maximize table space

                outfile = out_dir / f"{problem}_noise{noise_label}_Dx{xd}_time_tables.pdf"
                fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"Saved {outfile}")


def create_summary_table(df: pd.DataFrame, out_dir: Path, model_order: list[str]) -> None:
    """Create professional summary tables: one image per problem per noise level per xdim and dim with all methods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ["RRMSE", "RMSE", "NIS", "Training_Time"]
    stats = ["Mean", "Median", "Min", "Max"]
    
    # Group by problem, noise_label, xdim, and dim
    for (problem, noise_label, xdim, dim), group_df in df.groupby(["problem", "noise_label", "xdim", "dim"]):
        # Skip only if xdim or dim is truly missing (shouldn't happen after normalization, but just in case)
        if xdim == "" or xdim == "all":
            continue
        if dim == "" or dim == "all":
            continue
            
        # Create figure with subplots (one for each method)
        n_methods = len(model_order)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, max(5, len(metrics) * 1.2 + 2)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, model_family in enumerate(model_order):
            # Collect statistics for this problem/method/noise/xdim/dim combination
            sub_df = group_df[group_df["model_family"] == model_family]
            stats_data = {}
            for metric in metrics:
                vals = sub_df[metric].dropna().astype(float).values
                if len(vals) == 0:
                    stats_data[metric] = {"Mean": None, "Median": None, "Min": None, "Max": None}
                else:
                    stats_data[metric] = {
                        "Mean": float(np.mean(vals)),
                        "Median": float(np.median(vals)),
                        "Min": float(np.min(vals)),
                        "Max": float(np.max(vals)),
                    }
            
            # Build table data: metrics as rows, stats as columns
            table_data = []
            for metric in metrics:
                row = [metric]
                for stat in stats:
                    val = stats_data[metric][stat]
                    if val is None:
                        row.append("N/A")
                    else:
                        # Format time metrics with fewer decimal places
                        if "Time" in metric:
                            row.append(f"{val:.2f}")
                        else:
                            row.append(f"{val:.6f}")
                table_data.append(row)
            
            # Create table in the corresponding subplot
            create_table_in_ax(axes[idx], table_data, stats, metrics, model_family)
        
        # Add overall title
        if problem == "m2ax":
            title_text = f"{problem.upper()} - xDim: {xdim} - {dim} - Test Size: {noise_label}"
        else:
            title_text = f"{format_problem_title(str(problem))} - xDim: {xdim} - {dim} - Noise Level: {noise_label}"
        fig.suptitle(title_text, fontsize=16, fontweight="bold", y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        outfile = out_dir / f"{problem}_xdim{xdim}_{dim}_{noise_label}_summary_tables.pdf"
        plt.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {outfile}")


def make_violin_plots(df: pd.DataFrame, out_dir: Path, model_order: list[str], remove_outliers: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    # Color map for standard noise levels; fallback colors if others exist
    noise_color_map = {
        "0.0": "#1f77b4",
        "0.002": "#ff7f0e",
        "0.005": "#ff7f0e",
        "0.05": "#2ca02c",
        "0.08": "#2ca02c",
    }
    extra_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    def _plot_grouped_metric(
        ax,
        subset: pd.DataFrame,
        metric: str,
        levels: list[str],
        level_label: str,
        is_m2ax: bool,
    ) -> dict:
        """
        Plot one metric axis where:
        - x-axis groups are methods
        - within each method group, we have one violin per level (noise or test_size),
          distinguished by color.
        Returns a dict mapping level -> color actually used (for legend construction).
        """
        n_methods = len(model_order)
        n_levels = len(levels)
        if n_levels == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}

        width = 0.8  # total width per method group
        inner_width = width / max(n_levels, 1)

        data_groups = []
        positions = []
        colors = []

        # Assign colors to each level
        level_colors: dict[str, str] = {}
        extra_idx = 0
        for lvl in levels:
            if lvl in noise_color_map:
                level_colors[lvl] = noise_color_map[lvl]
            else:
                # Assign from extra palette
                level_colors[lvl] = extra_colors[extra_idx % len(extra_colors)]
                extra_idx += 1

        for mi, method in enumerate(model_order):
            center = mi + 1  # method positions at 1,2,3,...
            start = center - width / 2 + inner_width / 2
            for li, lvl in enumerate(levels):
                arr = (
                    subset.loc[
                        (subset["model_family"] == method)
                        & (subset["noise_label"] == lvl),
                        metric,
                    ]
                    .dropna()
                    .astype(float)
                    .values
                )
                if arr.size == 0:
                    continue
                # Remove outliers if requested
                if remove_outliers:
                    arr = remove_outliers_iqr(arr)
                if arr.size == 0:
                    continue
                xpos = start + li * inner_width
                data_groups.append(arr)
                positions.append(xpos)
                colors.append(level_colors[lvl])

        if not data_groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}

        # Add alternating background colors for each method group (white, light grey, white, light grey)
        bg_colors = ["white", "#e0e0e0", "white", "#e0e0e0"]  # white, light grey, white, light grey
        for mi, method in enumerate(model_order):
            if mi < len(model_order):
                bg_color = bg_colors[mi % len(bg_colors)]
                ax.axvspan(mi + 0.5, mi + 1.5, color=bg_color, alpha=0.3, zorder=0)

        parts = ax.violinplot(
            data_groups,
            positions=positions,
            widths=inner_width * 0.9,
            showmeans=False,
            showmedians=False,
            showextrema=True,
        )

        # Color each violin body according to its level
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_edgecolor("black")
            body.set_alpha(0.7)

        # Overlay mean (pink) and median (dashed black) for each group, spanning full inner width
        for xpos, arr in zip(positions, data_groups):
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(
                mean_v,
                xpos - inner_width * 0.5,
                xpos + inner_width * 0.5,
                colors="#ff69b4",
                linewidth=2,
            )
            ax.hlines(
                med_v,
                xpos - inner_width * 0.5,
                xpos + inner_width * 0.5,
                colors="black",
                linewidth=2,
                linestyles="--",
            )

        # X-ticks at method centers
        ax.set_xticks([mi + 1 for mi in range(n_methods)])
        # Format model labels for display
        labels = [display_label_for_model_family(m) for m in model_order]
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Disable default grid, only show horizontal grid lines
        ax.grid(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4, which="major")
        # Add vertical grid lines at boundaries between groups (after violins so they're visible)
        # Boundaries are at the edges of each background section: 0.5, 1.5, 2.5, 3.5, 4.5
        for boundary in [mi + 0.5 for mi in range(n_methods + 1)]:
            ax.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=10)

        # We return level_colors so caller can construct the appropriate legend
        return level_colors

    def _plot_stacked_dim_metric(
        ax,
        subset: pd.DataFrame,
        metric: str,
        dims: list[str],
        noise_levels: list[str],
    ) -> tuple[dict, dict]:
        """
        Plot one metric axis where:
        - x-axis groups are methods
        - within each method group, violins for different dims are stacked at same x position
        - within each dim, violins for different noise levels are side by side
        Returns (noise_level_colors, dim_colors) for legend construction.
        """
        n_methods = len(model_order)
        n_dims = len(dims)
        n_noise = len(noise_levels)
        
        if n_dims == 0 or n_noise == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}, {}

        # Assign colors to noise levels
        noise_colors: dict[str, str] = {}
        extra_idx = 0
        for nl in noise_levels:
            if nl in noise_color_map:
                noise_colors[nl] = noise_color_map[nl]
            else:
                noise_colors[nl] = extra_colors[extra_idx % len(extra_colors)]
                extra_idx += 1

        # Assign colors to dims (for legend)
        dim_colors: dict[str, str] = {}
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
        for i, d in enumerate(dims):
            dim_colors[d] = base_colors[i % len(base_colors)]

        # Horizontal spacing: methods get positions, within each method we have noise levels
        method_width = 0.8
        noise_width = method_width / max(n_noise, 1) if n_noise > 1 else method_width * 0.6
        # Small offset for dim stacking (violins at nearly same x position)
        dim_offset = 0.02  # small offset so violins are slightly separated but appear stacked

        data_groups = []
        x_positions = []
        colors = []
        dim_labels_list = []

        for mi, method in enumerate(model_order):
            method_center = mi + 1
            method_start = method_center - method_width / 2
            
            # For each dim, stack violins at nearly the same x position
            for di, dim in enumerate(dims):
                for ni, noise in enumerate(noise_levels):
                    arr = (
                        subset.loc[
                            (subset["model_family"] == method)
                            & (subset["dim"] == dim)
                            & (subset["noise_label"] == noise),
                            metric,
                        ]
                        .dropna()
                        .astype(float)
                        .values
                    )
                    if arr.size == 0:
                        continue
                    # Remove outliers if requested
                    if remove_outliers:
                        arr = remove_outliers_iqr(arr)
                    if arr.size == 0:
                        continue
                    # Remove outliers if requested
                    if remove_outliers:
                        arr = remove_outliers_iqr(arr)
                    if arr.size == 0:
                        continue
                    
                    # Horizontal position: method center + noise offset + small dim offset for stacking
                    if n_noise > 1:
                        base_xpos = method_start + ni * noise_width + noise_width / 2
                    else:
                        base_xpos = method_center
                    
                    # Add small offset for dim stacking (dims at nearly same x, slightly offset)
                    xpos = base_xpos + (di - (n_dims - 1) / 2) * dim_offset
                    
                    data_groups.append(arr)
                    x_positions.append(xpos)
                    colors.append(noise_colors[noise])
                    dim_labels_list.append(dim)

        if not data_groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}, {}

        # Plot violins
        parts = ax.violinplot(
            data_groups,
            positions=x_positions,
            widths=noise_width * 0.7,  # Slightly narrower to show stacking
            showmeans=False,
            showmedians=False,
            showextrema=True,
        )

        # Color each violin body by noise level, with alpha based on dim (darker = higher dim)
        for i, (body, col, dim) in enumerate(zip(parts["bodies"], colors, dim_labels_list)):
            dim_idx = dims.index(dim)
            # Make higher dims (stacked on top) slightly more transparent
            alpha = 0.7 - (dim_idx * 0.1)
            alpha = max(0.4, alpha)  # Don't make too transparent
            body.set_facecolor(col)
            body.set_edgecolor("black")
            body.set_alpha(alpha)

        # Overlay mean and median lines
        for xpos, arr in zip(x_positions, data_groups):
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            line_width = noise_width * 0.35
            ax.hlines(mean_v, xpos - line_width, xpos + line_width, 
                     colors="#ff69b4", linewidth=2)
            ax.hlines(med_v, xpos - line_width, xpos + line_width,
                     colors="black", linewidth=2, linestyles="--")

        # X-ticks at method centers
        ax.set_xticks([mi + 1 for mi in range(n_methods)])
        # Format model labels for display
        labels = [display_label_for_model_family(m) for m in model_order]
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Disable default grid, only show horizontal grid lines
        ax.grid(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4, which="major")
        # Add vertical grid lines at boundaries between groups (after violins so they're visible)
        # Boundaries are at the edges of each background section: 0.5, 1.5, 2.5, 3.5, 4.5
        for boundary in [mi + 0.5 for mi in range(n_methods + 1)]:
            ax.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=10)

        return noise_colors, dim_colors

    # Handle non-M2AX problems: group by problem and xdim, create rows for each dim
    non_m2ax_df = df[df["problem"] != "m2ax"]
    for (problem, xdim), prob_xdim_df in non_m2ax_df.groupby(["problem", "xdim"]):
        # Skip only if xdim is empty string (not "all", which means no xdim in filename)
        if xdim == "":
            continue

        # Get all dims and noise levels for this problem/xdim
        # Sort dims numerically (5D before 20D) to match paper figure row order.
        dims = sorted(
            [d for d in prob_xdim_df["dim"].unique() if d != "" and d != "all"],
            key=lambda v: float(v.rstrip("D")) if isinstance(v, str) and v.endswith("D") else float(v),
        )
        noise_levels = sorted(
            [n for n in prob_xdim_df["noise_label"].unique() if n != "unknown"],
            key=lambda v: float(v),
        )
        
        if not dims or not noise_levels:
            continue

        # Create subplots: one row per dim, each row has RRMSE and NIS side by side
        # Use 16:9 aspect ratio (PowerPoint standard); widen when many models.
        n_rows = len(dims)
        n_methods = len(model_order)
        fig_width = max(16, 1.55 * n_methods)
        fig_height = 3 * n_rows
        fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), sharex="col")
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        level_colors = None
        
        # Create plots for each dim
        for row_idx, dim in enumerate(dims):
            dim_df = prob_xdim_df[prob_xdim_df["dim"] == dim]
            
            # Left: RRMSE
            level_colors = _plot_grouped_metric(
                axes[row_idx, 0],
                dim_df,
                "RRMSE",
                noise_levels,
                "noise",
                is_m2ax=False,
            )
            
            # Right: NIS
            _ = _plot_grouped_metric(
                axes[row_idx, 1],
                dim_df,
                "NIS",
                noise_levels,
                "noise",
                is_m2ax=False,
            )
            
            # Add dim label to left plot with subscript
            # Extract number from dim (e.g., "10D" -> "10")
            dim_number = dim.rstrip("D") if dim.endswith("D") else dim
            axes[row_idx, 0].text(0.02, 0.98, f"$N = {dim_number}D_x$", transform=axes[row_idx, 0].transAxes,
                                 fontsize=16, fontweight="bold", va="top", ha="left",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Title: show D_x only if xdim is not "all"
        if xdim == "all":
            title = f"{format_problem_title(str(problem))}"
        else:
            title = f"{format_problem_title(str(problem))} - $D_x$ = {xdim}"
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)

        # Create legend with noise levels
        from matplotlib.lines import Line2D

        if level_colors:
            noise_handles = [
                Line2D([0], [0], marker="s", color="none", markerfacecolor=col, 
                      markersize=10, label=f"{lvl} noise")
                for lvl, col in level_colors.items()
            ]
            # Mean/median style handles
            mean_med_handles = [
                Line2D([0], [0], color="#ff69b4", lw=2, label="Mean"),
                Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
            ]

            # Place legend on the right side of the figure
            fig.legend(
                handles=noise_handles + mean_med_handles,
                loc="center right",
                bbox_to_anchor=(1.0, 0.5),
                title="Noise levels",
                frameon=False,
                fontsize=14,
                title_fontsize=16,
            )

        # Adjust layout to leave space for legend on the right
        plt.tight_layout(rect=[0, 0, 0.85, 0.97])
        # Filename: include xdim only if it's not "all"
        suffix = "_no_outliers" if remove_outliers else ""
        if xdim == "all":
            outfile = out_dir / f"{problem}_grouped_stacked_violin{suffix}.pdf"
        else:
            outfile = out_dir / f"{problem}_xdim{xdim}_grouped_stacked_violin{suffix}.pdf"
        _save_fig(outfile, fig)
        plt.close(fig)

    # Handle M2AX: group by xdim, create rows for each dim
    m2ax_df = df[df["problem"] == "m2ax"]
    if not m2ax_df.empty:
        for xdim, xdim_df in m2ax_df.groupby("xdim"):
            if xdim == "" or xdim == "all":
                continue

            # Get all dims and test_size levels for this xdim
            dims = sorted([d for d in xdim_df["dim"].unique() if d != "" and d != "all"])
            test_sizes = sorted([n for n in xdim_df["noise_label"].unique() if n != "unknown"])
            
            if not dims or not test_sizes:
                continue

            # Create subplots: one row per dim, each row has RRMSE and NIS side by side
            # Use 16:9 aspect ratio (PowerPoint standard)
            n_rows = len(dims)
            # Width = 16, height per row = 3 (so 3 rows = 9, maintaining 16:9 ratio)
            fig_width = 16
            fig_height = 3 * n_rows
            fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), sharex="col")
            
            # Handle single row case
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            level_colors = None
            
            # Create plots for each dim
            for row_idx, dim in enumerate(dims):
                dim_df = xdim_df[xdim_df["dim"] == dim]
                
                # Left: RRMSE
                level_colors = _plot_grouped_metric(
                    axes[row_idx, 0],
                    dim_df,
                    "RRMSE",
                    test_sizes,
                    "test_size",
                    is_m2ax=True,
                )
                
                # Right: NIS
                _ = _plot_grouped_metric(
                    axes[row_idx, 1],
                    dim_df,
                    "NIS",
                    test_sizes,
                    "test_size",
                    is_m2ax=True,
                )
                
                # Add dim label to left plot with subscript
                # Extract number from dim (e.g., "10D" -> "10")
                dim_number = dim.rstrip("D") if dim.endswith("D") else dim
                axes[row_idx, 0].text(0.02, 0.98, f"$N = {dim_number}D_x$", transform=axes[row_idx, 0].transAxes,
                                     fontsize=12, fontweight="bold", va="top", ha="left",
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            fig.suptitle(f"M2AX - $D_x$ = {xdim}", fontsize=22, fontweight="bold", y=0.995)

            # Build legend labels based on training fraction inferred from test_size
            from matplotlib.lines import Line2D

            if level_colors:
                handles = []
                for ts, col in level_colors.items():
                    try:
                        ts_val = float(ts)
                        train_frac = max(0.0, 1.0 - ts_val)
                        label = f"{train_frac:.3f} train (tsize={ts})"
                    except Exception:
                        label = f"tsize={ts}"
                    handles.append(
                        Line2D([0], [0], marker="s", color="none", markerfacecolor=col,
                              markersize=10, label=label)
                    )
                
                # Mean/median style handles
                mean_med_handles = [
                    Line2D([0], [0], color="#ff69b4", lw=2, label="Mean"),
                    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
                ]

                # Place legend on the right side of the figure
                fig.legend(
                    handles=handles + mean_med_handles,
                    loc="center right",
                    bbox_to_anchor=(1.0, 0.5),
                    title="Training data amount",
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                )

            # Adjust layout to leave space for legend on the right
            plt.tight_layout(rect=[0, 0, 0.85, 0.97])
            suffix = "_no_outliers" if remove_outliers else ""
            m2ax_outfile = out_dir / f"m2ax_xdim{xdim}_grouped_stacked_violin{suffix}.pdf"
            _save_fig(m2ax_outfile, fig)
            plt.close(fig)


def make_violin_plots_by_noise(df: pd.DataFrame, out_dir: Path, model_order: list[str], remove_outliers: bool = False) -> None:
    """Alternative version: noise levels on x-axis, models as colors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    # Color map for models
    model_color_map = {
        "gpplus": "#1f77b4",
        "tabpfn_v2.5": "#ff7f0e",
        "tabpfn_v2": "#ffa500",  # Darker orange for v2
        "gpytorch": "#2ca02c",
        "seek": "#9467bd",
    }
    _alt_model_colors = [
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    def _plot_grouped_metric_by_noise(
        ax,
        subset: pd.DataFrame,
        metric: str,
        noise_levels: list[str],
        models: list[str],
    ) -> dict:
        """
        Plot one metric axis where:
        - x-axis groups are noise levels
        - within each noise level group, we have one violin per model,
          distinguished by color.
        Returns a dict mapping model -> color actually used (for legend construction).
        """
        n_noise = len(noise_levels)
        n_models = len(models)
        if n_noise == 0 or n_models == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}

        width = 0.8  # total width per noise level group
        inner_width = width / max(n_models, 1)

        data_groups = []
        positions = []
        colors = []

        # Assign colors to each model
        model_colors: dict[str, str] = {}
        for mi, model in enumerate(models):
            if model in model_color_map:
                model_colors[model] = model_color_map[model]
            else:
                model_colors[model] = _alt_model_colors[mi % len(_alt_model_colors)]

        for ni, noise in enumerate(noise_levels):
            center = ni + 1  # noise level positions at 1,2,3,...
            start = center - width / 2 + inner_width / 2
            for mi, model in enumerate(models):
                arr = (
                    subset.loc[
                        (subset["model_family"] == model)
                        & (subset["noise_label"] == noise),
                        metric,
                    ]
                    .dropna()
                    .astype(float)
                    .values
                )
                if arr.size == 0:
                    continue
                # Remove outliers if requested
                if remove_outliers:
                    arr = remove_outliers_iqr(arr)
                if arr.size == 0:
                    continue
                xpos = start + mi * inner_width
                data_groups.append(arr)
                positions.append(xpos)
                colors.append(model_colors[model])

        if not data_groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            return {}

        # Add alternating background colors for each noise level group (white, light grey, white)
        bg_colors = ["white", "#e0e0e0", "white"]  # white, light grey, white
        for ni, noise in enumerate(noise_levels):
            if ni < len(noise_levels):
                bg_color = bg_colors[ni % len(bg_colors)]
                ax.axvspan(ni + 0.5, ni + 1.5, color=bg_color, alpha=0.3, zorder=0)

        parts = ax.violinplot(
            data_groups,
            positions=positions,
            widths=inner_width * 0.9,
            showmeans=False,
            showmedians=False,
            showextrema=True,
        )

        # Color each violin body according to its model
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_edgecolor("black")
            body.set_alpha(0.7)

        # Overlay mean (pink) and median (dashed black) for each group
        for xpos, arr in zip(positions, data_groups):
            mean_v = float(np.mean(arr))
            med_v = float(np.median(arr))
            ax.hlines(
                mean_v,
                xpos - inner_width * 0.5,
                xpos + inner_width * 0.5,
                colors="#ff69b4",
                linewidth=2,
            )
            ax.hlines(
                med_v,
                xpos - inner_width * 0.5,
                xpos + inner_width * 0.5,
                colors="black",
                linewidth=2,
                linestyles="--",
            )

        # X-ticks at noise level centers
        ax.set_xticks([ni + 1 for ni in range(n_noise)])
        ax.set_xticklabels(noise_levels, fontsize=14)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Disable default grid, only show horizontal grid lines
        ax.grid(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4, which="major")
        # Add vertical grid lines at boundaries between groups (after violins so they're visible)
        # Boundaries are at the edges of each background section: 0.5, 1.5, 2.5, etc.
        for boundary in [ni + 0.5 for ni in range(n_noise + 1)]:
            ax.axvline(boundary, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, zorder=10)

        return model_colors

    # Handle non-M2AX problems: group by problem and xdim, create rows for each dim
    non_m2ax_df = df[df["problem"] != "m2ax"]
    for (problem, xdim), prob_xdim_df in non_m2ax_df.groupby(["problem", "xdim"]):
        # Skip only if xdim is empty string or "all" (which means no xdim was found in filename)
        # We want to create plots for each Dx value (5, 10, 20, 40, 80) if they exist
        if xdim == "" or xdim == "all":
            continue

        # Get all dims and noise levels for this problem/xdim
        dims = sorted([d for d in prob_xdim_df["dim"].unique() if d != "" and d != "all"])
        noise_levels = sorted([n for n in prob_xdim_df["noise_label"].unique() if n != "unknown"])
        models = [m for m in model_order if m in prob_xdim_df["model_family"].unique()]
        
        if not dims or not noise_levels or not models:
            continue

        # Create subplots: one row per dim, each row has RRMSE and NIS side by side
        # Use 16:9 aspect ratio (PowerPoint standard)
        n_rows = len(dims)
        fig_width = 16
        fig_height = 3 * n_rows
        fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), sharex="col")
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        model_colors = None
        
        # Create plots for each dim
        for row_idx, dim in enumerate(dims):
            dim_df = prob_xdim_df[prob_xdim_df["dim"] == dim]
            
            # Left: RRMSE
            model_colors = _plot_grouped_metric_by_noise(
                axes[row_idx, 0],
                dim_df,
                "RRMSE",
                noise_levels,
                models,
            )
            
            # Right: NIS
            _ = _plot_grouped_metric_by_noise(
                axes[row_idx, 1],
                dim_df,
                "NIS",
                noise_levels,
                models,
            )
            
            # Add dim label to left plot with subscript
            dim_number = dim.rstrip("D") if dim.endswith("D") else dim
            axes[row_idx, 0].text(0.02, 0.98, f"$N = {dim_number}D_x$", transform=axes[row_idx, 0].transAxes,
                                 fontsize=16, fontweight="bold", va="top", ha="left",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Title: show D_x only if xdim is not "all"
        if xdim == "all":
            title = f"{format_problem_title(str(problem))}"
        else:
            title = f"{format_problem_title(str(problem))} - $D_x$ = {xdim}"
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)

        # Create legend with models
        from matplotlib.lines import Line2D

        if model_colors:
            model_handles = []
            for m, col in model_colors.items():
                label = display_label_for_model_family(m)
                model_handles.append(
                    Line2D([0], [0], marker="s", color="none", markerfacecolor=col, 
                          markersize=10, label=label)
                )
            # Mean/median style handles
            mean_med_handles = [
                Line2D([0], [0], color="#ff69b4", lw=2, label="Mean"),
                Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
            ]

            # Place legend on the right side of the figure
            fig.legend(
                handles=model_handles + mean_med_handles,
                loc="center right",
                bbox_to_anchor=(1.0, 0.5),
                title="Models",
                frameon=False,
                fontsize=14,
                title_fontsize=16,
            )

        plt.tight_layout(rect=[0, 0, 0.85, 0.97])
        # Filename: include xdim only if it's not "all"
        suffix = "_no_outliers" if remove_outliers else ""
        if xdim == "all":
            outfile = out_dir / f"{problem}_by_noise_grouped_stacked_violin{suffix}.pdf"
        else:
            outfile = out_dir / f"{problem}_xdim{xdim}_by_noise_grouped_stacked_violin{suffix}.pdf"
        _save_fig(outfile, fig)
        plt.close(fig)

    # Handle M2AX: group by xdim, create rows for each dim
    m2ax_df = df[df["problem"] == "m2ax"]
    if not m2ax_df.empty:
        for xdim, xdim_df in m2ax_df.groupby("xdim"):
            if xdim == "" or xdim == "all":
                continue

            # Get all dims and test_size levels for this xdim
            dims = sorted([d for d in xdim_df["dim"].unique() if d != "" and d != "all"])
            test_sizes = sorted([n for n in xdim_df["noise_label"].unique() if n != "unknown"])
            models = [m for m in model_order if m in xdim_df["model_family"].unique()]
            
            if not dims or not test_sizes or not models:
                continue

            # Create subplots: one row per dim, each row has RRMSE and NIS side by side
            n_rows = len(dims)
            fig_width = 16
            fig_height = 3 * n_rows
            fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), sharex="col")
            
            # Handle single row case
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            model_colors = None
            
            # Create plots for each dim
            for row_idx, dim in enumerate(dims):
                dim_df = xdim_df[xdim_df["dim"] == dim]
                
                # Left: RRMSE
                model_colors = _plot_grouped_metric_by_noise(
                    axes[row_idx, 0],
                    dim_df,
                    "RRMSE",
                    test_sizes,
                    models,
                )
                
                # Right: NIS
                _ = _plot_grouped_metric_by_noise(
                    axes[row_idx, 1],
                    dim_df,
                    "NIS",
                    test_sizes,
                    models,
                )
                
                # Add dim label to left plot with subscript
                dim_number = dim.rstrip("D") if dim.endswith("D") else dim
                axes[row_idx, 0].text(0.02, 0.98, f"$N = {dim_number}D_x$", transform=axes[row_idx, 0].transAxes,
                                     fontsize=12, fontweight="bold", va="top", ha="left",
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            fig.suptitle(f"M2AX - $D_x$ = {xdim}", fontsize=22, fontweight="bold", y=0.995)

            # Build legend with models
            from matplotlib.lines import Line2D

            if model_colors:
                model_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="none",
                        markerfacecolor=col,
                        markersize=10,
                        label=display_label_for_model_family(m),
                    )
                    for m, col in model_colors.items()
                ]
                # Mean/median style handles
                mean_med_handles = [
                    Line2D([0], [0], color="#ff69b4", lw=2, label="Mean"),
                    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
                ]

                # Place legend on the right side of the figure
                fig.legend(
                    handles=model_handles + mean_med_handles,
                    loc="center right",
                    bbox_to_anchor=(1.0, 0.5),
                    title="Models",
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                )

            plt.tight_layout(rect=[0, 0, 0.85, 0.97])
            suffix = "_no_outliers" if remove_outliers else ""
            m2ax_outfile = out_dir / f"m2ax_xdim{xdim}_by_noise_grouped_stacked_violin{suffix}.pdf"
            _save_fig(m2ax_outfile, fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make violin plots for RRMSE and NIS per problem.")
    parser.add_argument(
        "--gp_dir",
        type=Path,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_April28/20_runs_real_datasets"),
        help="Directory containing GP+ JSON results (gp_data section).",
    )
    parser.add_argument(
        "--gpytorch_dir",
        type=Path,
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_final/1_11"),
        default=None,
        help="Directory containing GPytorch JSON results (defaults to --gp_dir when omitted).",
    )
    parser.add_argument(
        "--tabpfn_v25_dir",
        type=Path,
        default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_April28/20_runs_real_datasets_pfn"),
        help="Directory containing TabPFN v2.5 JSON results.",
    )
    parser.add_argument(
        "--tabpfn_v2_dir",
        type=Path,
        default=None,
        # default=Path("C:/Users/forty/tyler_gpplus/gp-private/results_final/pfnv2_0_IQR"),
        help="Directory containing TabPFN v2 JSON results.",
    )
    parser.add_argument(
        "--seek_dir",
        type=Path,
        # default=Path("c:/Users/forty/tyler_gpplus/gp-private/seek_results"),
        default=None,
        help="Directory containing Seek JSON results.",
    )
    parser.add_argument(
        "--extra_gp",
        action="append",
        default=None,
        # default=["GP+ (student_t)=C:/Users/forty/tyler_gpplus/gp-private/results_April28/20_runs_Gaussian_student_t/",
        # "TabPFN v2.5 (student_t)=C:/Users/forty/tyler_gpplus/gp-private/results_April28/20_runs_TabPFN_student_t/",
        # "TabPFN v2.5 (CPU)=C:/Users/forty/tyler_gpplus/gp-private/results_April28/20_runs_TabPFN_cpu/"],
        metavar="NAME=PATH",
        help=(
            "Extra GP results directory: gp_data rows are tagged with NAME (model id). "
            "Repeat, e.g. --extra_gp median_runs=C:/out/median --extra_gp mean_runs=C:/out/mean. "
            "NAME must not match built-in ids (gpplus, gpytorch, ...). PATH must not contain '='."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Optional list of problem names to include (e.g., rosenbrock wing buckling borehole).",
    )
    parser.add_argument(
        "--noise_levels",
        nargs="*",
        default=["0.005", "0.05"],
        help=(
            "Noise labels to include (default: 0.005 0.05). "
            "For real datasets with no noise tags, rows are preserved automatically."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plots_violin_May01_real_datasets"),
        help="Directory to write violin plot pdfs.",
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        default=False,
        help="Remove outliers using IQR method before plotting. Creates additional plots with '_no_outliers' suffix.",
    )
    parser.add_argument(
        "--model_order",
        nargs="*",
        default=None,
        help=(
            "Optional explicit x-axis model order (model_family ids). "
            "Example: gpplus 'GP+ (PE)' 'GP+ (LOO)' tabpfn_v2.5 tabpfn_v2 gpytorch "
            "'GP+ (log)' 'GP+ (PE log)' 'GP+ (LOO log)' 'GPyTorch (log)'."
        ),
    )
    args = parser.parse_args()

    problems = set(args.problems) if args.problems else None
    extra_named_gp = parse_extra_named_gp_entries(args.extra_gp)
    df = collect_per_run_rows(
        args.gp_dir,
        args.gpytorch_dir,
        args.seek_dir,
        args.tabpfn_v25_dir,
        args.tabpfn_v2_dir,
        problems,
        extra_named_gp=extra_named_gp,
        noise_levels=args.noise_levels,
    )
    if df.empty:
        raise SystemExit("No data found; nothing to plot.")

    present_models = set(df["model_family"].dropna().unique())
    if args.model_order:
        model_order = [m for m in args.model_order if m in present_models]
        missing = [m for m in args.model_order if m not in present_models]
        if missing:
            print(f"Warning: --model_order entries with no data (skipped): {missing}")
    else:
        requested_model_order = get_model_order(
            gpytorch_dir=args.gpytorch_dir,
            tabpfn_v25_dir=args.tabpfn_v25_dir,
            tabpfn_v2_dir=args.tabpfn_v2_dir,
            seek_dir=args.seek_dir,
            extra_named_gp=extra_named_gp,
        )
        model_order = [m for m in requested_model_order if m in present_models]
    if not model_order:
        raise SystemExit("No model data found for the selected directories.")
    print(f"Model order: {model_order}")
    print(f"Rows: {len(df)}; models present: {sorted(present_models)}")
    # create_summary_table(df, args.output_dir, model_order)
    # create_time_tables(df, args.output_dir, model_order)
    # create_comprehensive_tables(df, args.output_dir, model_order)
    # create_comprehensive_tables_latex(df, args.output_dir, model_order)
    make_violin_plots(df, args.output_dir, model_order, remove_outliers=False)
    # make_violin_plots_by_noise(df, args.output_dir, model_order, remove_outliers=False)
    
    # Create additional plots with outliers removed if requested
    if args.remove_outliers:
        print("Removing outliers")
        # make_violin_plots(df, args.output_dir, model_order, remove_outliers=True)
        # make_violin_plots_by_noise(df, args.output_dir, model_order, remove_outliers=True)


if __name__ == "__main__":
    main()
