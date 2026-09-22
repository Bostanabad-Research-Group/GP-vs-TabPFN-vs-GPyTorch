"""
Build IDETC LaTeX tables (RRMSE/NIS + timing) with LOO columns.

Matches:
  results_paper/IDETC_figures_orig_ref/plots_gpplus_comparison/regression_results_table.tex
  IDETC_results_new/cost_table/timing_table_LOO.tex

Default outputs:
  results_IDETC/plots_gpplus_comparison/regression_results_table.tex   (median ± std)
  results_IDETC/plots_gpplus_comparison/regression_results_table2.tex  (mean ± std)
  results_IDETC/cost_table/timing_table_LOO.tex

Data defaults (all under results_IDETC):
  10_runs_logging_full_Gaussian_orig
  10_runs_logging_full_PE_orig
  10_runs_logging_full_Gaussian_LOO_orig
  10_runs_PFN_V2.5
  10_runs_PFN_V2.0
  10_runs_gpytorch_corrected_LBFGS

Run from experiments/:

  python experimental_utils/IDETC_create_tables.py
  python experimental_utils/IDETC_create_tables.py --only regression
  python experimental_utils/IDETC_create_tables.py --only timing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = EXPERIMENTS_DIR / "results_paper" / "benchmarks"

DIR_GP = "10_runs_logging_full_Gaussian"
DIR_PE = "10_runs_logging_full_PE"
DIR_LOO = "10_runs_logging_full_Gaussian_LOO"
DIR_PFN25 = "10_runs_PFN_V2.5"
DIR_PFN20 = "10_runs_PFN_V2.0"
DIR_GPYTORCH = "10_runs_gpytorch"

PROBLEM_XDIM_DEFAULT = {
    "wing": 10,
    "buckling": 4,
    "borehole": 8,
    "griewank": 20,
    "zakharov": 20,
    "dixon_price": 40,
    "rosenbrock": 80,
}

PROBLEM_DISPLAY = {
    "buckling": "Buckling",
    "buckling_logscale": "Buckling (log)",
    "borehole": "Borehole",
    "wing": "Wing Weight",
    "ackley": "Ackley",
    "griewank": "Griewank",
    "zakharov": "Zakharov",
    "zakharov_logscale": "Zakharov (log)",
    "dixon_price": "Dixon-Price",
    "rosenbrock": "Rosenbrock",
}

ROW_LAYOUT: List[Tuple[str, int, int]] = [
    ("buckling", 4, 5),
    ("buckling", 4, 20),
    ("borehole", 8, 5),
    ("borehole", 8, 20),
    ("wing", 10, 5),
    ("wing", 10, 20),
    ("ackley", 20, 5),
    ("ackley", 20, 20),
    ("griewank", 20, 5),
    ("griewank", 20, 20),
    ("zakharov", 20, 5),
    ("zakharov", 20, 20),
    ("ackley", 40, 5),
    ("ackley", 40, 20),
    ("dixon_price", 40, 5),
    ("dixon_price", 40, 20),
    ("rosenbrock", 80, 5),
    ("rosenbrock", 80, 20),
]

# Same as ROW_LAYOUT, but inserts buckling/zakharov log-scale blocks after each linear block.
ROW_LAYOUT_WITH_LOGSCALE: List[Tuple[str, int, int]] = [
    ("buckling", 4, 5),
    ("buckling", 4, 20),
    ("buckling_logscale", 4, 5),
    ("buckling_logscale", 4, 20),
    ("borehole", 8, 5),
    ("borehole", 8, 20),
    ("wing", 10, 5),
    ("wing", 10, 20),
    ("ackley", 20, 5),
    ("ackley", 20, 20),
    ("griewank", 20, 5),
    ("griewank", 20, 20),
    ("zakharov", 20, 5),
    ("zakharov", 20, 20),
    ("zakharov_logscale", 20, 5),
    ("zakharov_logscale", 20, 20),
    ("ackley", 40, 5),
    ("ackley", 40, 20),
    ("dixon_price", 40, 5),
    ("dixon_price", 40, 20),
    ("rosenbrock", 80, 5),
    ("rosenbrock", 80, 20),
]

LOGSCALE_ROOT = EXPERIMENTS_DIR / "results_logscale_study"
DIR_GP_LOG = "10_runs_logging_full_Gaussian_logscale"
DIR_PE_LOG = "10_runs_logging_full_PE_logscale"
DIR_LOO_LOG = "10_runs_logging_full_Gaussian_LOO_logscale"
DIR_GPYTORCH_LOG = "10_runs_gpytorch_corrected_LBFGS_logscale"
LOGSCALE_PROBLEMS = ("buckling", "zakharov")

# Side-by-side orig vs log columns (matches plots_gpplus_logscale_comparison_paper).
PAPER_MODEL_ORDER = (
    "gpplus",
    "GP+ (log)",
    "tabpfn_v2.5",
    "tabpfn_v2",
    "gpytorch",
    "GPyTorch (log)",
)

PAPER_MODEL_DISPLAY = {
    "gpplus": "GP+",
    "GP+ (log)": "GP+ (log)",
    "tabpfn_v2.5": "PFN 2.5",
    "tabpfn_v2": "PFN 2.0",
    "gpytorch": "GPyTorch",
    "GPyTorch (log)": "GPyTorch (log)",
}

PAPER_ROW_LAYOUT: List[Tuple[str, int, int]] = [
    ("buckling", 4, 5),
    ("buckling", 4, 20),
    ("zakharov", 20, 5),
    ("zakharov", 20, 20),
]

NOISE_LEVELS = ("0.002", "0.08")

# Internal ids used as keys in collected stats.
MODEL_ORDER = (
    "gpplus",
    "GP+ (PE)",
    "GP+ (LOO)",
    "tabpfn_v2.5",
    "tabpfn_v2",
    "gpytorch",
)

MODEL_DISPLAY = {
    "gpplus": "GP+",
    "GP+ (PE)": "GP+ (PE)",
    "GP+ (LOO)": "GP+ (LOO)",
    "tabpfn_v2.5": "PFN 2.5",
    "tabpfn_v2": "PFN 2.0",
    "gpytorch": "GPyTorch",
}


def _parse_float_token(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _extract_meta(path: Path, model_root: Path) -> Dict[str, Any]:
    rel = path.relative_to(model_root)
    problem = rel.parts[0].lower() if rel.parts else ""
    if problem.endswith(".json"):
        problem = ""
    stem = path.stem
    stem_l = stem.lower()

    # Filename-based problem detection (flat dirs / mismatched folders).
    for name in (
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
    ):
        if name in stem_l:
            problem = "dixon_price" if name == "dixonprice" else name
            break

    noise_match = re.search(r"noiseTest(?P<t>[0-9.]+)_noiseTrain(?P<tr>[0-9.]+)", stem)
    noise_test = _parse_float_token(noise_match.group("t")) if noise_match else ""
    noise_train = _parse_float_token(noise_match.group("tr")) if noise_match else ""

    xdim_match = re.search(r"([0-9]+)Dx", stem) or re.search(r"([0-9]+)xdim", stem)
    xdim = int(xdim_match.group(1)) if xdim_match else PROBLEM_XDIM_DEFAULT.get(problem)

    dim_match = re.search(r"([0-9]+)Dn", stem)
    if dim_match:
        n_mult = int(dim_match.group(1))
    else:
        dim_match = re.search(r"_([0-9]+)D_", stem)
        n_mult = int(dim_match.group(1)) if dim_match else None

    return {
        "problem": problem,
        "xdim": xdim,
        "n_mult": n_mult,
        "noise_test": noise_test,
        "noise_train": noise_train,
    }


def _iter_result_jsons(model_root: Path) -> Iterable[Path]:
    if not model_root.exists():
        return
    for path in model_root.rglob("*.json"):
        p = str(path).lower()
        if "trainer_analysis" in p or "gp_trainer_analysis" in p:
            continue
        if "backup" in path.parts:
            continue
        yield path


def _section_for_model(model_id: str) -> str:
    return "tabpfn_data" if model_id.startswith("tabpfn") else "gp_data"


def _as_finite_float(v: Any) -> Optional[float]:
    """Parse a numeric metric; skip None / NaN / Inf (failed GPyTorch runs)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _metric_arrays(metrics: List[Dict], model_id: str) -> Dict[str, np.ndarray]:
    rrmse, nis, train, pred = [], [], [], []
    is_pfn = model_id.startswith("tabpfn")
    for m in metrics:
        if not isinstance(m, dict):
            continue
        # Failed GPyTorch runs write RRMSE/NIS as NaN; JSON summary already
        # excludes those. Match that by dropping non-finite per-run values.
        r = _as_finite_float(m.get("RRMSE"))
        if r is not None:
            rrmse.append(r)
        n = _as_finite_float(m.get("NIS"))
        if n is not None:
            nis.append(n)

        # GP+ writes Train_Time; TabPFN / some dumps use Training_Time.
        t_train = _as_finite_float(m.get("Training_Time"))
        if t_train is None:
            t_train = _as_finite_float(m.get("Train_Time"))
        if t_train is not None:
            train.append(t_train)

        t_pred = _as_finite_float(m.get("Prediction_Time"))
        if t_pred is None and is_pfn:
            t_pred = _as_finite_float(m.get("Time"))
        if t_pred is not None:
            pred.append(t_pred)

    return {
        "RRMSE": np.asarray(rrmse, dtype=float),
        "NIS": np.asarray(nis, dtype=float),
        "Train": np.asarray(train, dtype=float),
        "Pred": np.asarray(pred, dtype=float),
    }


def _median_std(arr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if arr.size == 0:
        return None, None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.median(finite)), float(np.std(finite, ddof=0))


def _mean_std(arr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if arr.size == 0:
        return None, None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _center_std(arr: np.ndarray, center: str) -> Tuple[Optional[float], Optional[float]]:
    if center == "mean":
        return _mean_std(arr)
    return _median_std(arr)


StatsKey = Tuple[str, int, int, str, str]
StatsVal = Dict[str, Tuple[Optional[float], Optional[float]]]


def collect_stats(
    model_dirs: Dict[str, Path],
    *,
    center: str = "median",
) -> Dict[StatsKey, StatsVal]:
    """key -> {metric: (center, std)} for RRMSE, NIS, Train, Pred.

    ``center`` is ``\"median\"`` or ``\"mean\"`` (non-finite per-run values dropped).
    """
    if center not in ("median", "mean"):
        raise ValueError(f"center must be 'median' or 'mean', got {center!r}")
    table: Dict[StatsKey, StatsVal] = {}
    for model_id, model_root in model_dirs.items():
        if model_root is None or not model_root.exists():
            print(f"Warning: missing model dir for {model_id}: {model_root}")
            continue
        section = _section_for_model(model_id)
        for path in _iter_result_jsons(model_root):
            meta = _extract_meta(path, model_root)
            if not meta["problem"] or meta["xdim"] is None or meta["n_mult"] is None:
                continue
            if not meta["noise_test"] or not meta["noise_train"]:
                continue
            if meta["noise_test"] != meta["noise_train"]:
                continue
            if meta["noise_test"] not in NOISE_LEVELS:
                continue

            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data.get(section, {}).get("metrics")
            if not isinstance(metrics, list):
                continue
            arrays = _metric_arrays(metrics, model_id)
            key: StatsKey = (
                str(meta["problem"]),
                int(meta["xdim"]),
                int(meta["n_mult"]),
                str(meta["noise_test"]),
                model_id,
            )
            table[key] = {
                "RRMSE": _center_std(arrays["RRMSE"], center),
                "NIS": _center_std(arrays["NIS"], center),
                "Train": _center_std(arrays["Train"], center),
                "Pred": _center_std(arrays["Pred"], center),
            }
    return table


def _fmt_center_token(center: Optional[float]) -> Optional[str]:
    """Center value as printed in the table (same rounding as ``_fmt_metric_value``)."""
    if center is None or not np.isfinite(center):
        return None
    am = abs(center)
    if am >= 100:
        return f"{center:.1f}"
    if am >= 10:
        return f"{center:.2f}"
    return f"{center:.3f}"


def _display_center(center: Optional[float]) -> Optional[float]:
    """Numeric value after table rounding — used for tie-aware best-model bolding."""
    token = _fmt_center_token(center)
    return float(token) if token is not None else None


def _fmt_metric_value(center: Optional[float], std: Optional[float]) -> str:
    """Compact ``center ± std`` formatting for regression tables."""
    if (
        center is None
        or std is None
        or not np.isfinite(center)
        or not np.isfinite(std)
    ):
        return "--"
    med_s = _fmt_center_token(center)
    astd = abs(std)
    if astd >= 10:
        std_s = f"{std:.1f}"
    else:
        std_s = f"{std:.2f}"
    return f"{med_s} $\\pm$ {std_s}"


def _fmt_time_value(median: Optional[float], std: Optional[float]) -> str:
    """Train/Fit/Inf formatting (mostly 2 decimal places)."""
    if (
        median is None
        or std is None
        or not np.isfinite(median)
        or not np.isfinite(std)
    ):
        return "--"
    med_s = f"{median:.2f}"
    std_s = f"{std:.3f}" if 0 < abs(std) < 0.01 else f"{std:.2f}"
    return f"{med_s} $\\pm$ {std_s}"


def _fmt_pred_time_value(median: Optional[float], std: Optional[float]) -> str:
    """Pred (s) formatting — always 3 decimal places for median and std."""
    if (
        median is None
        or std is None
        or not np.isfinite(median)
        or not np.isfinite(std)
    ):
        return "--"
    return f"{median:.3f} $\\pm$ {std:.3f}"


def _bold_if(txt: str, bold: bool) -> str:
    return f"\\textbf{{{txt}}}" if bold and txt != "--" else txt


def _best_display_and_cells(
    pairs: List[Tuple[Optional[float], Optional[float]]],
) -> List[str]:
    """Format cells; bold every model whose displayed center equals the row-best."""
    displayed = [_display_center(v[0]) for v in pairs]
    finite = [d for d in displayed if d is not None]
    best = min(finite) if finite else None
    cells: List[str] = []
    for (center, std), disp in zip(pairs, displayed):
        cells.append(
            _bold_if(
                _fmt_metric_value(center, std),
                best is not None and disp is not None and disp == best,
            )
        )
    return cells


def make_regression_results_table(
    stats: Dict[StatsKey, StatsVal],
    *,
    center: str = "median",
    row_layout: Optional[Sequence[Tuple[str, int, int]]] = None,
    problem_col_width: str = "1.6cm",
    caption_extra: str = "",
) -> str:
    """Full RRMSE|NIS table with LOO. ``center`` is ``median`` or ``mean``."""
    if center not in ("median", "mean"):
        raise ValueError(f"center must be 'median' or 'mean', got {center!r}")
    layout = list(row_layout) if row_layout is not None else list(ROW_LAYOUT)
    center_tex = "median" if center == "median" else "mean"
    lines: List[str] = []
    lines.append("% Auto-generated by experimental_utils/IDETC_create_tables.py")
    lines.append("\\begin{table}[!h]")
    lines.append("    \\centering")
    lines.append("    {")
    lines.append("    \\setlength{\\tabcolsep}{2.5pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.9}")
    lines.append("    \\tiny")
    lines.append(
        "    \\caption{RRMSE and NIS statistics across benchmark problems for GP+, GP+ (PE), GP+ (LOO), "
        f"PFN 2.5, PFN 2.0, and GPyTorch. Entries are {center_tex} $\\pm$ std over runs. "
        f"Best models for a given problem (lowest {center_tex} per metric) are in bold."
        f"{caption_extra}}}"
    )
    lines.append(
        "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}"
        f">{{\\centering\\arraybackslash}}p{{{problem_col_width}}}"
        ">{\\centering\\arraybackslash}p{0.5cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        "|c|c|c|c|c|c!{\\vrule width 1.0pt}c|c|c|c|c|c}"
    )
    lines.append("    \\toprule")
    lines.append(
        "    & & & & \\multicolumn{6}{c!{\\vrule width 1.0pt}}{\\textbf{RRMSE}} & "
        "\\multicolumn{6}{c}{\\textbf{NIS}} \\\\"
    )
    lines.append("    \\midrule")
    lines.append(
        "    \\textbf{Problem} & \\textbf{N} & \\textbf{D$_x$} & \\textbf{Noise} & "
        "\\textbf{GP+} & \\textbf{GP+ (PE)} & \\textbf{GP+ (LOO)} & "
        "\\textbf{PFN 2.5} & \\textbf{PFN 2.0} & \\textbf{GPyTorch} & "
        "\\textbf{GP+} & \\textbf{GP+ (PE)} & \\textbf{GP+ (LOO)} & "
        "\\textbf{PFN 2.5} & \\textbf{PFN 2.0} & \\textbf{GPyTorch} \\\\"
    )
    lines.append("    \\midrule")

    for idx, (problem_key, xdim, n_mult) in enumerate(layout):
        name = PROBLEM_DISPLAY[problem_key]
        # One problem name per (problem, Dx) block; no rule between 5Dx / 20Dx.
        first_of_block = idx == 0 or (layout[idx - 1][0], layout[idx - 1][1]) != (
            problem_key,
            xdim,
        )
        first_noise = True
        for noise in NOISE_LEVELS:
            if first_noise:
                if first_of_block:
                    left = (
                        f"\\textbf{{{name}}} & \\textbf{{{n_mult}D$_x$}} & "
                        f"\\textbf{{{xdim}}} & {noise}"
                    )
                else:
                    left = f" & \\textbf{{{n_mult}D$_x$}} & \\textbf{{{xdim}}} & {noise}"
            else:
                left = f" &  &  & {noise}"
            row = {
                m: stats.get((problem_key, xdim, n_mult, noise, m), {})
                for m in MODEL_ORDER
            }
            rrmse = [row[m].get("RRMSE", (None, None)) for m in MODEL_ORDER]
            nis = [row[m].get("NIS", (None, None)) for m in MODEL_ORDER]
            rrmse_cells = _best_display_and_cells(rrmse)
            nis_cells = _best_display_and_cells(nis)
            lines.append("    " + left + " & " + " & ".join(rrmse_cells + nis_cells) + " \\\\[2pt]")
            first_noise = False

        if idx < len(layout) - 1:
            nxt_key, nxt_xdim, _ = layout[idx + 1]
            if (problem_key, xdim) != (nxt_key, nxt_xdim):
                lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular*}")
    lines.append("    }")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _remap_problem_keys(
    stats: Dict[StatsKey, StatsVal],
    *,
    src_problems: Sequence[str],
    suffix: str,
) -> Dict[StatsKey, StatsVal]:
    """Copy entries for selected problems under ``{problem}{suffix}`` keys."""
    out: Dict[StatsKey, StatsVal] = {}
    wanted = set(src_problems)
    for (problem, xdim, n_mult, noise, model_id), val in stats.items():
        if problem not in wanted:
            continue
        out[(f"{problem}{suffix}", xdim, n_mult, noise, model_id)] = val
    return out


def collect_stats_with_logscale(
    model_dirs: Dict[str, Path],
    *,
    logscale_root: Path = LOGSCALE_ROOT,
    center: str = "median",
) -> Dict[StatsKey, StatsVal]:
    """Linear IDETC stats plus remapped buckling/zakharov log-scale rows."""
    stats = collect_stats(model_dirs, center=center)

    log_gp_dirs = {
        "gpplus": logscale_root / DIR_GP_LOG,
        "GP+ (PE)": logscale_root / DIR_PE_LOG,
        "GP+ (LOO)": logscale_root / DIR_LOO_LOG,
        "gpytorch": logscale_root / DIR_GPYTORCH_LOG,
    }
    print("Collecting log-scale stats from:")
    for mid, p in log_gp_dirs.items():
        print(f"  {MODEL_DISPLAY.get(mid, mid)} (log): {p} ({'OK' if p.exists() else 'MISSING'})")

    log_stats = collect_stats(log_gp_dirs, center=center)
    stats.update(_remap_problem_keys(log_stats, src_problems=LOGSCALE_PROBLEMS, suffix="_logscale"))

    # PFNs have no separate log-scale runs; reuse linear PFN metrics on log rows.
    pfn_dirs = {
        "tabpfn_v2.5": model_dirs["tabpfn_v2.5"],
        "tabpfn_v2": model_dirs["tabpfn_v2"],
    }
    pfn_stats = collect_stats(pfn_dirs, center=center)
    stats.update(_remap_problem_keys(pfn_stats, src_problems=LOGSCALE_PROBLEMS, suffix="_logscale"))
    return stats


def collect_paper_logscale_sidebyside_stats(
    *,
    logscale_root: Path = LOGSCALE_ROOT,
    center: str = "median",
) -> Dict[StatsKey, StatsVal]:
    """Stats for paper columns: GP+/GPyTorch linear+log, PFN 2.5/2.0 (buckling, zakharov)."""
    linear_dirs = {
        "gpplus": logscale_root / DIR_GP,
        "tabpfn_v2.5": logscale_root / DIR_PFN25,
        "tabpfn_v2": logscale_root / DIR_PFN20,
        "gpytorch": logscale_root / "10_runs_gpytorch_corrected_LBFGS",
    }
    log_dirs = {
        "GP+ (log)": logscale_root / DIR_GP_LOG,
        "GPyTorch (log)": logscale_root / DIR_GPYTORCH_LOG,
    }
    print("Collecting paper side-by-side stats from:")
    for mid, p in {**linear_dirs, **log_dirs}.items():
        print(f"  {PAPER_MODEL_DISPLAY.get(mid, mid)}: {p} ({'OK' if p.exists() else 'MISSING'})")
    stats = collect_stats(linear_dirs, center=center)
    stats.update(collect_stats(log_dirs, center=center))
    return stats


def make_paper_logscale_comparison_table(
    stats: Dict[StatsKey, StatsVal],
    *,
    center: str = "median",
) -> str:
    """RRMSE|NIS table for buckling + zakharov with paper model columns."""
    if center not in ("median", "mean"):
        raise ValueError(f"center must be 'median' or 'mean', got {center!r}")
    center_tex = "median" if center == "median" else "mean"
    n_models = len(PAPER_MODEL_ORDER)
    model_headers = " & ".join(f"\\textbf{{{PAPER_MODEL_DISPLAY[m]}}}" for m in PAPER_MODEL_ORDER)
    col_spec = "|" + "|".join(["c"] * n_models) + "!{\\vrule width 1.0pt}" + "|".join(["c"] * n_models)

    lines: List[str] = []
    lines.append("% Auto-generated by experimental_utils/IDETC_create_tables.py (--paper_logscale)")
    lines.append("\\begin{table}[!h]")
    lines.append("    \\centering")
    lines.append("    {")
    lines.append("    \\setlength{\\tabcolsep}{2.5pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.9}")
    lines.append("    \\tiny")
    lines.append(
        "    \\caption{RRMSE and NIS for Buckling and Zakharov comparing GP+, GP+ (log), "
        f"PFN 2.5, PFN 2.0, GPyTorch, and GPyTorch (log). Entries are {center_tex} $\\pm$ std over runs. "
        f"Best models for a given setting (lowest {center_tex} per metric) are in bold.}}"
    )
    lines.append(
        "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}"
        ">{\\centering\\arraybackslash}p{1.6cm}"
        ">{\\centering\\arraybackslash}p{0.5cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        f"{col_spec}}}"
    )
    lines.append("    \\toprule")
    lines.append(
        f"    & & & & \\multicolumn{{{n_models}}}{{c!{{\\vrule width 1.0pt}}}}{{\\textbf{{RRMSE}}}} & "
        f"\\multicolumn{{{n_models}}}{{c}}{{\\textbf{{NIS}}}} \\\\"
    )
    lines.append("    \\midrule")
    lines.append(
        "    \\textbf{Problem} & \\textbf{N} & \\textbf{D$_x$} & \\textbf{Noise} & "
        f"{model_headers} & {model_headers} \\\\"
    )
    lines.append("    \\midrule")

    layout = PAPER_ROW_LAYOUT
    for idx, (problem_key, xdim, n_mult) in enumerate(layout):
        name = PROBLEM_DISPLAY[problem_key]
        first_of_block = idx == 0 or (layout[idx - 1][0], layout[idx - 1][1]) != (
            problem_key,
            xdim,
        )
        first_noise = True
        for noise in NOISE_LEVELS:
            if first_noise:
                if first_of_block:
                    left = (
                        f"\\textbf{{{name}}} & \\textbf{{{n_mult}D$_x$}} & "
                        f"\\textbf{{{xdim}}} & {noise}"
                    )
                else:
                    left = f" & \\textbf{{{n_mult}D$_x$}} & \\textbf{{{xdim}}} & {noise}"
            else:
                left = f" &  &  & {noise}"
            row = {
                m: stats.get((problem_key, xdim, n_mult, noise, m), {})
                for m in PAPER_MODEL_ORDER
            }
            rrmse = [row[m].get("RRMSE", (None, None)) for m in PAPER_MODEL_ORDER]
            nis = [row[m].get("NIS", (None, None)) for m in PAPER_MODEL_ORDER]
            rrmse_cells = _best_display_and_cells(rrmse)
            nis_cells = _best_display_and_cells(nis)
            lines.append("    " + left + " & " + " & ".join(rrmse_cells + nis_cells) + " \\\\[2pt]")
            first_noise = False

        if idx < len(layout) - 1:
            nxt_key, nxt_xdim, _ = layout[idx + 1]
            if (problem_key, xdim) != (nxt_key, nxt_xdim):
                lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular*}")
    lines.append("    }")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def make_timing_table(stats: Dict[StatsKey, StatsVal]) -> str:
    """Training/inference table with LOO — matches timing_table_LOO.tex."""
    lines: List[str] = []
    lines.append("% Auto-generated by experimental_utils/make_IDETC_tables.py")
    lines.append("\\begin{table*}[!h]")
    lines.append("    \\centering")
    lines.append("    \\setlength{\\tabcolsep}{2.5pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.9}")
    lines.append("    \\tiny")
    lines.append(
        "    \\caption{Training and inference time (median $\\pm$ std) for GP+, GP+ (PE), GP+ (LOO), "
        "PFN 2.5, PFN 2.0, and GPyTorch across benchmark problems.}"
    )
    lines.append(
        "    \\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}"
        ">{\\centering\\arraybackslash}p{1.6cm}"
        ">{\\centering\\arraybackslash}p{0.5cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        ">{\\centering\\arraybackslash}p{0.4cm}"
        "|c|c|c|c|c|c|c|c|c|c|c|c}"
    )
    lines.append("    \\toprule")
    lines.append(
        "    \\multicolumn{4}{c|}{} & "
        "\\multicolumn{2}{c|}{\\textbf{GP+}} & "
        "\\multicolumn{2}{c|}{\\textbf{GP+ (PE)}} & "
        "\\multicolumn{2}{c|}{\\textbf{GP+ (LOO)}} & "
        "\\multicolumn{2}{c|}{\\textbf{PFN 2.5}} & "
        "\\multicolumn{2}{c|}{\\textbf{PFN 2.0}} & "
        "\\multicolumn{2}{c}{\\textbf{GPyTorch}} \\\\"
    )
    lines.append("    \\midrule")
    lines.append(
        "    \\textbf{Problem} & \\textbf{N} & \\textbf{D$_x$} & \\textbf{Noise} & "
        "\\textbf{Train (s)} & \\textbf{Pred (s)} & "
        "\\textbf{Train (s)} & \\textbf{Pred (s)} & "
        "\\textbf{Train (s)} & \\textbf{Pred (s)} & "
        "\\textbf{Fit (s)} & \\textbf{Inf (s)} & "
        "\\textbf{Fit (s)} & \\textbf{Inf (s)} & "
        "\\textbf{Train (s)} & \\textbf{Pred (s)} \\\\"
    )
    lines.append("    \\midrule")

    for idx, (problem_key, xdim, n_mult) in enumerate(ROW_LAYOUT):
        name = PROBLEM_DISPLAY[problem_key]
        first_noise = True
        for noise in NOISE_LEVELS:
            # Match old TABLE A2 layout: repeat problem name on each N group.
            if first_noise:
                left = (
                    f"\\textbf{{{name}}} & \\textbf{{{n_mult}D$_x$}} & "
                    f"\\textbf{{{xdim}}} & {noise}"
                )
            else:
                left = f" &  &  & {noise}"
            cells: List[str] = []
            for m in MODEL_ORDER:
                entry = stats.get((problem_key, xdim, n_mult, noise, m), {})
                train = entry.get("Train", (None, None))
                pred = entry.get("Pred", (None, None))
                cells.append(_fmt_time_value(train[0], train[1]))
                # PFN Inf stays on train-style formatting; Pred columns use 3 dp.
                if m.startswith("tabpfn"):
                    cells.append(_fmt_time_value(pred[0], pred[1]))
                else:
                    cells.append(_fmt_pred_time_value(pred[0], pred[1]))
            lines.append("    " + left + " & " + " & ".join(cells) + " \\\\[2pt]")
            first_noise = False

        if idx < len(ROW_LAYOUT) - 1:
            nxt_key, nxt_xdim, _ = ROW_LAYOUT[idx + 1]
            if (problem_key, xdim) != (nxt_key, nxt_xdim):
                lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular*}")
    lines.append("    \\label{tab timing loo}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def _default_model_dirs(root: Path, gpytorch_dir: Path) -> Dict[str, Path]:
    return {
        "gpplus": root / DIR_GP,
        "GP+ (PE)": root / DIR_PE,
        "GP+ (LOO)": root / DIR_LOO,
        "tabpfn_v2.5": root / DIR_PFN25,
        "tabpfn_v2": root / DIR_PFN20,
        "gpytorch": gpytorch_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IDETC regression + timing LaTeX tables.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--gpytorch_dir",
        type=Path,
        default=None,
        help="Default: <root>/10_runs_gpytorch_corrected_LBFGS",
    )
    parser.add_argument(
        "--regression_out",
        type=Path,
        default=None,
        help="Default: <root>/plots_gpplus_comparison/regression_results_table.tex",
    )
    parser.add_argument(
        "--regression_mean_out",
        type=Path,
        default=None,
        help="Default: <root>/plots_gpplus_comparison/regression_results_table2.tex (mean±std)",
    )
    parser.add_argument(
        "--timing_out",
        type=Path,
        default=None,
        help="Default: <root>/cost_table/timing_table_LOO.tex",
    )
    parser.add_argument(
        "--with_logscale",
        action="store_true",
        help="Also write regression tables with Buckling/Zakharov log-scale rows.",
    )
    parser.add_argument(
        "--logscale_root",
        type=Path,
        default=LOGSCALE_ROOT,
        help="Default: experiments/results_logscale_study",
    )
    parser.add_argument(
        "--paper_logscale",
        action="store_true",
        help="Write buckling/zakharov paper side-by-side table (GP+/log, PFN, GPyTorch/log).",
    )
    parser.add_argument(
        "--paper_out",
        type=Path,
        default=None,
        help="Default: results_logscale_study/plots_gpplus_logscale_comparison_paper/regression_results_table.tex",
    )
    parser.add_argument("--only", choices=("both", "regression", "timing", "paper"), default="both")
    args = parser.parse_args()

    root = args.root
    gpytorch_dir = args.gpytorch_dir or (root / DIR_GPYTORCH)
    model_dirs = _default_model_dirs(root, gpytorch_dir)
    regression_out = args.regression_out or (root / "plots_gpplus_comparison" / "regression_results_table.tex")
    regression_mean_out = args.regression_mean_out or (
        root / "plots_gpplus_comparison" / "regression_results_table2.tex"
    )
    timing_out = args.timing_out or (root / "cost_table" / "timing_table_LOO.tex")
    paper_out = args.paper_out or (
        LOGSCALE_ROOT / "plots_gpplus_logscale_comparison_paper" / "regression_results_table.tex"
    )

    if args.only == "paper" or args.paper_logscale:
        for center, out in (
            ("median", paper_out),
            ("mean", paper_out.with_name("regression_results_table2.tex")),
        ):
            stats_paper = collect_paper_logscale_sidebyside_stats(
                logscale_root=args.logscale_root,
                center=center,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                make_paper_logscale_comparison_table(stats_paper, center=center),
                encoding="utf-8",
            )
            print(f"Wrote {out}")
        if args.only == "paper":
            return

    print("Collecting stats from:")
    for mid, p in model_dirs.items():
        print(f"  {MODEL_DISPLAY.get(mid, mid)}: {p} ({'OK' if p.exists() else 'MISSING'})")

    if args.only in ("both", "regression"):
        stats_med = collect_stats(model_dirs, center="median")
        print(f"Collected {len(stats_med)} median entries")
        regression_out.parent.mkdir(parents=True, exist_ok=True)
        regression_out.write_text(
            make_regression_results_table(stats_med, center="median"),
            encoding="utf-8",
        )
        print(f"Wrote {regression_out}")

        stats_mean = collect_stats(model_dirs, center="mean")
        print(f"Collected {len(stats_mean)} mean entries")
        regression_mean_out.parent.mkdir(parents=True, exist_ok=True)
        regression_mean_out.write_text(
            make_regression_results_table(stats_mean, center="mean"),
            encoding="utf-8",
        )
        print(f"Wrote {regression_mean_out}")

        if args.with_logscale:
            log_caption = (
                " Buckling (log) / Zakharov (log) rows use log-scale GP+/GPyTorch; "
                "PFN columns reuse the corresponding linear runs."
            )
            for center, base_out in (("median", regression_out), ("mean", regression_mean_out)):
                stats_log = collect_stats_with_logscale(
                    model_dirs,
                    logscale_root=args.logscale_root,
                    center=center,
                )
                out = base_out.with_name(base_out.stem + "_with_logscale" + base_out.suffix)
                out.write_text(
                    make_regression_results_table(
                        stats_log,
                        center=center,
                        row_layout=ROW_LAYOUT_WITH_LOGSCALE,
                        problem_col_width="2.1cm",
                        caption_extra=log_caption,
                    ),
                    encoding="utf-8",
                )
                print(f"Wrote {out}")

    if args.only in ("both", "timing"):
        stats_med = collect_stats(model_dirs, center="median")
        timing_out.parent.mkdir(parents=True, exist_ok=True)
        timing_out.write_text(make_timing_table(stats_med), encoding="utf-8")
        print(f"Wrote {timing_out}")


if __name__ == "__main__":
    main()
