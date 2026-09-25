"""Compare a fresh run with the archived paper results.

Writes ``results/summary.md`` and, under each suite folder,
``<suite>_results_comparison/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments_regression" / "experimental_utils"))
sys.path.insert(0, str(ROOT / "experiments_regression"))
sys.path.insert(0, str(ROOT / "experiments_BO"))

import IDETC_create_tables as tables  # noqa: E402
from plot_BO import collect_runs  # noqa: E402
from plot_BO_IDETC import IDETC_PROBLEMS, MAXIMIZATION_PROBLEMS, NOISE_HIGH  # noqa: E402
from result_paths import RESULTS, comparison_dir, new_results, original_results  # noqa: E402

REG_MODELS = {
    "gpplus": "10_runs_logging_full_Gaussian",
    "GP+ (PE)": "10_runs_logging_full_PE",
    "GP+ (LOO)": "10_runs_logging_full_Gaussian_LOO",
    "tabpfn_v2.5": "10_runs_PFN_V2.5",
    "tabpfn_v2": "10_runs_PFN_V2.0",
    "gpytorch": "10_runs_gpytorch",
}
BO_MODELS = [
    ("GP+", "GP+"),
    ("PFN_V2.0", "PFN 2.0"),
    ("PFN_V2.5", "PFN 2.5"),
]


def _dirs(root: Path) -> dict[str, Path]:
    found = {}
    for key, name in REG_MODELS.items():
        path = root / name
        if path.is_dir():
            found[key] = path
    return found


def _regression() -> pd.DataFrame:
    paper = tables.collect_stats(_dirs(original_results("regression") / "benchmarks"))
    new = tables.collect_stats(_dirs(new_results("regression") / "benchmarks"))
    rows = []
    for key, paper_metrics in paper.items():
        new_metrics = new.get(key)
        problem, xdim, n_mult, noise, model = key
        for metric in ("RRMSE", "NIS"):
            p_med, _p_std = paper_metrics[metric]
            n_med = None if new_metrics is None else new_metrics[metric][0]
            rows.append(
                {
                    "problem": problem,
                    "Dx": xdim,
                    "N_over_Dx": n_mult,
                    "noise": noise,
                    "model": model,
                    "metric": metric,
                    "paper_median": p_med,
                    "new_median": n_med,
                    "abs_diff": None if p_med is None or n_med is None else abs(n_med - p_med),
                    "rel_diff": None
                    if p_med is None or n_med is None or abs(p_med) < 1e-12
                    else abs(n_med - p_med) / abs(p_med),
                }
            )
    return pd.DataFrame(rows)


def _bo_finals(root: Path) -> dict[tuple, float]:
    out = {}
    for folder, label in BO_MODELS:
        path = root / folder
        if not path.is_dir():
            continue
        data, _stems = collect_runs(path, use_clean_y=False)
        for prob_folder, dim_key, title in IDETC_PROBLEMS:
            runs = data.get((prob_folder, dim_key, NOISE_HIGH), [])
            finals = [float(hist[-1]) for _rid, hist, _t in runs if hist]
            if not finals:
                continue
            out[(title, label)] = float(np.median(finals))
    return out


def _bo() -> pd.DataFrame:
    paper = _bo_finals(original_results("bo"))
    new = _bo_finals(new_results("bo"))
    rows = []
    for key, p_med in paper.items():
        title, label = key
        n_med = new.get(key)
        rows.append(
            {
                "problem": title,
                "model": label,
                "paper_median": p_med,
                "new_median": n_med,
                "abs_diff": None if n_med is None else abs(n_med - p_med),
                "rel_diff": None if n_med is None or abs(p_med) < 1e-12 else abs(n_med - p_med) / abs(p_med),
                "goal": "maximize" if any(
                    title.startswith(name) and (folder, dim) in MAXIMIZATION_PROBLEMS
                    for folder, dim, name in IDETC_PROBLEMS
                ) else "minimize",
            }
        )
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame, path: Path, cols: list[str]) -> None:
    if frame.empty:
        path.write_text("No overlapping results yet.\n", encoding="utf-8")
        return
    show = frame[cols]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in show.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append("" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REG_COLORS = {
    "gpplus": "#42A5F5",
    "GP+ (PE)": "#E53935",
    "GP+ (LOO)": "#43A047",
    "gpytorch": "#6D4C41",
    "tabpfn_v2": "#FB8C00",
    "tabpfn_v2.5": "#9C27B0",
}
BO_ORDER = [
    "Buckling",
    "Borehole",
    "Wing",
    "Ackley 20D",
    "Griewank 20D",
    "Zakharov 20D",
    "Ackley 40D",
    "Dixon-Price 40D",
]
BO_MODELS_ORDER = ["GP+", "PFN 2.0", "PFN 2.5"]


def _plot_regression(frame: pd.DataFrame, path: Path) -> None:
    matched = frame.dropna(subset=["paper_median", "new_median"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, metric in zip(axes, ("RRMSE", "NIS")):
        part = matched[matched["metric"] == metric]
        lo = min(part["paper_median"].min(), part["new_median"].min())
        hi = max(part["paper_median"].max(), part["new_median"].max())
        lo = max(lo * 0.7, 1e-4)
        hi = hi * 1.4
        ax.plot([lo, hi], [lo, hi], color="0.55", linewidth=1, zorder=0)
        for model, color in REG_COLORS.items():
            rows = part[part["model"] == model]
            if rows.empty:
                continue
            ax.scatter(
                rows["paper_median"],
                rows["new_median"],
                s=28,
                color=color,
                alpha=0.85,
                label=model,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Archived median")
        ax.set_ylabel("This run")
        ax.set_title(metric)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Regression medians, archived result against this run")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_bo(frame: pd.DataFrame, path: Path) -> None:
    problems = [name for name in BO_ORDER if name in set(frame["problem"])]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), squeeze=False)
    width = 0.36
    x = np.arange(len(BO_MODELS_ORDER))
    for ax, problem in zip(axes.ravel(), problems):
        part = frame[frame["problem"] == problem]
        paper = []
        new = []
        for model in BO_MODELS_ORDER:
            row = part[part["model"] == model]
            paper.append(np.nan if row.empty else float(row["paper_median"].iloc[0]))
            new.append(np.nan if row.empty else float(row["new_median"].iloc[0]))
        ax.bar(x - width / 2, paper, width, color="#5C6BC0", label="Archived")
        ax.bar(x + width / 2, new, width, color="#26A69A", label="This run")
        ax.axhline(0, color="0.4", linewidth=0.6)
        ax.set_xticks(x, BO_MODELS_ORDER, fontsize=8)
        ax.set_title(problem, fontsize=10)
        finite = [v for v in paper + new if np.isfinite(v)]
        if finite and min(finite) <= 0:
            ax.set_yscale("symlog", linthresh=max(1.0, abs(min(finite)) * 1e-3))
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0, 0].legend(fontsize=8, frameon=True)
    fig.suptitle("Bayesian optimization final best, noise 0.08")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def _has_metrics_json(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for path in folder.glob("*.json"):
        name = path.name.lower()
        if "trainer" in name or name.startswith("failed"):
            continue
        return True
    return False


def _study_gaps() -> list[str]:
    """1D and log-scale cases in the archive that this run does not have yet."""
    new = new_results("regression")
    checks = [
        (
            "GP+ log-scale",
            "logscale/10_runs_logging_full_Gaussian_logscale",
            ("buckling", "zakharov"),
        ),
        (
            "GP+ (PE) log-scale",
            "logscale/10_runs_logging_full_PE_logscale",
            ("buckling", "zakharov"),
        ),
        (
            "GP+ (LOO) log-scale",
            "logscale/10_runs_logging_full_Gaussian_LOO_logscale",
            ("buckling", "zakharov"),
        ),
        (
            "GPyTorch log-scale",
            "logscale/10_runs_gpytorch_corrected_LBFGS_logscale",
            ("buckling", "zakharov"),
        ),
        (
            "1D",
            "onedim/A22_regression_1D",
            ("discontinuity", "triangle_wave", "chirp", "localized_bump", "damped_sine"),
        ),
        (
            "1D tuned TabPFN",
            "onedim/A22_regression_1D_tabpfn_tuned",
            ("discontinuity", "triangle_wave", "chirp", "localized_bump", "damped_sine"),
        ),
    ]
    lines = []
    for label, rel, names in checks:
        missing = [name for name in names if not _has_metrics_json(new / rel / name)]
        if missing:
            lines.append(f"- {label}: no new result for {', '.join(missing)}")
    return lines


def _failure_rows() -> list[dict]:
    import json

    rows = []
    for path in (
        new_results("regression") / "failures.jsonl",
        new_results("bo") / "failures.jsonl",
        new_results("classification") / "failures.jsonl",
    ):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    reg_out = comparison_dir("regression")
    bo_out = comparison_dir("bo")
    comparison_dir("classification").mkdir(parents=True, exist_ok=True)
    reg_out.mkdir(parents=True, exist_ok=True)
    bo_out.mkdir(parents=True, exist_ok=True)
    reg = _regression()
    bo = _bo()
    reg.to_csv(reg_out / "comparison.csv", index=False)
    bo.to_csv(bo_out / "comparison.csv", index=False)
    _md(
        reg.sort_values(["problem", "Dx", "N_over_Dx", "noise", "model", "metric"]),
        reg_out / "comparison.md",
        ["problem", "Dx", "N_over_Dx", "noise", "model", "metric", "paper_median", "new_median", "rel_diff"],
    )
    _md(
        bo.sort_values(["problem", "model"]) if not bo.empty else bo,
        bo_out / "comparison.md",
        ["problem", "model", "goal", "paper_median", "new_median", "rel_diff"],
    )
    if not reg.empty:
        _plot_regression(reg, reg_out / "comparison.png")
    if not bo.empty:
        _plot_bo(bo, bo_out / "comparison.png")
    matched = reg.dropna(subset=["new_median", "rel_diff"])
    far = matched[matched["rel_diff"] > 0.1]
    failures = _failure_rows()
    lines = [
        "# Results",
        "",
        "Each suite has the archived paper run, this run, and a comparison of the two.",
        "",
        "- `regression_results/regression_original_results`, `regression_new_results`, `regression_results_comparison`",
        "- `bo_results/bo_original_results`, `bo_new_results`, `bo_results_comparison`",
        "- `classification_results/classification_original_results`, `classification_new_results`, `classification_results_comparison`",
        "",
        "# Comparison with archived paper results",
        "",
        f"Regression rows with a new run: {len(matched)} of {len(reg)}.",
        f"Regression rows whose median moved by more than 10%: {len(far)}.",
        "",
        "1D and log-scale cases still missing from this run:",
        "",
    ]
    gaps = _study_gaps()
    if gaps:
        lines.extend(gaps)
    else:
        lines.append("None. The 1D examples and the log-scale study both have a new run.")
    lines.extend(
        [
            "",
            "Classification has no archived CSVs, so there is nothing to compare until a paper archive is added.",
            "",
        ]
    )
    if failures:
        lines.append(f"Experiments that failed or timed out: {len(failures)}.")
        lines.append("")
        for row in failures:
            lines.append(f"- **{row.get('status')}** `{row.get('label')}`")
        lines.append("")
        lines.append("Details are in each suite's `*_new_results/failures.md`.")
    else:
        lines.append("No experiment failures were recorded.")
    lines.extend(
        [
            "",
            "Regression comparison: `regression_results/regression_results_comparison/`.",
            "Bayesian optimization comparison: `bo_results/bo_results_comparison/`.",
        ]
    )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
