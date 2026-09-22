"""Build the regression summary figure and tables.

Reads either the archived paper results or a new run under ``results/``.
The combined figure is one PDF (RRMSE, NIS, and NCRPS) plus a PNG of the
RRMSE panel. Per-problem violin PDFs are optional.

    python plot_summary.py
    python plot_summary.py --source results --per-problem-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "experimental_utils"))

import IDETC_create_tables as tables  # noqa: E402
import IDETC_plot_gpplus_comparison as cmp_plots  # noqa: E402
import IDETC_plot_gpplus_logscale_comparison as log_plots  # noqa: E402

PAPER_PANELS = [
    ("buckling", "4", "Buckling"),
    ("borehole", "8", "Borehole"),
    ("wing", "10", "Wing Weight"),
    ("ackley", "20", "Ackley 20D"),
    ("griewank", "20", "Griewank 20D"),
    ("zakharov", "20", "Zakharov 20D"),
    ("ackley", "40", "Ackley 40D"),
    ("dixon_price", "40", "Dixon-Price 40D"),
    ("rosenbrock", "80", "Rosenbrock 80D"),
]
N_LEVELS = [("5D", r"$N=5D_x$"), ("20D", r"$N=20D_x$")]
MODEL_ORDER = [
    "gpplus",
    "GP+ (PE)",
    "GP+ (LOO)",
    "tabpfn_v2.5",
    "tabpfn_v2",
    "gpytorch",
]


def result_roots(source: str) -> tuple[Path, Path, Path]:
    base = HERE / ("results_paper" if source == "paper" else "results")
    if source == "paper":
        return base / "benchmarks", base / "logscale", base / "onedim"
    return base / "benchmarks", base / "logscale", base / "onedim"


def _pick(root: Path, *names: str) -> Path | None:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def benchmark_dirs(root: Path) -> dict[str, Path]:
    found = {
        "gpplus": _pick(root, "10_runs_logging_full_Gaussian", "10_runs_logging_full_Gaussian_orig"),
        "GP+ (PE)": _pick(root, "10_runs_logging_full_PE", "10_runs_logging_full_PE_orig"),
        "GP+ (LOO)": _pick(root, "10_runs_logging_full_Gaussian_LOO", "10_runs_logging_full_Gaussian_LOO_orig"),
        "tabpfn_v2.5": _pick(root, "10_runs_PFN_V2.5"),
        "tabpfn_v2": _pick(root, "10_runs_PFN_V2.0"),
        "gpytorch": _pick(root, "10_runs_gpytorch", "10_runs_gpytorch_corrected_LBFGS"),
    }
    return {key: path for key, path in found.items() if path is not None}


def _collect(dirs: dict[str, Path]) -> pd.DataFrame:
    empty = Path("__missing__")
    return cmp_plots.collect_idetc_df(
        gp_dir=dirs.get("gpplus", empty),
        pe_dir=dirs.get("GP+ (PE)", empty),
        loo_dir=dirs.get("GP+ (LOO)"),
        pfn25_dir=dirs.get("tabpfn_v2.5", empty),
        pfn20_dir=dirs.get("tabpfn_v2", empty),
        gpytorch_dir=dirs.get("gpytorch", empty),
        include_loo="GP+ (LOO)" in dirs,
        noise_levels=["0.002", "0.08"],
    )


def _panel_frame(df: pd.DataFrame, problem: str, xdim: str, n_level: str) -> pd.DataFrame:
    part = df[
        (df["problem"].astype(str).str.lower() == problem)
        & (df["xdim"].astype(str) == xdim)
        & (df["dim"].astype(str) == n_level)
    ]
    return part


def _combined_pdf(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    metrics = [m for m in ("RRMSE", "NIS", "NCRPS") if m in df.columns]
    if df.empty or not metrics:
        print("No regression benchmark rows to plot.")
        return
    models = [m for m in MODEL_ORDER if m in set(df["model_family"].unique())]
    noise_levels = sorted(
        [n for n in df["noise_label"].unique() if n not in ("unknown", "all")],
        key=lambda v: float(v),
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(PAPER_PANELS)
    with PdfPages(out_pdf) as pdf:
        for metric in metrics:
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.15 * n_rows), squeeze=False)
            for row, (problem, xdim, title) in enumerate(PAPER_PANELS):
                for col, (n_level, n_label) in enumerate(N_LEVELS):
                    ax = axes[row, col]
                    part = _panel_frame(df, problem, xdim, n_level)
                    if part.empty:
                        ax.set_axis_off()
                        ax.set_title(f"{title}, {n_label}: no data", fontsize=8)
                        continue
                    cmp_plots._plot_metric_axis(
                        ax, part, metric, models, noise_levels, remove_outliers=False
                    )
                    for lab in ax.get_xticklabels():
                        lab.set_fontsize(7)
                        lab.set_rotation(35)
                        lab.set_ha("right")
                    ax.tick_params(axis="y", labelsize=7)
                    ax.set_title(f"{title}, {n_label}", fontsize=9)
            fig.suptitle(f"Regression {metric}", fontsize=14)
            handles = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc=cmp_plots.NOISE_COLOR_MAP.get(noise, "gray"),
                    ec="black",
                    label=f"Noise {noise}",
                )
                for noise in noise_levels
            ]
            fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            if metric == "RRMSE":
                fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"Added {metric} page to {out_pdf.name}")
    print(f"Saved {out_png}")


def _write_tables(dirs: dict[str, Path], summary: Path, per_problem: bool) -> None:
    stats = tables.collect_stats(dirs, center="median")
    if not stats:
        print("No regression rows for the summary table.")
        return
    summary.mkdir(parents=True, exist_ok=True)
    tex = tables.make_regression_results_table(stats, center="median")
    (summary / "regression_results_table.tex").write_text(tex, encoding="utf-8")
    timing = tables.make_timing_table(stats)
    (summary / "regression_timing_table.tex").write_text(timing, encoding="utf-8")

    rows = []
    for key, metrics in stats.items():
        problem, xdim, n_mult, noise, model = key
        r_c, r_s = metrics["RRMSE"]
        n_c, n_s = metrics["NIS"]
        rows.append(
            {
                "problem": problem,
                "Dx": xdim,
                "N_over_Dx": n_mult,
                "noise": noise,
                "model": model,
                "RRMSE_median": r_c,
                "RRMSE_std": r_s,
                "NIS_median": n_c,
                "NIS_std": n_s,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary / "regression_summary.csv", index=False)
    _write_markdown(frame, summary / "regression_summary.md")
    print(f"Saved {summary / 'regression_summary.md'}")

    if not per_problem:
        return
    table_dir = summary / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for problem, part in frame.groupby("problem"):
        _write_markdown(part, table_dir / f"{problem}.md")


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Problem | Dx | N/Dx | Noise | Model | RRMSE | NIS |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    ordered = frame.sort_values(["problem", "Dx", "N_over_Dx", "noise", "model"])
    for row in ordered.itertuples(index=False):
        lines.append(
            "| {problem} | {Dx} | {N_over_Dx} | {noise} | {model} | {rrmse} | {nis} |".format(
                problem=row.problem,
                Dx=row.Dx,
                N_over_Dx=row.N_over_Dx,
                noise=row.noise,
                model=row.model,
                rrmse=_pm(row.RRMSE_median, row.RRMSE_std),
                nis=_pm(row.NIS_median, row.NIS_std),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pm(center, std) -> str:
    if center is None or (isinstance(center, float) and not np.isfinite(center)):
        return ""
    if std is None or (isinstance(std, float) and not np.isfinite(std)):
        return f"{center:.3g}"
    return f"{center:.3g} ± {std:.3g}"


def _logscale_figure(bench: Path, log_root: Path, summary: Path, per_problem: bool) -> None:
    dirs = benchmark_dirs(bench)
    gp_log = _pick(log_root, "10_runs_logging_full_Gaussian_logscale")
    pe_log = _pick(log_root, "10_runs_logging_full_PE_logscale")
    loo_log = _pick(log_root, "10_runs_logging_full_Gaussian_LOO_logscale")
    gy_log = _pick(log_root, "10_runs_gpytorch_corrected_LBFGS_logscale")
    if gp_log is None or "gpplus" not in dirs:
        print("Log-scale results not found; skipping that figure.")
        return
    missing = Path("__missing_model__")
    df = log_plots.collect_logscale_df(
        gp_dir=dirs["gpplus"],
        pe_dir=dirs.get("GP+ (PE)", missing),
        loo_dir=dirs.get("GP+ (LOO)"),
        gp_log_dir=gp_log,
        pe_log_dir=pe_log or missing,
        loo_log_dir=loo_log,
        pfn25_dir=dirs.get("tabpfn_v2.5", missing),
        pfn20_dir=dirs.get("tabpfn_v2", missing),
        gpytorch_dir=dirs.get("gpytorch", missing),
        gpytorch_log_dir=gy_log or missing,
        include_loo=loo_log is not None and "GP+ (LOO)" in dirs,
        problems={"buckling", "zakharov"},
    )
    if df.empty:
        print("Log-scale collector returned no rows.")
        return
    if per_problem:
        out = summary / "logscale_per_problem"
        order = [m for m in df["model_family"].unique()]
        log_plots.make_separate_metric_plots(df, out, order, remove_outliers=False)
    # One small combined image: buckling and zakharov, RRMSE only.
    problems = [("buckling", "4", "Buckling"), ("zakharov", "20", "Zakharov")]
    fig, axes = plt.subplots(len(problems), 2, figsize=(14, 6), squeeze=False)
    preferred = [
        "gpplus",
        "GP+ (log)",
        "GP+ (PE)",
        "GP+ (PE log)",
        "GP+ (LOO)",
        "GP+ (LOO log)",
        "tabpfn_v2.5",
        "tabpfn_v2",
        "gpytorch",
        "GPyTorch (log)",
    ]
    present = set(df["model_family"].unique())
    models = [m for m in preferred if m in present]
    models.extend(m for m in present if m not in models)
    noise_levels = sorted(
        [n for n in df["noise_label"].unique() if n not in ("unknown", "all")],
        key=lambda v: float(v),
    )
    for row, (problem, xdim, title) in enumerate(problems):
        for col, (n_level, n_label) in enumerate(N_LEVELS):
            ax = axes[row, col]
            part = _panel_frame(df, problem, xdim, n_level)
            if part.empty:
                ax.set_axis_off()
                continue
            cmp_plots._plot_metric_axis(ax, part, "RRMSE", models, noise_levels, False)
            for lab in ax.get_xticklabels():
                lab.set_fontsize(6)
                lab.set_rotation(35)
                lab.set_ha("right")
            ax.set_title(f"{title}, {n_label}", fontsize=9)
    fig.suptitle("Log-scale target preprocessing (RRMSE)")
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=cmp_plots.NOISE_COLOR_MAP.get(noise, "gray"),
            ec="black",
            label=f"Noise {noise}",
        )
        for noise in noise_levels
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    dest = summary / "regression_logscale_final.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved {dest}")


def write_summary(source: str = "paper", per_problem: bool = False) -> Path:
    bench, log_root, onedim = result_roots(source)
    summary = (HERE / ("results_paper" if source == "paper" else "results")) / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    dirs = benchmark_dirs(bench)
    if not dirs:
        print(f"No benchmark result folders under {bench}")
    else:
        df = _collect(dirs)
        _combined_pdf(df, summary / "regression_final.pdf", summary / "regression_final.png")
        if per_problem and not df.empty:
            order = [m for m in MODEL_ORDER if m in set(df["model_family"].unique())]
            cmp_plots.make_separate_metric_plots(
                df, summary / "per_problem", order, remove_outliers=False
            )
        _write_tables(dirs, summary, per_problem)
    _logscale_figure(bench, log_root, summary, per_problem)
    _onedim_figure(onedim, summary)
    return summary


def _onedim_figure(onedim: Path, summary: Path) -> None:
    root = onedim / "A22_regression_1D"
    if not root.is_dir():
        print(f"No 1D results at {root}")
        return
    tuned = onedim / "A22_regression_1D_tabpfn_tuned"
    out = summary / "onedim"
    cmd = [
        sys.executable,
        str(HERE / "A22_paper_figure_1d_examples.py"),
        "--results-root",
        str(root),
        "--out-dir",
        str(out),
        "--math-labels",
    ]
    if tuned.is_dir():
        cmd.extend(["--tuned-root", str(tuned)])
    else:
        cmd.append("--no-tuned")
    import subprocess

    subprocess.run(cmd, cwd=str(HERE), check=False)
    previews = sorted(out.glob("*preview*.png"))
    if previews:
        dest = summary / "regression_1d_final.png"
        dest.write_bytes(previews[0].read_bytes())
        print(f"Saved {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression summary figure and tables.")
    parser.add_argument("--source", choices=("paper", "results"), default="paper")
    parser.add_argument("--per-problem-plots", action="store_true")
    args = parser.parse_args()
    write_summary(args.source, args.per_problem_plots)


if __name__ == "__main__":
    main()
