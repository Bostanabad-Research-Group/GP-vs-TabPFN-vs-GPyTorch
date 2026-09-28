"""Run the paper's classification experiments.

New runs go to ``results/classification_results/classification_new_results/<dataset>/``.
The archive in ``classification_original_results/`` uses the same folders as a new run.

    python run_all.py --rerun
    python run_all.py --source paper
    python run_all.py --rerun --problems stellar --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCH = HERE / "benchmarks"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE / "onedim"))
sys.path.insert(0, str(REPO))

from result_paths import new_results, original_results  # noqa: E402

NEW_RESULTS = new_results("classification")
ORIGINAL_RESULTS = original_results("classification")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification experiments from the MLE paper.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--per-problem-plots", action="store_true")
    parser.add_argument("--source", choices=("paper", "results"), default="results")
    parser.add_argument(
        "--problems",
        nargs="*",
        default=["electrical_grid", "truss_6d", "stellar", "steel_plates", "onedim"],
    )
    parser.add_argument("--models", nargs="*", default=None, help="Ignored. The sweep runs every kernel and both TabPFN versions.")
    parser.add_argument("--sections", nargs="*", default=None, help="Ignored.")
    parser.add_argument("--noise", nargs="*", default=None, help="Ignored.")
    parser.add_argument("--num-runs", type=int, default=None, help="Ignored. Seeds are fixed at 0..9.")
    parser.add_argument("--train-sizes", nargs="*", default=None, help="Ignored.")
    parser.add_argument("--no-trainer-logs", action="store_true", help="Ignored.")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=12,
        help="Stop one dataset if it runs longer than this, then continue.",
    )
    parser.add_argument("--job-file", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def execute_job(job: dict) -> None:
    name = job["problem"]
    if name == "onedim":
        import onedim_example

        print("\n=== 1D classification", flush=True)
        onedim_example.main()
        return
    _run_dataset(name)


def _run_job_file(path: Path) -> None:
    import json

    job = json.loads(path.read_text(encoding="utf-8"))
    try:
        execute_job(job)
    except Exception:
        text = traceback.format_exc()
        trace = job.get("trace_path")
        if trace:
            Path(trace).write_text(text, encoding="utf-8")
        print(text, flush=True)
        raise SystemExit(1) from None


def _protect(args, name: str) -> None:
    from experiment_guard import run_isolated

    if name == "onedim":
        save = str(NEW_RESULTS / "onedim")
    else:
        save = str(NEW_RESULTS / name)
    run_isolated(
        Path(__file__).resolve(),
        {
            "label": f"classification {name}",
            "model": "gpplus+tabpfn",
            "problem": name,
            "save_path": save,
            "skip_existing": False,
        },
        timeout_s=float(args.timeout_hours) * 3600.0,
        failures_path=NEW_RESULTS / "failures.jsonl",
    )


def _run_dataset(name: str) -> None:
    if name == "electrical_grid":
        import run_electrical_grid as mod
    elif name == "truss_6d":
        import run_truss_6d as mod
    elif name == "stellar":
        import run_stellar as mod
    elif name == "steel_plates":
        import run_steel_plates as mod
    else:
        return
    print(f"\n=== Classification {name}", flush=True)
    mod.run_sweep(mod.SPEC, mod.SWEEP, loader_cfg=getattr(mod, "DATA_CONFIG", None) or getattr(mod, "DATA_PATHS", None))


def _write_tables(root: Path, per_problem: bool) -> None:
    from plot_gpc_comparison import normalize_summary

    frames = []
    for path, name in _summary_csvs(root):
        frame = normalize_summary(pd.read_csv(path))
        frame.insert(0, "dataset", name)
        frames.append(frame)
    if not frames:
        print(f"No classification summary CSVs under {root}")
        return
    all_rows = pd.concat(frames, ignore_index=True)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(summary / "classification_summary.csv", index=False)
    _markdown(all_rows, summary / "classification_summary.md")
    if per_problem:
        table_dir = summary / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for dataset, part in all_rows.groupby("dataset"):
            _markdown(part, table_dir / f"{dataset}.md")
    print(f"Saved {summary / 'classification_summary.md'}")


def _markdown(frame: pd.DataFrame, path: Path) -> None:
    cols = [
        c
        for c in (
            "dataset",
            "model",
            "kernel",
            "train_size",
            "accuracy_mean",
            "accuracy_std",
            "ece_mean",
            "ece_std",
            "final_train_nll_mean",
            "final_train_nll_std",
        )
        if c in frame.columns
    ]
    show = frame[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in show.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_csvs(root: Path) -> list[tuple[str, str]]:
    """Archive files sit next to the README. New runs use one folder per dataset."""
    found = []
    for name in ("electrical_grid", "truss_6d", "stellar", "steel_plates"):
        nested = root / name / f"{name}_summary.csv"
        flat = sorted(root.glob(f"{name}_summary*.csv"))
        path = nested if nested.is_file() else (flat[0] if flat else None)
        if path is not None:
            found.append((str(path), name))
    return found


def _plot(source: str, per_problem: bool) -> None:
    root = ORIGINAL_RESULTS if source == "paper" else NEW_RESULTS
    summaries = _summary_csvs(root)
    if source == "paper" and not summaries:
        print(
            "No classification summaries in classification_original_results/. "
            "Looking in classification_new_results/ instead."
        )
        root = NEW_RESULTS
        summaries = _summary_csvs(root)
    if not summaries:
        print(f"No classification summary CSVs under {root}")
        return
    os.chdir(BENCH)
    import plot_gpc_comparison as plotter

    plotter.SUMMARY_CSVS = summaries
    plotter.OUTPUT_DIR = str(root / "plots_per_dataset")
    plotter.PAPER_DIR = str(root / "summary")
    plotter.WRITE_PER_DATASET = per_problem
    plotter.main()
    _write_tables(root, per_problem)
    onedim = root / "onedim" / "onedim_example.png"
    if onedim.exists():
        dest = root / "summary" / "classification_1d.png"
        dest.write_bytes(onedim.read_bytes())
        print(f"Saved {dest}")


def main() -> None:
    args = _parse()
    if args.job_file:
        _run_job_file(Path(args.job_file))
        return
    if args.rerun:
        for name in args.problems:
            _protect(args, name)
        from experiment_guard import write_failure_report

        write_failure_report(NEW_RESULTS / "failures.jsonl")
        source = "results"
    else:
        source = args.source
    if not args.no_plot:
        _plot(source, args.per_problem_plots)


if __name__ == "__main__":
    main()
