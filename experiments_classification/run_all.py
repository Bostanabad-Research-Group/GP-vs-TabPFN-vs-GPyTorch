"""Run the paper's classification experiments.

New runs go to ``results/<dataset>/``. There is no archived classification
CSV in this repository, so plotting needs a run first.

    python run_all.py --rerun
    python run_all.py --rerun --problems stellar --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmarks"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE / "onedim"))


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
    return parser.parse_args()


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
    print(f"\n=== Classification {name}")
    mod.run_sweep(mod.SPEC, mod.SWEEP, loader_cfg=getattr(mod, "DATA_CONFIG", None) or getattr(mod, "DATA_PATHS", None))


def _write_tables(root: Path, per_problem: bool) -> None:
    frames = []
    for name in ("electrical_grid", "truss_6d", "stellar", "steel_plates"):
        path = root / name / f"{name}_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
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


def _plot(source: str, per_problem: bool) -> None:
    root = HERE / ("results_paper" if source == "paper" else "results")
    archived = list((HERE / "results_paper").glob("*/*_summary.csv")) if (HERE / "results_paper").is_dir() else []
    if source == "paper" and not archived:
        print(
            "Classification CSVs were not part of the archived paper results. "
            "Looking in results/ instead. Run with --rerun to create them."
        )
        root = HERE / "results"
    os.chdir(BENCH)
    import plot_gpc_comparison as plotter

    plotter.SUMMARY_CSVS = [
        (str(root / "electrical_grid" / "electrical_grid_summary.csv"), "electrical_grid"),
        (str(root / "truss_6d" / "truss_6d_summary.csv"), "truss_6d"),
        (str(root / "stellar" / "stellar_summary.csv"), "stellar"),
        (str(root / "steel_plates" / "steel_plates_summary.csv"), "steel_plates"),
    ]
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
    if args.rerun:
        for name in args.problems:
            if name == "onedim":
                import onedim_example

                print("\n=== 1D classification")
                onedim_example.main()
            else:
                _run_dataset(name)
        source = "results"
    else:
        source = args.source
    if not args.no_plot:
        _plot(source, args.per_problem_plots)


if __name__ == "__main__":
    main()
