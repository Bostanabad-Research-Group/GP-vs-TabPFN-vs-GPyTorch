"""Rebuild or rerun every experiment in the paper.

By default this only rebuilds figures and tables from the archived results.
It does not train anything, and it does not write into the ``*_original_results`` folders.

    python run_all_experiments.py

Rerun one regression problem, then plot that new run:

    python run_all_experiments.py --rerun --suite regression --problems wing --models gpplus

Rerun everything, including the 1D examples, the tuned TabPFN 1D overlay, and the log-scale study (this is thousands of model fits):

    python run_all_experiments.py --rerun --suite all --per-problem-plots

``--per-problem-plots`` also writes one figure per problem. Leave it off to
keep only the three combined figures and the summary tables.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES = {
    "regression": ROOT / "experiments_regression" / "run_all.py",
    "bo": ROOT / "experiments_BO" / "run_all.py",
    "classification": ROOT / "experiments_classification" / "run_all.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper experiments: regression, BO, and classification.")
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        choices=["all", "regression", "bo", "classification"],
    )
    parser.add_argument("--rerun", action="store_true", help="Train. New files go to each suite's results/ folder.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--per-problem-plots",
        action="store_true",
        help="Also write one figure per problem. Off by default to save disk.",
    )
    parser.add_argument("--source", choices=("paper", "results"), default="paper")
    parser.add_argument("--problems", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--noise", nargs="*")
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--train-sizes", nargs="*")
    parser.add_argument("--no-trainer-logs", action="store_true")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=12,
        help="Stop one configuration if it runs longer than this, then continue.",
    )
    args = parser.parse_args()

    names = list(SUITES) if "all" in args.suite else args.suite
    for name in names:
        cmd = [sys.executable, str(SUITES[name])]
        if args.rerun:
            cmd.append("--rerun")
        if args.no_plot:
            cmd.append("--no-plot")
        if args.per_problem_plots:
            cmd.append("--per-problem-plots")
        if args.no_trainer_logs:
            cmd.append("--no-trainer-logs")
        cmd.extend(["--source", args.source])
        if args.problems:
            cmd.append("--problems")
            cmd.extend(args.problems)
        if args.models:
            cmd.append("--models")
            cmd.extend(args.models)
        if args.sections:
            cmd.append("--sections")
            cmd.extend(args.sections)
        if args.noise:
            cmd.append("--noise")
            cmd.extend(str(v) for v in args.noise)
        if args.num_runs is not None:
            cmd.extend(["--num-runs", str(args.num_runs)])
        if args.train_sizes:
            cmd.append("--train-sizes")
            cmd.extend(str(v) for v in args.train_sizes)
        if args.timeout_hours is not None:
            cmd.extend(["--timeout-hours", str(args.timeout_hours)])
        print("\n" + " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=str(SUITES[name].parent), check=False)
        if completed.returncode != 0:
            print(
                f"{name} runner exited with code {completed.returncode}. "
                "Continuing with the remaining suites.",
                flush=True,
            )


if __name__ == "__main__":
    main()
