"""Run the paper's Bayesian optimization experiments.

New runs go to ``results/bo_results/bo_new_results/<model>/``.
Archived runs stay in ``bo_original_results/``.

    python run_all.py
    python run_all.py --rerun --models gp --problems wing --noise 0.08
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from result_paths import new_results  # noqa: E402

RESULTS = new_results("bo")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BO experiments from the MLE paper.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--per-problem-plots", action="store_true")
    parser.add_argument("--source", choices=("paper", "results"), default="paper")
    parser.add_argument("--problems", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["gp", "pfn25", "pfn20"])
    parser.add_argument("--noise", nargs="*", type=float, default=[0.002, 0.08])
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--sections", nargs="*", default=None, help="Ignored. Accepted so the root runner can forward it.")
    parser.add_argument("--train-sizes", nargs="*", type=int, default=None, help="Ignored.")
    parser.add_argument("--no-trainer-logs", action="store_true", help="Ignored.")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=12,
        help="Stop one configuration if it runs longer than this, then continue.",
    )
    parser.add_argument("--job-file", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _cases():
    from B1_wing_SF_GPvsPFN import wing_SF_GPvsPFN_BO
    from B2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN_BO
    from B3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN_BO
    from B4_ackley_GPvsPFN import ackley_GPvsPFN_BO
    from B7_zakharov_GPvsPFN import zakharov_GPvsPFN_BO
    from B8_griewank_GPvsPFN import griewank_GPvsPFN_BO
    from B9_dixon_price_GPvsPFN import dixon_price_GPvsPFN_BO

    return [
        ("wing", None, "B1_wing", wing_SF_GPvsPFN_BO),
        ("buckling", None, "B2_buckling", buckling_SF_GPvsPFN_BO),
        ("borehole", None, "B3_borehole", borehole_SF_GPvsPFN_BO),
        ("ackley", 20, "B4_ackley", ackley_GPvsPFN_BO),
        ("griewank", 20, "B8_griewank", griewank_GPvsPFN_BO),
        ("zakharov", 20, "B7_zakharov", zakharov_GPvsPFN_BO),
        ("ackley", 40, "B4_ackley", ackley_GPvsPFN_BO),
        ("dixon_price", 40, "B9_dixon_price", dixon_price_GPvsPFN_BO),
    ]


def _selected(problems):
    cases = _cases()
    if not problems:
        return cases
    wanted = {p.lower() for p in problems}
    return [case for case in cases if case[0] in wanted]


def execute_job(job: dict) -> None:
    import run_BO

    if job.get("version"):
        run_BO.PFN_VERSION = job["version"]
    fn = dict((name, func) for name, _dims, _sub, func in _cases())[job["problem"]]
    kwargs = dict(
        num_runs=int(job["num_runs"]),
        num_inits=int(job["num_inits"]),
        start_size=5,
        noise_train=float(job["noise"]),
        noise_test=float(job["noise"]),
        max_iter=30,
        patience_no_improve=10,
        save_path=job["save_path"],
        run_models=job["run_models"],
    )
    if job.get("dims") is not None:
        kwargs["dimensions"] = int(job["dims"])
    fn(**kwargs)


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


def _protect(args, job: dict) -> None:
    from experiment_guard import run_isolated

    run_isolated(
        Path(__file__).resolve(),
        job,
        timeout_s=float(args.timeout_hours) * 3600.0,
        failures_path=RESULTS / "failures.jsonl",
    )


def rerun(args) -> None:
    jobs = []
    if "gp" in args.models or "gpplus" in args.models:
        jobs.append(("GP+", "gp", None))
    if "pfn25" in args.models or "pfn" in args.models:
        jobs.append(("PFN_V2.5", "pfn", "v2.5"))
    if "pfn20" in args.models:
        jobs.append(("PFN_V2.0", "pfn", "v2.0"))

    for folder, run_models, version in jobs:
        num_inits = 0 if run_models == "pfn" else 16
        for problem, dims, sub, _fn in _selected(args.problems):
            for noise in args.noise:
                save = RESULTS / folder / sub
                _protect(
                    args,
                    {
                        "label": f"BO {folder} {problem} dims={dims} noise={noise}",
                        "model": folder,
                        "problem": problem,
                        "dims": dims,
                        "train_size": 5,
                        "noise": noise,
                        "save_path": str(save),
                        "num_runs": args.num_runs,
                        "num_inits": num_inits,
                        "run_models": run_models,
                        "version": version,
                    },
                )


def main() -> None:
    args = _parse()
    if args.job_file:
        _run_job_file(Path(args.job_file))
        return
    if args.rerun:
        rerun(args)
        from experiment_guard import write_failure_report

        write_failure_report(RESULTS / "failures.jsonl")
        source = "results"
    else:
        source = args.source
    if not args.no_plot:
        from plot_summary import write_summary

        write_summary(source, args.per_problem_plots)


if __name__ == "__main__":
    main()
