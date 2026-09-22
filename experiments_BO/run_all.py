"""Run the paper's Bayesian optimization experiments.

New runs go to ``results/<model>/``. Archived runs stay in ``results_paper/``.

    python run_all.py
    python run_all.py --rerun --models gp --problems wing --noise 0.08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"


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


def rerun(args) -> None:
    import run_BO

    jobs = []
    if "gp" in args.models or "gpplus" in args.models:
        jobs.append(("GP+", "gp", None))
    if "pfn25" in args.models or "pfn" in args.models:
        jobs.append(("PFN_V2.5", "pfn", "v2.5"))
    if "pfn20" in args.models:
        jobs.append(("PFN_V2.0", "pfn", "v2.0"))

    for folder, run_models, version in jobs:
        if version is not None:
            run_BO.PFN_VERSION = version
        num_inits = 0 if run_models == "pfn" else 16
        for problem, dims, sub, fn in _selected(args.problems):
            for noise in args.noise:
                save = RESULTS / folder / sub
                print(f"\n=== BO {folder} {problem} dims={dims} noise={noise}")
                kwargs = dict(
                    num_runs=args.num_runs,
                    num_inits=num_inits,
                    start_size=5,
                    noise_train=noise,
                    noise_test=noise,
                    max_iter=30,
                    patience_no_improve=10,
                    save_path=str(save),
                    run_models=run_models,
                )
                if dims is not None:
                    kwargs["dimensions"] = dims
                fn(**kwargs)


def main() -> None:
    args = _parse()
    if args.rerun:
        rerun(args)
        source = "results"
    else:
        source = args.source
    if not args.no_plot:
        from plot_summary import write_summary

        write_summary(source, args.per_problem_plots)


if __name__ == "__main__":
    main()
