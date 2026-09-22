"""Run the paper's regression experiments and write summary figures.

New runs are saved under ``results/``. The archived paper runs stay in
``results_paper/`` and are not overwritten.

Plot the archived results (no training):

    python run_all.py

Rerun one problem with GP+ only:

    python run_all.py --rerun --problems wing --models gpplus --no-plot

Rerun the full paper suite (long) and rebuild figures, including per-problem PDFs:

    python run_all.py --rerun --per-problem-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS = HERE / "results"

# (problem key, input dimension, folder name)
BENCHMARKS = [
    ("wing", 10, "wing"),
    ("buckling", 4, "buckling"),
    ("borehole", 8, "borehole"),
    ("ackley", 20, "ackley"),
    ("griewank", 20, "griewank"),
    ("zakharov", 20, "zakharov"),
    ("ackley", 40, "ackley"),
    ("dixon_price", 40, "dixon_price"),
    ("rosenbrock", 80, "rosenbrock"),
]
LOGSCALE_PROBLEMS = {"buckling", "zakharov"}
KERNEL_ARG_PROBLEMS = {"zakharov", "dixon_price", "rosenbrock"}
DIMENSION_ARG_PROBLEMS = {"ackley", "griewank", "zakharov", "dixon_price", "rosenbrock"}
PAPER_1D = ("discontinuity", "triangle_wave", "chirp", "localized_bump", "damped_sine")

GP_MODELS = {
    "gpplus": ("10_runs_logging_full_Gaussian", "nll", "gaussian"),
    "pe": ("10_runs_logging_full_PE", "nll", "pe"),
    "loo": ("10_runs_logging_full_Gaussian_LOO", "loo", "gaussian"),
}
GP_LOG_DIRS = {
    "gpplus": "10_runs_logging_full_Gaussian_logscale",
    "pe": "10_runs_logging_full_PE_logscale",
    "loo": "10_runs_logging_full_Gaussian_LOO_logscale",
}
PFN_MODELS = {
    "pfn25": ("10_runs_PFN_V2.5", "v2.5"),
    "pfn20": ("10_runs_PFN_V2.0", "v2.0"),
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression experiments from the MLE paper.")
    parser.add_argument("--rerun", action="store_true", help="Train models. Writes only under results/.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--per-problem-plots", action="store_true")
    parser.add_argument("--no-trainer-logs", action="store_true")
    parser.add_argument("--source", choices=("paper", "results"), default="paper")
    parser.add_argument("--problems", nargs="*", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gpplus", "pe", "loo", "gpytorch", "pfn25", "pfn20"],
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["benchmarks", "logscale", "onedim"],
        choices=["benchmarks", "logscale", "onedim"],
    )
    parser.add_argument("--noise", nargs="*", type=float, default=[0.002, 0.08])
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--train-sizes", nargs="*", type=int, default=[5, 20])
    return parser.parse_args()


def _silence_inline_plots() -> None:
    import gpplus.utils.metrics_functions as metrics

    metrics.plot_metrics = lambda *args, **kwargs: None
    import experimental_utils.plot_tabpfn1d_comparison as oned

    oned.save_1d_all_runs_gp_tabpfn_plot = lambda *args, **kwargs: None
    oned.save_1d_train_gp_tabpfn_plot = lambda *args, **kwargs: None


def _gp_functions():
    from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
    from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
    from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
    from A4_ackley_GPvsPFN import ackley_GPvsPFN
    from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
    from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
    from A8_griewank_GPvsPFN import griewank_GPvsPFN
    from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN

    return {
        "wing": wing_SF_GPvsPFN,
        "buckling": buckling_SF_GPvsPFN,
        "borehole": borehole_SF_GPvsPFN,
        "ackley": ackley_GPvsPFN,
        "rosenbrock": rosenbrock_GPvsPFN,
        "zakharov": zakharov_GPvsPFN,
        "griewank": griewank_GPvsPFN,
        "dixon_price": dixon_price_GPvsPFN,
    }


def _gpytorch_functions():
    from A1_wing_SF_GPvsPFN_gpytorch import wing_SF_GPvsPFN
    from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN
    from A3_borehole_SF_GPvsPFN_gpytorch import borehole_SF_GPvsPFN
    from A4_Ackley_GPvsPFN_gpytorch import ackley_GPvsPFN
    from A6_rosenbrock_GPvsPFN_gpytorch import rosenbrock_GPvsPFN
    from A7_zakharov_GPvsPFN_gpytorch import zakharov_GPvsPFN
    from A8_griewank_GPvsPFN_gpytorch import griewank_GPvsPFN
    from A9_dixon_price_GPvsPFN_gpytorch import dixon_price_GPvsPFN

    return {
        "wing": wing_SF_GPvsPFN,
        "buckling": buckling_SF_GPvsPFN,
        "borehole": borehole_SF_GPvsPFN,
        "ackley": ackley_GPvsPFN,
        "rosenbrock": rosenbrock_GPvsPFN,
        "zakharov": zakharov_GPvsPFN,
        "griewank": griewank_GPvsPFN,
        "dixon_price": dixon_price_GPvsPFN,
    }


def _set_kernel(kind: str, dims: int, problem: str) -> None:
    import defaults
    import gpplus

    if kind == "pe" and problem != "buckling":
        defaults.SF_kernel = gpplus.kernels.LogScaleKernel(
            gpplus.kernels.PowerExponentialKernel(ard_num_dims=dims)
        )
    else:
        defaults.SF_kernel = None


def _selected(cases, problems):
    if not problems:
        return cases
    wanted = {p.lower() for p in problems}
    return [case for case in cases if case[0] in wanted]


def _call_gp(fn, problem, dims, train_size, noise, save_path, *, num_runs, loss_type, kernel_kind, logscale):
    kwargs = dict(
        num_runs=num_runs,
        num_test=5000,
        train_size=train_size,
        num_inits=16,
        save_path=str(save_path),
        noise_train=noise,
        noise_test=noise,
        run_models="gp",
        loss_type=loss_type,
    )
    if problem in DIMENSION_ARG_PROBLEMS:
        kwargs["dimensions"] = dims
    if problem in KERNEL_ARG_PROBLEMS:
        kwargs["kernel_type"] = "PowerExponential" if kernel_kind == "pe" else "Gaussian"
    if logscale and problem in LOGSCALE_PROBLEMS:
        kwargs["standardize_y_log_scale"] = True
    fn(**kwargs)


def _call_gpytorch(fn, problem, dims, train_size, noise, save_path, *, num_runs, logscale):
    kwargs = dict(
        num_runs=num_runs,
        num_test=5000,
        train_size=train_size,
        num_inits=16,
        save_path=str(save_path),
        noise_train=noise,
        noise_test=noise,
    )
    if problem in DIMENSION_ARG_PROBLEMS:
        kwargs["dimensions"] = dims
    if logscale and problem in LOGSCALE_PROBLEMS:
        kwargs["standardize_y_log_scale"] = True
    fn(**kwargs)


def _run_gp_family(args, *, logscale: bool) -> None:
    fns = _gp_functions()
    cases = _selected(BENCHMARKS, args.problems)
    if logscale:
        cases = [case for case in cases if case[0] in LOGSCALE_PROBLEMS]
    for model in args.models:
        if model not in GP_MODELS:
            continue
        folder, loss_type, kernel_kind = GP_MODELS[model]
        if logscale:
            folder = GP_LOG_DIRS[model]
        root = RESULTS / ("logscale" if logscale else "benchmarks") / folder
        for problem, dims, sub in cases:
            _set_kernel(kernel_kind, dims, problem)
            fn = fns[problem]
            for train_size in args.train_sizes:
                for noise in args.noise:
                    save = root / sub
                    print(f"\n=== GP {model} {problem} Dx={dims} N={train_size}Dx noise={noise} log={logscale}")
                    _call_gp(
                        fn,
                        problem,
                        dims,
                        train_size,
                        noise,
                        save,
                        num_runs=args.num_runs,
                        loss_type=loss_type,
                        kernel_kind=kernel_kind,
                        logscale=logscale,
                    )


def _run_gpytorch(args, *, logscale: bool) -> None:
    if "gpytorch" not in args.models:
        return
    fns = _gpytorch_functions()
    cases = _selected(BENCHMARKS, args.problems)
    if logscale:
        cases = [case for case in cases if case[0] in LOGSCALE_PROBLEMS]
    folder = (
        "10_runs_gpytorch_corrected_LBFGS_logscale" if logscale else "10_runs_gpytorch"
    )
    root = RESULTS / ("logscale" if logscale else "benchmarks") / folder
    for problem, dims, sub in cases:
        fn = fns[problem]
        for train_size in args.train_sizes:
            for noise in args.noise:
                print(f"\n=== GPyTorch {problem} Dx={dims} N={train_size}Dx noise={noise} log={logscale}")
                _call_gpytorch(
                    fn,
                    problem,
                    dims,
                    train_size,
                    noise,
                    root / sub,
                    num_runs=args.num_runs,
                    logscale=logscale,
                )


def _run_pfn(args) -> None:
    import defaults

    fns = _gp_functions()
    cases = _selected(BENCHMARKS, args.problems)
    for model in args.models:
        if model not in PFN_MODELS:
            continue
        folder, version = PFN_MODELS[model]
        defaults.PFN_VERSION = version
        defaults.SF_kernel = None
        root = RESULTS / "benchmarks" / folder
        for problem, dims, sub in cases:
            fn = fns[problem]
            for train_size in args.train_sizes:
                for noise in args.noise:
                    print(f"\n=== TabPFN {version} {problem} Dx={dims} N={train_size}Dx noise={noise}")
                    kwargs = dict(
                        num_runs=args.num_runs,
                        num_test=5000,
                        train_size=train_size,
                        num_inits=0,
                        save_path=str(root / sub),
                        noise_train=noise,
                        noise_test=noise,
                        run_models="pfn",
                    )
                    if problem in DIMENSION_ARG_PROBLEMS:
                        kwargs["dimensions"] = dims
                    fn(**kwargs)


def _run_onedim(args) -> None:
    from A22_regression_1D import REGRESSION_1D_FUNCTIONS, regression_1D_GPvsPFN

    names = PAPER_1D
    if args.problems:
        wanted = [p for p in args.problems if p in REGRESSION_1D_FUNCTIONS]
        names = tuple(wanted) if wanted else ()
    out = RESULTS / "onedim" / "A22_regression_1D"
    for name in names:
        print(f"\n=== 1D {name}")
        regression_1D_GPvsPFN(
            function_name=name,
            num_runs=args.num_runs,
            train_size=20,
            dimensions=1,
            noise_train=0.0,
            noise_test=0.0,
            save_path=str(out / name),
        )


def rerun(args) -> None:
    import defaults

    if args.no_trainer_logs:
        defaults.TRAINER_INFO = False
    if not args.per_problem_plots:
        _silence_inline_plots()
    if "benchmarks" in args.sections:
        _run_gp_family(args, logscale=False)
        _run_gpytorch(args, logscale=False)
        _run_pfn(args)
    if "logscale" in args.sections:
        _run_gp_family(args, logscale=True)
        _run_gpytorch(args, logscale=True)
    if "onedim" in args.sections:
        _run_onedim(args)


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
