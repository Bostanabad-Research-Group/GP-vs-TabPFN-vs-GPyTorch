# Regression experiments

These scripts reproduce Section 4.2 (1D examples) and Section 4.3 (benchmarks and log-scale targets) of "On the Brittleness of Maximum Likelihood Estimation for Gaussian Process Hyperparameter Optimization" (arXiv:2608.13793). The earlier uncertainty-quantification paper does not include this log-scale study. See the repository README for both citations.

## Layout

| Path | Role |
|---|---|
| `A1_wing`, `A2_buckling`, `A3_borehole`, `A4_ackley`, `A6_rosenbrock`, `A7_zakharov`, `A8_griewank`, `A9_dixon_price` | One problem each. The `*_gpytorch.py` files are the GPyTorch versions. |
| `A22_regression_1D.py` | 1D GP+ vs TabPFN driver |
| `A22_paper_figure_1d_examples.py` | Five-panel figure from saved 1D curves |
| `A22_regression_1D_reference_equations.py` | Equations for the 1D examples |
| `defaults.py` | GP+ optimizer, scaling, and TabPFN version |
| `defaults_gpytorch.py` | GPyTorch settings. L-BFGS is the paper default. Set `USE_ADAM = True` in that file to switch. |
| `load_experimental_data.py` | Benchmark and 1D data generators |
| `run_all.py` | Runs the paper suite and builds the summary |
| `plot_summary.py` | Figures and tables only |
| `experimental_utils/` | Violin plots, LaTeX tables, 1D plotting helpers |
| `results/regression_results/regression_original_results/` | Archived paper outputs. Do not overwrite. |
| `results/regression_results/regression_new_results/` | Where a new run is written |
| `results/regression_results/regression_results_comparison/` | Archived median against this run |

`regression_original_results/benchmarks/` holds GP+, GP+ (PE), GP+ (LOO), TabPFN v2.0, TabPFN v2.5, and GPyTorch. `regression_original_results/logscale/` holds the log-transformed Buckling and Zakharov refits. `regression_original_results/onedim/` holds the 1D curves, including the tuned TabPFN overlay used in Figure 1.

## Paper settings

Benchmarks use 10 repeats, 5,000 Sobol test points, 16 hyperparameter restarts, and L-BFGS (one outer step, up to 2,000 inner iterations). Inputs are scaled to \([-1, 1]\). Responses are standardized, except in the log-scale study.

Noise is Gaussian with standard deviation \(0.002 c\) or \(0.08 c\), where \(c\) is the response scale used by the data generator. Training size is \(N = 5 D_x\) or \(N = 20 D_x\).

| Problem | \(D_x\) | Notes |
|---|---|---|
| Buckling | 4 | Categorical inputs. Stays on the mixed-variable kernel for every GP+ variant. |
| Borehole | 8 | |
| Wing weight | 10 | |
| Ackley | 20 and 40 | Bounds \([-5, 10]\) |
| Griewank | 20 | Bounds \([-600, 600]\) |
| Zakharov | 20 | Bounds \([-5, 10]\). Also in the log-scale study. |
| Dixon-Price | 40 | Bounds \([-10, 10]\) |
| Rosenbrock | 80 | Bounds \([-5, 10]\). Shown separately in the paper because LOO is far worse. |

GP+ (PE) swaps the Gaussian kernel for the power-exponential kernel. GP+ (LOO) swaps the marginal likelihood for the leave-one-out log pseudo-likelihood. GPyTorch uses L-BFGS, not Adam. TabPFN is not trained; `defaults.PFN_VERSION` selects v2.5 or v2.0.

The 1D figure is noise-free, with 20 training points, seed 42, and 10 repeats. The five panels are discontinuous sine, triangle wave, chirp, localized bump, and damped sine. Tuned TabPFN uses the small-samples checkpoint, 16 estimators, and softmax temperature 0.45. That configuration is in `A22_tune_tabpfn_1d.py`. A rerun with `--sections onedim`, or a full regression rerun, writes both the default curves and the tuned overlay under `regression_new_results/onedim/`. The archived tuned curves are in `regression_original_results/onedim/`.

## Run

From this folder, or from the repository root via `python run_all_experiments.py --suite regression`.

Plot the archive:

```bash
python run_all.py
```

Outputs:

- `results/regression_results/regression_original_results/summary/regression_final.png` and `regression_final.pdf`
- `results/regression_results/regression_original_results/summary/regression_1d_final.png`
- `results/regression_results/regression_original_results/summary/regression_logscale_final.png`
- `results/regression_results/regression_original_results/summary/regression_summary.md` and `regression_results_table.tex`
- `results/regression_results/regression_original_results/summary/regression_timing_table.tex`

The Markdown and CSV tables report median ± standard deviation of RRMSE and NIS, which is Table A1. The LaTeX file is the wide paper table. With `--per-problem-plots`, each problem also gets its own violin PDF and a smaller Markdown table under `summary/tables/`.

Rerun, without touching the archive:

```bash
python run_all.py --rerun --problems wing zakharov --models gpplus pfn25
python run_all.py --rerun --sections onedim
python run_all.py --rerun --sections logscale --problems buckling
```

`--no-trainer-logs` skips the per-iteration trainer dumps. `--per-problem-plots` during a rerun also keeps the violin plots that each problem script writes while it trains. If one configuration raises or runs longer than `--timeout-hours` (12 by default), that configuration is recorded in `results/regression_results/regression_new_results/failures.md` and the suite continues with the next one. A finished result file is not repeated. The file name must match the dimension, so Ackley 20D does not stand in for Ackley 40D.

A full rerun is long, especially GP+ (PE) and Rosenbrock at \(D_x = 80\).
