# Bayesian optimization experiments

These scripts reproduce Section 4.4. The surrogates are GP+ and TabPFN. The paper does not run a separate GPyTorch optimizer here. GPyTorch appears in the regression suite.

## Layout

| Path | Role |
|---|---|
| `B1_wing`, `B2_buckling`, `B3_borehole`, `B4_ackley`, `B6_rosenbrock`, `B7_zakharov`, `B8_griewank`, `B9_dixon_price` | One problem each. Rosenbrock (`B6`) is not in the paper figure. |
| `run_BO.py` | One BO loop for GP+ or TabPFN |
| `defaults.py` | EI, 30 iterations, patience 10, 5,000 TabPFN candidates, 64 GP acquisition starts |
| `load_experimental_data.py` | Same objectives as the regression suite |
| `run_all.py` | Paper suite |
| `plot_summary.py` | Eight-panel figure and the final-value table |
| `plot_BO_IDETC.py` | The same eight-panel figure, used by `plot_summary.py` |
| `results_paper/` | Archived high-noise runs (noise 0.08) for GP+, TabPFN v2.0, and TabPFN v2.5 |
| `results/` | New runs |

## Paper settings

Ten trials. The initial design has \(N_0 = 5 D_x\) scrambled Sobol points. Each trial may add 30 evaluations and stops after 10 iterations with no improvement in the best noisy observation. The acquisition function is expected improvement.

GP+ is retrained from scratch after every observation, using this folder's `defaults.py`: L-BFGS, 16 restarts, and up to 2,000 iterations (`max_eval` is 2,500 here, not the regression suite's 5,000). EI is maximized with L-BFGS from 64 starts. TabPFN is not retrained. EI is evaluated on 5,000 Sobol candidates and the best candidate is chosen (`defaults.BO_GI_PFN = False`).

Buckling and Borehole are maximization problems. The figure plots the negative of the best value so every panel trends downward. The summary table reports the original objective.

Problems, in figure order: Buckling (\(D_x=4\)), Borehole (8), Wing weight (10), Ackley 20, Griewank 20, Zakharov 20, Ackley 40, Dixon-Price 40.

The archive contains the noise 0.08 runs shown in the paper. The runner can also use noise 0.002. The paper reports that the low-noise curves look similar and does not include that figure.

## Run

Plot the archive:

```bash
python run_all.py
```

This writes `results_paper/summary/BO_final.png` and `results_paper/summary/bo_summary.md`.

Rerun GP+ on one problem:

```bash
python run_all.py --rerun --problems wing --models gp --noise 0.08
```

Rerun TabPFN v2.5 on the full set:

```bash
python run_all.py --rerun --models pfn25 --noise 0.08
```

`--per-problem-plots` writes one curve figure per problem under `summary/per_problem/` and one Markdown table per problem under `summary/tables/`.

New files land in `results/GP+`, `results/PFN_V2.5`, or `results/PFN_V2.0`.
