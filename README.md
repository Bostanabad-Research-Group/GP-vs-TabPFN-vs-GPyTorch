# GP+ vs TabPFN vs GPyTorch

This repository reproduces the experiments in [On the Brittleness of Maximum Likelihood Estimation for Gaussian Process Hyperparameter Optimization](https://arxiv.org/abs/2608.13793).

The study trains Gaussian processes with maximum likelihood and compares them with TabPFN v2.0 and v2.5. The GP baselines are:

- **GP+**, this repository's library (`gpplus/`). It uses a log-scale kernel parameterization, a soft clamp on hyperparameters, and 16 random restarts.
- **GPyTorch**, used with its own training loop on the regression benchmarks. Regression uses L-BFGS. Classification uses Adam.

The three experiment suites match the paper sections:

| Folder | Paper section | What is compared |
|---|---|---|
| `experiments_regression/` | 4.2 and 4.3 | 1D examples, analytic benchmarks, and a log-scale target study |
| `experiments_BO/` | 4.4 | Expected-improvement Bayesian optimization |
| `experiments_classification/` | 4.5 and the 1D Dirichlet example | Four classification datasets |

Published outputs are in each suite's `results_paper/` folder. A new run writes to the adjacent `results/` folder, so it does not overwrite the paper results.

## Setup

Use Python 3.11.

```bash
cd GP-vs-TabPFN-vs-GPyTorch
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs this checkout of GP+ with `pip install -e .`. GPyTorch and TabPFN are pinned there as well.

The experiment scripts use `import gpplus`. They do not load the `gpplus/` folder just because it is next to them. Python uses the copy registered in the active environment. The `-e .` install makes that copy a link to this repository, so edits under `gpplus/` are what the scripts run.

Install into a new virtual environment. Do not install another GP+ checkout, such as `GPvsPFN` or `GP-Plus`, into the same environment. The last install of the project named `gpplus` is the one that runs.

After installing, confirm the import points at this folder:

```bash
python -c "import gpplus; print(gpplus.__file__)"
```

The path should end in `GP-vs-TabPFN-vs-GPyTorch\gpplus\__init__.py`. If it points somewhere else, the experiments will train with that other library.

TabPFN v2.5 is a gated Hugging Face model. The first run asks you to create a token at https://huggingface.co/settings/tokens and to enable read access to public gated repositories. Accept the model license on the TabPFN v2.5 model card, then paste the token when prompted.

GPs run on CPU. TabPFN is meant to run on a GPU. If CUDA is not available, TabPFN falls back to CPU and is much slower. The numbers will still be comparable.

## Rebuild the paper figures

From the repository root, this does not train anything. It reads `results_paper/` and writes summary figures and tables next to those results.

```bash
python run_all_experiments.py
```

You get these combined figures from the archive:

- `experiments_regression/results_paper/summary/regression_final.png` (RRMSE for every regression benchmark; the PDF also has NIS and NCRPS)
- `experiments_regression/results_paper/summary/regression_1d_final.png`
- `experiments_regression/results_paper/summary/regression_logscale_final.png`
- `experiments_BO/results_paper/summary/BO_final.png`

Classification has no archived CSVs in this repository, so that figure appears only after a rerun (see below).

Each summary folder also contains a table in Markdown, CSV, and, for regression, LaTeX. The regression table is the same layout as Table A1 in the paper: median ± standard deviation of RRMSE and NIS.

Add per-problem figures when you want them. They are off by default because they use a lot of disk.

```bash
python run_all_experiments.py --per-problem-plots
```

Those files go in `results_paper/summary/per_problem/` and `results_paper/summary/tables/` when you are plotting the archive. After a rerun they go in `results/summary/` instead. Classification writes its per-dataset figures under `experiments_classification/results/plots_per_dataset/`.

## Rerun experiments

Training always writes to `results/`, never to `results_paper/`.

```bash
python run_all_experiments.py --rerun --suite regression --problems wing --models gpplus
python run_all_experiments.py --rerun --suite bo --problems buckling --models gp --noise 0.08
python run_all_experiments.py --rerun --suite classification --problems stellar
```

After a rerun, the figures are built from `results/` rather than from the archive. Plot only, without training again:

```bash
python run_all_experiments.py --suite regression --source results
```

`--suite all --rerun` repeats the full paper. That is on the order of the 3,170 fits reported in the paper. Use `--problems` and `--models` to run a slice.

Model names:

| Name | Regression | Bayesian optimization |
|---|---|---|
| `gpplus` | GP+, Gaussian kernel, marginal likelihood | same GP+ run |
| `gp` | not accepted | same as `gpplus` |
| `pe` | GP+ with the power-exponential kernel | not used |
| `loo` | GP+ with the leave-one-out log pseudo-likelihood | not used |
| `gpytorch` | GPyTorch with L-BFGS | not used |
| `pfn25` | TabPFN v2.5 | TabPFN v2.5 |
| `pfn20` | TabPFN v2.0 | TabPFN v2.0 |

Regression also accepts `--sections benchmarks logscale onedim` and `--noise 0.002 0.08`.

## What each suite runs

Regression benchmarks, 10 repeats, 5,000 test points, noise levels 0.002 and 0.08 times the response scale, training sizes \(N = 5 D_x\) and \(N = 20 D_x\):

| Problem | \(D_x\) |
|---|---|
| Buckling | 4 |
| Borehole | 8 |
| Wing weight | 10 |
| Ackley | 20 and 40 |
| Griewank | 20 |
| Zakharov | 20 |
| Dixon-Price | 40 |
| Rosenbrock | 80 |

Buckling keeps the mixed-variable kernel in every GP+ variant. The power-exponential switch applies to the other problems. The log-scale study refits Buckling and Zakharov after a log transform of the response. Metrics are reported on the original scale.

The 1D regression figure uses five noise-free problems with 20 training points: discontinuous sine, triangle wave, chirp, localized bump, and damped sine. The equations are in `experiments_regression/A22_regression_1D_reference_equations.py`.

Bayesian optimization uses the same problems except Rosenbrock. Each trial starts from \(N_0 = 5 D_x\) Sobol points, then takes at most 30 expected-improvement steps and stops after 10 iterations without improvement. GP+ maximizes EI with L-BFGS from 64 starts. TabPFN scores EI on 5,000 Sobol candidates. The archived runs, and Figure 8 in the paper, use noise 0.08. The runner can also do noise 0.002.

Classification uses Electrical Grid Stability, Truss 6D, Stellar (SDSS17), and Steel Plates Faults. GP+ fits a Dirichlet classifier with Gaussian, power-exponential, and Matérn-1/2 kernels, Adam, and 10 seeds. TabPFN v2.0 and v2.5 are the baselines. Training sizes are in the classification README. The 1D classification figure sweeps the Dirichlet concentration \(\alpha_\epsilon\).

## GP+ library

`gpplus/` is the library used to train the GP+ models. The log-scale kernel and the soft clamp from Section 3.2 live under `gpplus/kernels/` and `gpplus/constraints/`. The experiment scripts call `gpplus.utils.train_eval_gp` for regression and `gpplus.models.gpc.GPC` for classification. Bayesian optimization retrains a GP+ model at every iteration (`experiments_BO/run_BO.py`).

`docs/` is the older manual for the GP+ library. Its install page says `pip install gpplus`. Do not follow that for these experiments. That command can install a different GP+ than the copy in this folder. Use the setup above.

## Later comparison

`results/` and `results_paper/` use the same folder layout on purpose. A later script can pair those trees and report where a new run disagrees with the paper. That script is not part of this repository yet.
