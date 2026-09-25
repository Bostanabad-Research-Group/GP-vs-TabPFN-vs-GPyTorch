# GP+ vs TabPFN vs GPyTorch

This repository reproduces the longer paper:

**On the Brittleness of Maximum Likelihood Estimation for Gaussian Process Hyperparameter Optimization.** Tyler R. Johnson, Kian Ben-Jacob, Christopher P. Muller, and Ramin Bostanabad. arXiv:2608.13793, 13 August 2026. [arxiv.org/abs/2608.13793](https://arxiv.org/abs/2608.13793)

That paper keeps the regression benchmarks and adds the studies already stored in `results/regression_results/regression_original_results/`: the five 1D regression examples, including the tuned TabPFN overlay, and the log-scale target study on Buckling and Zakharov. It also adds Bayesian optimization and classification. A full rerun includes those 1D and log-scale fits. It does not stop after the benchmark table.

The earlier paper is:

**On the Uncertainty Quantification Ability of Tabular Foundation Models.** Tyler R. Johnson, Kian Ben-Jacob, Nima Negarandeh, Oriol Vendrell-Gallart, and Ramin Bostanabad. arXiv:2606.01427, and IEEE Computing in Science & Engineering, 10 June 2026, [doi.org/10.1109/MCSE.2026.3701626](https://doi.org/10.1109/MCSE.2026.3701626). [arxiv.org/abs/2606.01427](https://arxiv.org/abs/2606.01427)

That version compares default GP+ with TabPFN v2.5 on regression, including a smaller set of 1D examples. It does not include the log-scale target study, Bayesian optimization, or classification. The code for that paper is [github.com/kianswarehouse/GPvsPFN](https://github.com/kianswarehouse/GPvsPFN).

The brittleness study trains Gaussian processes with maximum likelihood and compares them with TabPFN v2.0 and v2.5. The GP baselines are:

- **GP+**, this repository's library (`gpplus/`). It uses a log-scale kernel parameterization, a soft clamp on hyperparameters, and 16 random restarts.
- **GPyTorch**, used with its own training loop on the regression benchmarks. Regression uses L-BFGS. Classification uses Adam.

The three experiment suites match the paper sections:

| Folder | Paper section | What is compared |
|---|---|---|
| `experiments_regression/` | 4.2 and 4.3 | 1D examples, analytic benchmarks, and a log-scale target study |
| `experiments_BO/` | 4.4 | Expected-improvement Bayesian optimization |
| `experiments_classification/` | 4.5 and the 1D Dirichlet example | Four classification datasets |

All outputs live under `results/`. Each suite has three folders: the archived paper run (`*_original_results`), a new run (`*_new_results`), and the comparison (`*_results_comparison`). `results/summary.md` is the short report.

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

From the repository root, this does not train anything. It reads each suite's `*_original_results` folder and writes summary figures and tables there.

```bash
python run_all_experiments.py
```

You get these combined figures from the archive:

- `results/regression_results/regression_original_results/summary/regression_final.png` (RRMSE for every regression benchmark; the PDF also has NIS and NCRPS)
- `results/regression_results/regression_original_results/summary/regression_1d_final.png`
- `results/regression_results/regression_original_results/summary/regression_logscale_final.png`
- `results/bo_results/bo_original_results/summary/BO_final.png`

The paper's classification results are currently missing from `results/classification_results/classification_original_results/`. They should be added soon. Until then, the classification figure appears only after a rerun.

Each summary folder also contains a table in Markdown, CSV, and, for regression, LaTeX. The regression table is the same layout as Table A1 in the paper: median ± standard deviation of RRMSE and NIS.

Add per-problem figures when you want them. They are off by default because they use a lot of disk.

```bash
python run_all_experiments.py --per-problem-plots
```

Those files go in `*_original_results/summary/per_problem/` and `*_original_results/summary/tables/` when you are plotting the archive. After a rerun they go in `*_new_results/summary/` instead. Classification writes its per-dataset figures under `results/classification_results/classification_new_results/plots_per_dataset/`.

## Rerun experiments

Training always writes to `*_new_results/`, never to `*_original_results/`.

```bash
python run_all_experiments.py --rerun --suite regression --problems wing --models gpplus
python run_all_experiments.py --rerun --suite bo --problems buckling --models gp --noise 0.08
python run_all_experiments.py --rerun --suite classification --problems stellar
```

After a rerun, the figures are built from `results/` rather than from the archive. Plot only, without training again:

```bash
python run_all_experiments.py --suite regression --source results
```

`--suite all --rerun` repeats the full paper. That is on the order of the 3,170 fits reported in the paper. Use `--problems` and `--models` to run a slice. A configuration that errors or exceeds `--timeout-hours` (12 by default) is written to that suite's `*_new_results/failures.md` and the runner moves on. `python compare_to_paper.py` writes `results/summary.md` and the files in each `*_results_comparison` folder. Finished result files are left in place and are not repeated. A file counts as finished only when its dimension matches the job, so Ackley 20D does not stand in for Ackley 40D.

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

Buckling keeps the mixed-variable kernel in every GP+ variant. The power-exponential switch applies to the other problems. The log-scale study refits Buckling and Zakharov after a log transform of the response, for GP+, GP+ (PE), GP+ (LOO), and GPyTorch. Metrics are reported on the original scale. TabPFN is not refit on the log scale. The main text of the brittleness paper shows the Buckling GP+ and GPyTorch comparison. The PE, LOO, and Zakharov log-scale fits are in this repository.

The 1D regression figure uses five noise-free problems with 20 training points: discontinuous sine, triangle wave, chirp, localized bump, and damped sine. The equations are in `experiments_regression/A22_regression_1D_reference_equations.py`. The tuned TabPFN overlay uses the small-samples checkpoint, 16 estimators, and softmax temperature 0.45. A full regression rerun writes both the default 1D curves and that tuned overlay.

`--suite regression --rerun` runs benchmarks, then the log-scale study, then both 1D fits, as long as `--models` still includes `gpplus`, `pe`, `loo`, and `gpytorch`. Passing a shorter model list leaves those log-scale fits out.

Bayesian optimization uses the same problems except Rosenbrock. Each trial starts from \(N_0 = 5 D_x\) Sobol points, then takes at most 30 expected-improvement steps and stops after 10 iterations without improvement. GP+ maximizes EI with L-BFGS from 64 starts. TabPFN scores EI on 5,000 Sobol candidates. The archived runs, and Figure 8 in the paper, use noise 0.08. The runner can also do noise 0.002.

Classification uses Electrical Grid Stability, Truss 6D, Stellar (SDSS17), and Steel Plates Faults. GP+ fits a Dirichlet classifier with Gaussian, power-exponential, and Matérn-1/2 kernels, Adam, and 10 seeds. TabPFN v2.0 and v2.5 are the baselines. Training sizes are in the classification README. The 1D classification figure sweeps the Dirichlet concentration \(\alpha_\epsilon\).

## GP+ library

`gpplus/` is the library used to train the GP+ models. The log-scale kernel and the soft clamp from Section 3.2 live under `gpplus/kernels/` and `gpplus/constraints/`. The experiment scripts call `gpplus.utils.train_eval_gp` for regression and `gpplus.models.gpc.GPC` for classification. Bayesian optimization retrains a GP+ model at every iteration (`experiments_BO/run_BO.py`).

`docs/` is the older manual for the GP+ library. Its install page says `pip install gpplus`. Do not follow that for these experiments. That command can install a different GP+ than the copy in this folder. Use the setup above.

## Comparison

`python compare_to_paper.py` pairs `*_original_results` with `*_new_results`. The report is `results/summary.md`. Regression and Bayesian optimization also get a table and a figure in their `*_results_comparison` folder.
