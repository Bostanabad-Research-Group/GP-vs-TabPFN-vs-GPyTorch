# Classification experiments

These scripts reproduce Section 4.5 and the one-dimensional Dirichlet example in Section 4.2.

GP+ turns classification into regression with the Dirichlet targets of Milios et al. One latent GP is fit per class. The kernels share hyperparameters except for a class-specific noise term. Training uses Adam, which is the setting in the paper. TabPFN v2.0 and v2.5 are fit with their default classifiers. There is no separate GPyTorch training script in this folder. The Matérn kernel is the GPyTorch kernel, wrapped by GP+.

The archived repository does not include the classification CSVs or the 1D figure. Those are created in `results/classification_results/classification_new_results/` when you rerun.

## Datasets

| Script | Data | \(D_x\) | Classes | Training sizes | Seeds |
|---|---|---|---|---|---|
| `benchmarks/run_electrical_grid.py` | UCI 471, Electrical Grid Stability | 11 continuous | 2 | 50, 100, 200, 300 | 0–9 |
| `benchmarks/run_truss_6d.py` | Sharpe et al. parquet in `benchmarks/separate_datasets/` | 6 (3 continuous, 3 categorical) | 2 | 50, 100, 200, 300 | 0–9 |
| `benchmarks/run_stellar.py` | SDSS17 CSV in `benchmarks/separate_datasets/` | 8 continuous | 3 | 5, 10, 20, 30 | 0–9 |
| `benchmarks/run_steel_plates.py` | UCI 198, Steel Plates Faults | 27 after encoding | 7 | 50, 100, 200 | 0–9 |
| `onedim/onedim_example.py` | Fixed 1D two-class sample | 1 | 2 | the fixed sample | GP+ at \(\alpha_\epsilon \in \{0.001, 0.01, 0.1\}\) |

Categorical inputs are one-hot encoded before GP+ sees them. TabPFN receives the same encoded matrix in these scripts. Kernels are Gaussian (RBF ARD), power-exponential, and Matérn \(\nu = 0.5\). Each GP uses 16 restarts, Adam at learning rate 0.01 for up to 2,000 epochs, and early stopping on a 20% validation split (patience 20, checked every 10 epochs). Predictive probabilities use 256 Monte Carlo samples. Metrics are training negative log likelihood, accuracy, and expected calibration error with 10 bins.

Electrical Grid and Steel Plates are downloaded with `ucimlrepo` on first run. Truss and Stellar are already in `benchmarks/separate_datasets/`.

## Run

```bash
python run_all.py --rerun
python run_all.py --rerun --problems stellar onedim
```

Outputs:

- `results/classification_results/classification_new_results/<dataset>/<dataset>_raw.csv` and `<dataset>_summary.csv`
- `results/classification_results/classification_new_results/summary/classification_final.png` and `classification_grid.pdf`
- `results/classification_results/classification_new_results/summary/classification_summary.md`
- `results/classification_results/classification_new_results/onedim/onedim_example.png`, copied to `summary/classification_1d.png`

`--per-problem-plots` also writes one figure per dataset under `classification_new_results/plots_per_dataset/` and one Markdown table per dataset under `classification_new_results/summary/tables/`.

Plot again without training:

```bash
python run_all.py --source results
```

`results/classification_results/classification_original_results/` is reserved for a future archive of these CSVs. Nothing in the current paper archive belongs in that folder yet, so a plot-only call looks in `classification_new_results/`.
