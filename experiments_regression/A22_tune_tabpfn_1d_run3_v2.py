"""
Apply one TabPFN config to the 5 paper 1D problems, using A22 data generation
(10 splits) but fitting only run 3.

Writes:
  results_1D/A22_regression_1D_tabpfn_tunedv2/<function>/predictions.npz
  results_1D/A22_regression_1D_tabpfn_tunedv2/1D_regression_figure/

Usage (from experiments/, SI_env):
  python A22_tune_tabpfn_1d_run3_v2.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor

import defaults
from A22_regression_1D import REGRESSION_1D_FUNCTIONS, _X_BOUNDS_DEPENDENT_FUNCTIONS, _ensure_ncrps
from A22_tune_tabpfn_1d import _ckpt_path, _make_regressor
from experimental_utils.a22_results_io import save_predictions_npz
from gpplus.utils import set_seed, train_eval_PFN
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings

warnings.filterwarnings("ignore")

BASE_ROOT = Path("./results_1D/A22_regression_1D")
OUT_ROOT = Path("./results_1D/A22_regression_1D_tabpfn_tunedv2")
PAPER_FUNCS = (
    "discontinuity",
    "triangle_wave",
    "chirp",
    "localized_bump",
    "damped_sine",
)
RUN_INDEX = 3  # 1-based; matches paper figure
NUM_RUNS = 10
TRAIN_SIZE = 20
NUM_TEST = 5000

# One config, slightly more aggressive than v1 (small-samples n16 T0.45)
TUNED_CFG = dict(
    name="small-samples_n32_T0.3",
    n_estimators=32,
    softmax_temperature=0.3,
    average_before_softmax=False,
    model_path=_ckpt_path("small-samples"),
    checkpoint="small-samples",
)


def _make_splits(function_name: str, seed: int = defaults.SEED):
    """Same Sobol + 10-split protocol as A22_regression_1D / A22_tune_tabpfn_1d."""
    fn_cfg = REGRESSION_1D_FUNCTIONS[function_name]
    generate_data_fn = fn_cfg["generate_data"]
    true_fn = fn_cfg["true_fn"]
    x_bounds = list(fn_cfg["default_x_bounds"])
    x_bounds_tuple = (float(x_bounds[0]), float(x_bounds[1]))

    if function_name in _X_BOUNDS_DEPENDENT_FUNCTIONS:

        def eval_true_fn(X: torch.Tensor) -> torch.Tensor:
            return true_fn(X, x_bounds=x_bounds_tuple)

    else:
        eval_true_fn = true_fn

    set_seed(seed)
    num_runs_gen = max(NUM_RUNS, 20)
    train_per_run = TRAIN_SIZE
    total_train = num_runs_gen * train_per_run
    total_samples = NUM_TEST + total_train

    X_train_all, y_train_all, X_test_all, y_test_all = generate_data_fn(
        n_train=total_train,
        n_test=NUM_TEST,
        dimensions=1,
        x_bounds=x_bounds,
        test_x_bounds=list(x_bounds),
        train_noise=0.0,
        test_noise=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=seed,
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)
    qual_dict = learn_encodings(X)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)

    x_test = X_test_all[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_true = eval_true_fn(X_test_all.to(dtype=torch.float64)).detach().cpu().numpy().ravel()

    return {
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
        "X_test_all": X_test_all,
        "y_test_all": y_test_all,
        "train_indices_2d": train_indices_2d,
        "source_cols": source_cols,
        "x_test": x_test,
        "y_true": y_true,
        "x_bounds": x_bounds,
    }


def fit_run3(function_name: str, cfg: dict) -> dict:
    data = _make_splits(function_name)
    i = RUN_INDEX - 1
    idx = data["train_indices_2d"][i]
    X_train = data["X_train_all"][idx].detach().clone().to(dtype=defaults.DTYPE_GP)
    y_train = data["y_train_all"][idx].detach().clone().to(dtype=defaults.DTYPE_GP)
    X_test = data["X_test_all"].detach().clone().to(dtype=defaults.DTYPE_GP)
    y_test = data["y_test_all"].detach().clone().to(dtype=defaults.DTYPE_GP)

    regressor = _make_regressor(cfg, defaults.TRAINER_AMP_DEVICE, defaults.SEED)
    metric, y_pred, y_std = train_eval_PFN(
        X_train,
        X_test,
        y_train,
        y_test,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        amp_dtype=defaults.DTYPE_PFN,
        regressor=regressor,
        source_cols=data["source_cols"],
        y_train_mean=None,
        y_train_std=None,
    )
    _ensure_ncrps(metric, y_test, y_pred, y_std)

    x_train = X_train[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_train_np = y_train.detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_std = np.asarray(y_std, dtype=np.float64).ravel() if y_std is not None else None

    out_dir = OUT_ROOT / function_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pad to 3 runs so the paper figure --run 3 index works; only run 3 is meaningful.
    runs = []
    for r in range(RUN_INDEX):
        if r == i:
            runs.append(
                {
                    "x_train": x_train,
                    "y_train": y_train_np,
                    "y_pred_tabpfn": y_pred,
                    "y_std_tabpfn": y_std,
                }
            )
        else:
            runs.append(
                {
                    "x_train": x_train,
                    "y_train": y_train_np,
                    "y_pred_tabpfn": y_pred,
                    "y_std_tabpfn": y_std,
                }
            )

    title = (
        f"A22_{function_name}_1Dx_{TRAIN_SIZE}Dn_"
        f"[{data['x_bounds'][0]},{data['x_bounds'][1]}]_tabpfnTunedV2_run{RUN_INDEX}"
    )
    save_predictions_npz(
        out_dir,
        x_test=data["x_test"],
        y_true_test=data["y_true"],
        runs=runs,
        title=title,
        function_name=function_name,
        noise_train=0.0,
        noise_test=0.0,
    )

    payload = {
        "function": function_name,
        "run_index": RUN_INDEX,
        "config": {k: v for k, v in cfg.items()},
        "metrics": {k: metric.get(k) for k in ("RRMSE", "NCRPS", "NIS_width", "NIS", "NIS_outside", "NLPD", "MAE")},
    }
    (out_dir / f"pfn_tuned_v2_run{RUN_INDEX}_{function_name}.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"  {function_name}: RRMSE={metric.get('RRMSE'):.4f} "
        f"NIS_w={metric.get('NIS_width'):.4f} NCRPS={metric.get('NCRPS'):.4f}"
    )
    return payload


def rebuild_figure() -> None:
    cmd = [
        sys.executable,
        "A22_paper_figure_1d_examples.py",
        "--math-labels",
        "--run",
        str(RUN_INDEX),
        "--results-root",
        str(BASE_ROOT),
        "--tuned-root",
        str(OUT_ROOT),
        "--out-dir",
        str(OUT_ROOT / "1D_regression_figure"),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Config: {TUNED_CFG['name']}", flush=True)
    print(f"Functions: {PAPER_FUNCS}", flush=True)
    print(f"Protocol: {NUM_RUNS} splits, fit run {RUN_INDEX} only", flush=True)

    summary = {"config": TUNED_CFG, "run_index": RUN_INDEX, "functions": {}}
    t0 = time.time()
    for fn in PAPER_FUNCS:
        print(f"\n[{fn}] regenerating {NUM_RUNS} splits, fitting run {RUN_INDEX}...", flush=True)
        summary["functions"][fn] = fit_run3(fn, TUNED_CFG)
    summary["elapsed_sec"] = time.time() - t0
    (OUT_ROOT / "tune_v2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nDone in {summary['elapsed_sec']:.1f}s -> {OUT_ROOT / 'tune_v2_summary.json'}", flush=True)

    if not args.no_figure:
        rebuild_figure()
        print("Figure ->", OUT_ROOT / "1D_regression_figure", flush=True)


if __name__ == "__main__":
    main()
