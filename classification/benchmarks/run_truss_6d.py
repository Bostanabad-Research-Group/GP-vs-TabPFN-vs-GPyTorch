# Truss 6D. Separate parquet files for train and test.
# Data: Sharpe et al., J. Mech. Des. 141 (2019). doi:10.1115/1.4044524

import os

import numpy as np
import pandas as pd

from gpc_benchmark import DatasetSpec, collect_levels, onehot_encode, run_sweep

# All entries are lists
SWEEP = {
    "alpha_epsilon":  [0.01],
    "train_sizes":    [50, 100, 200, 300],
    "num_mc_samples": [256],
    "adam":           [{"lr": 0.01, "num_epochs": 2000}],
    # None disables early stopping and runs the full epoch budget.
    "early_stopping": [{"val_fraction": 0.2, "patience": 20, "check_every": 10}],
    "seeds":          list(range(10)),
    "num_inits":      [16],
    "run_tabpfn":     True,
    # "gaussian", "matern_0.5", "powerexp"
    "kernels":        ["gaussian", "matern_0.5", "powerexp"],
}


DATA_PATHS = {
    "train_path": "separate_datasets/truss_6d.parquet",
    "test_path":  "separate_datasets/truss_6d_test.parquet",
    "target_col": "good",
    "cont_cols":  [0, 1, 2],           # area1, area2, area3
    "cat_cols":   [[3], [4], [5]],     # mat1, mat2, mat3
}


def load_truss_6d(cfg):
    for key in ("train_path", "test_path"):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError(
                f"Truss 6D file not found: {cfg[key]}"
            )

    target_col, cont_cols, cat_cols = cfg["target_col"], cfg["cont_cols"], cfg["cat_cols"]
    df_train = pd.read_parquet(cfg["train_path"])
    df_test = pd.read_parquet(cfg["test_path"])

    X_train_raw = df_train.drop(columns=[target_col]).to_numpy(dtype=np.float64)
    X_test_raw = df_test.drop(columns=[target_col]).to_numpy(dtype=np.float64)

    def _ensure_labels(y_raw):
        classes = sorted(np.unique(y_raw))
        mapping = {c: i for i, c in enumerate(classes)}
        return np.array([mapping[v] for v in y_raw], dtype=np.int64)

    y_train = _ensure_labels(df_train[target_col].to_numpy())
    y_test = _ensure_labels(df_test[target_col].to_numpy())

    # Levels pooled across splits so both encode to the same width
    levels = collect_levels(np.vstack([X_train_raw, X_test_raw]), cat_cols)
    X_train = onehot_encode(X_train_raw, cat_cols, cont_cols, levels=levels)
    X_test = onehot_encode(X_test_raw, cat_cols, cont_cols, levels=levels)

    print(f"Truss 6D: train_pool={X_train.shape}, test={X_test.shape}")
    print(f"  Train classes={np.unique(y_train)}, counts={np.bincount(y_train)}")
    print(f"  Test  classes={np.unique(y_test)},  counts={np.bincount(y_test)}")
    return X_train, y_train, X_test, y_test


SPEC = DatasetSpec(
    name="truss_6d",
    output_dir="truss_6d_results",
    loader=load_truss_6d,
    test_mode="external",
)


if __name__ == "__main__":
    run_sweep(SPEC, SWEEP, loader_cfg=DATA_PATHS)
