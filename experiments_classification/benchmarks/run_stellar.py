# Stellar Classification (SDSS17). Kaggle CSV in separate_datasets/.
# Data: SDSS DR17, via Kaggle by fedesoriano.

import os

import numpy as np
import pandas as pd

from gpc_benchmark import DatasetSpec, default_output_dir, run_sweep

_HERE = os.path.dirname(os.path.abspath(__file__))

# All entries are lists
SWEEP = {
    "alpha_epsilon":  [0.01],
    "train_sizes":    [5, 10, 20, 30],
    "num_mc_samples": [256],
    "adam":           [{"lr": 0.01, "num_epochs": 2000}],
    # None disables early stopping
    "early_stopping": [{"val_fraction": 0.2, "patience": 20, "check_every": 10}],
    "seeds":          list(range(10)),
    "num_inits":      [16],
    "run_tabpfn":     True,
    # "gaussian", "matern_0.5", "powerexp"
    "kernels":        ["gaussian", "matern_0.5", "powerexp"],
}


TEST_SIZE = 1000

DATA_CONFIG = {
    "dataset_path": os.path.join(_HERE, "separate_datasets", "star_classification.csv"),
    "target_col":   "class",
    "drop_cols": [
        "obj_ID", "run_ID", "rerun_ID", "cam_col",
        "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID",
    ],
}


def load_stellar(cfg):
    path = cfg["dataset_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}"
        )

    df = pd.read_csv(path)
    y_raw = df[cfg["target_col"]].values.astype(str)
    df = df.drop(columns=[cfg["target_col"]]
                 + [c for c in cfg["drop_cols"] if c in df.columns])
    X = df.values.astype(np.float64)

    classes = sorted(np.unique(y_raw))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y_raw], dtype=np.int64)

    print(f"Stellar SDSS17: X={X.shape}")
    print(f"  Classes: {classes} -> {list(range(len(classes)))}")
    print(f"  Class counts: {np.bincount(y)}")
    return X, y


SPEC = DatasetSpec(
    name="stellar",
    output_dir=default_output_dir("stellar"),
    loader=load_stellar,
    test_mode="subsample",
    test_size=TEST_SIZE,
)


if __name__ == "__main__":
    run_sweep(SPEC, SWEEP, loader_cfg=DATA_CONFIG)
