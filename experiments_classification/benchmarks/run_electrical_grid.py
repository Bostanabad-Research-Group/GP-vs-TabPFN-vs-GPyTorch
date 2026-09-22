# Electrical Grid Stability (UCI id=471). Obtained via ucimlrepo.

import numpy as np
from ucimlrepo import fetch_ucirepo

from gpc_benchmark import DatasetSpec, default_output_dir, run_sweep

# All entries are lists
SWEEP = {
    "alpha_epsilon":  [0.01],
    "train_sizes":    [50, 100, 200, 300],
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


def load_electrical_grid(_cfg):
    ds = fetch_ucirepo(id=471)
    df_X = ds.data.features.copy()
    df_y = ds.data.targets.copy()

    df_X = df_X.drop(columns=[c for c in df_X.columns if c in ("stab", "p1")])
    X = df_X.values.astype(np.float64)

    y_raw = (df_y["stabf"] if "stabf" in df_y.columns else df_y.iloc[:, 0])
    y_raw = y_raw.values.astype(str)

    valid = y_raw != "nan"
    if not valid.all():
        print(f"  Dropping {(~valid).sum()} rows with NaN target")
        X, y_raw = X[valid].copy(), y_raw[valid]

    classes = sorted(np.unique(y_raw))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y_raw], dtype=np.int64)

    print(f"Electrical Grid: X={X.shape}, classes={classes}")
    print(f"  Class counts: {np.bincount(y)}")
    print(f"  Features: {list(df_X.columns)}")
    return X, y


SPEC = DatasetSpec(
    name="electrical_grid",
    output_dir=default_output_dir("electrical_grid"),
    loader=load_electrical_grid,
    test_mode="complement",
)


if __name__ == "__main__":
    run_sweep(SPEC, SWEEP)
