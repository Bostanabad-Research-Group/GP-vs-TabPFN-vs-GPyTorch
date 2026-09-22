# Steel Plates Faults (UCI id=198). Obtained via ucimlrepo.

import numpy as np
from ucimlrepo import fetch_ucirepo

from gpc_benchmark import DatasetSpec, default_output_dir, onehot_encode, run_sweep

# All entries are lists
SWEEP = {
    "alpha_epsilon":  [0.01],
    "train_sizes":    [50, 100, 200],
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

CAT_COL_NAMES = [["TypeOfSteel_A300", "TypeOfSteel_A400"]]


def load_steel_plates(_cfg):
    ds = fetch_ucirepo(id=198)
    df_X = ds.data.features
    columns = list(df_X.columns)
    X_raw = df_X.values.astype(np.float64)

    missing = [n for grp in CAT_COL_NAMES for n in grp if n not in columns]
    if missing:
        raise ValueError(
            f"Steel Plates: declared categorical column(s) {missing} are not in "
            f"the frame ucimlrepo returned. Actual columns ({len(columns)}): "
            f"{columns}. Update CAT_COL_NAMES at the top of this script."
        )

    name_to_idx = {c: i for i, c in enumerate(columns)}
    cat_cols = [[name_to_idx[n] for n in grp] for grp in CAT_COL_NAMES]
    cat_flat = {i for grp in cat_cols for i in grp}
    cont_cols = [i for i in range(len(columns)) if i not in cat_flat]

    X = onehot_encode(X_raw, cat_cols, cont_cols)

    y = ds.data.targets.values.argmax(axis=1).astype(np.int64)
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y], dtype=np.int64)

    print(f"Steel Plates Faults: X={X.shape}, classes={np.unique(y)}, "
          f"counts={np.bincount(y)}")
    print(f"  Categorical groups (resolved): {CAT_COL_NAMES} -> {cat_cols}")
    print(f"  Continuous cols: {len(cont_cols)} (encoded first), "
          f"categorical block width: {len(cat_flat)} (encoded last)")
    return X, y


SPEC = DatasetSpec(
    name="steel_plates",
    output_dir=default_output_dir("steel_plates"),
    loader=load_steel_plates,
    test_mode="complement",
)


if __name__ == "__main__":
    run_sweep(SPEC, SWEEP)
