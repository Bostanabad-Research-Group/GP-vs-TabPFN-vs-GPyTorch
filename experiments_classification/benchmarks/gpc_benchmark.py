# Functions shared by the four run scripts
# Model: Dirichlet-based GP (Milios et al)
# Shared kernel (with the exception of nugget) between classes
# Heteroscedastic noise

import itertools
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from gpplus.kernels.log_scale_kernel import LogScaleKernel
from gpplus.models.gpc import GPC
from gpplus.training.eval_classification import compute_ece, evaluate_gpc_model
from gpplus.training.trainer import GPTrainer
from gpplus.utils.factory import build_scaled_kernel
from gpytorch.kernels import MaternKernel

from validation_stop_condition import ValidationNLLStopCondition

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    try:
        from tabpfn.constants import ModelVersion
        TABPFN_V2_AVAILABLE = True
    except ImportError:
        TABPFN_V2_AVAILABLE = False
except ImportError as _e:
    TABPFN_AVAILABLE = False
    TABPFN_V2_AVAILABLE = False
    print(f"TabPFN not available: {_e}")


def default_output_dir(name):
    """New runs go under results/classification_results/classification_new_results/<name>/."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from result_paths import new_results

    return os.path.join(str(new_results("classification")), name)


DTYPE = torch.float64

# Validation points excluded from train and test sets
MIN_VAL_POINTS = 10

# fixed_power=None means the PowerExp power is learned
KERNEL_CONFIGS = {
    "gaussian":   {"kind": "gaussian"},
    "matern_0.5": {"kind": "matern", "nu": 0.5},
    "powerexp":   {"kind": "powerexp", "fixed_power": None},
}

KERNEL_LABELS = {
    "gaussian":   "RBF ARD",
    "matern_0.5": "Matern nu=0.5 ARD",
    "powerexp":   "PowerExp ARD",
}

TABPFN_MODEL_NAMES = {
    "v25": "TabPFN v2.5",
    "v2":  "TabPFN v2.0",
}

class DatasetSpec:
    def __init__(self, name, output_dir, loader, test_mode, test_size=None):
        if test_mode not in ("external", "complement", "subsample"):
            raise ValueError(f"Unknown test_mode {test_mode!r}")
        if test_mode == "subsample" and not test_size:
            raise ValueError("test_mode='subsample' requires test_size")
        self.name = name
        self.output_dir = output_dir
        self.loader = loader
        self.test_mode = test_mode
        self.test_size = test_size


def scale_inputs(x_train, x_test):
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def collect_levels(X, cat_cols):
    levels = []
    for col_group in cat_cols:
        if len(col_group) > 1:
            levels.append(None)
        else:
            levels.append(sorted(np.unique(X[:, col_group[0]].astype(int))))
    return levels


def onehot_encode(X, cat_cols, cont_cols, levels=None):
    parts = [X[:, cont_cols]]
    for g, col_group in enumerate(cat_cols):
        if len(col_group) > 1:
            parts.append(X[:, col_group].astype(np.float64))
            continue
        col = X[:, col_group[0]].astype(int)
        unique_vals = sorted(np.unique(col)) if levels is None else levels[g]
        for v in unique_vals:
            parts.append((col == v).astype(np.float64).reshape(-1, 1))
    return np.hstack(parts)


def _allocate_stratified_counts(class_counts, n_total):
    proportions = class_counts / class_counts.sum()
    raw = proportions * n_total
    base = np.floor(raw).astype(int)
    for i, count in enumerate(class_counts):
        if base[i] == 0 and count > 0:
            base[i] = 1
    deficit = int(n_total - base.sum())
    if deficit > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:deficit]:
            base[idx] += 1
    elif deficit < 0:
        for idx in np.argsort(-base):
            if deficit == 0:
                break
            remove = min(-deficit, base[idx] - 1)
            base[idx] -= remove
            deficit += remove
    for i, count in enumerate(class_counts):
        if base[i] > count:
            overflow = base[i] - count
            base[i] = count
            for j in np.argsort(-(class_counts - base)):
                if j == i or overflow == 0:
                    continue
                take = min(overflow, class_counts[j] - base[j])
                base[j] += take
                overflow -= take
    return base


def validation_reserve(train_sizes, val_fraction):
    if val_fraction <= 0:
        return 0
    return max(MIN_VAL_POINTS, round(max(train_sizes) * val_fraction))


def train_universe_size(train_sizes, val_fraction):
    return max(train_sizes) + validation_reserve(train_sizes, val_fraction)


def stratified_nested_train_pools(y, train_sizes, max_train_size, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    class_order = {}
    for c in classes:
        idx_c = np.where(y == c)[0]
        class_order[c] = idx_c[rng.permutation(len(idx_c))]
    class_counts = np.array([len(class_order[c]) for c in classes], dtype=np.int64)
    pools = {}
    for ts in sorted(set(list(train_sizes) + [max_train_size])):
        counts = _allocate_stratified_counts(class_counts, ts)
        picked = [class_order[c][:counts[ci]] for ci, c in enumerate(classes)]
        pools[ts] = np.concatenate(picked)
    return pools


def stratified_test_sample(y, candidate_idx, n_test, seed):
    rng = np.random.default_rng(seed + 99999)
    y_cand = y[candidate_idx]
    classes = np.unique(y_cand)
    counts = np.array([np.sum(y_cand == c) for c in classes], dtype=np.int64)
    n_test = min(n_test, len(candidate_idx))
    alloc = _allocate_stratified_counts(counts, n_test)
    selected = []
    for c, n_take in zip(classes, alloc, strict=True):
        idx_c = candidate_idx[np.where(y_cand == c)[0]]
        chosen = idx_c[rng.choice(len(idx_c), size=int(n_take), replace=False)]
        selected.append(chosen)
    return np.concatenate(selected)


def make_validation_split(idx_train, idx_universe, n, val_fraction, seed, dataset_name=""):
    if val_fraction <= 0:
        return None

    n_val = max(MIN_VAL_POINTS, round(n * val_fraction))
    rng = np.random.default_rng(seed + 77777)
    available = np.setdiff1d(idx_universe, idx_train, assume_unique=False)
    if len(available) < n_val:
        print(f"[warn] {dataset_name} n={n}: reserve has {len(available)} point(s), "
              f"need {n_val} for validation — early stopping disabled for this config.",
              flush=True)
        return None
    return rng.choice(available, size=n_val, replace=False)


# Unbatched (batch_shape=()): one hyperparameter set shared across classes
# Sharing ties parameter values only
# Latent processes stay independent a priori
def build_kernel(kernel_label, input_dim):
    if kernel_label not in KERNEL_CONFIGS:
        raise ValueError(f"Unknown kernel {kernel_label!r}. "
                         f"Expected one of {list(KERNEL_CONFIGS)}.")
    cfg = KERNEL_CONFIGS[kernel_label]
    batch_shape = torch.Size()

    if cfg["kind"] == "matern":
        return LogScaleKernel(
            MaternKernel(nu=cfg["nu"], batch_shape=batch_shape, ard_num_dims=input_dim),
            batch_shape=batch_shape,
        )
    return build_scaled_kernel(
        kind=cfg["kind"],
        batch_shape=batch_shape,
        ard_num_dims=input_dim,
        fixed_power=cfg.get("fixed_power"),
    )


def build_model(train_x, train_y, kernel_label, input_dim, alpha_epsilon):
    covar_module = build_kernel(kernel_label, input_dim)
    return GPC(train_x=train_x, train_y=train_y,
               covar_module=covar_module, alpha_epsilon=alpha_epsilon)


def run_gp(X_train_pool, y_train_pool, X_test, y_test, input_dim,
           idx_train, idx_val, seed, kernel, alpha_epsilon, num_inits,
           adam, early_stopping, mc_sample_values):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_tr_raw = X_train_pool[idx_train]
    y_tr = y_train_pool[idx_train]
    X_tr, X_te = scale_inputs(X_tr_raw, X_test)

    train_x = torch.tensor(X_tr, dtype=DTYPE)
    train_y = torch.tensor(y_tr, dtype=torch.long)
    test_x = torch.tensor(X_te, dtype=DTYPE)
    test_y = torch.tensor(y_test, dtype=torch.long)

    model = build_model(train_x, train_y, kernel, input_dim, alpha_epsilon)

    trainer_kwargs = {
        "model": model,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": adam["lr"]},
        "num_epochs": adam["num_epochs"],
        "num_inits": num_inits,
        "seed": seed,
        "stop_conditions": [],
    }

    if idx_val is not None and early_stopping is not None:
        _, X_val = scale_inputs(X_tr_raw, X_train_pool[idx_val])
        trainer_kwargs["stop_conditions"] = [ValidationNLLStopCondition(
            val_x=torch.tensor(X_val, dtype=DTYPE),
            val_y=torch.tensor(y_train_pool[idx_val], dtype=torch.long),
            patience=early_stopping["patience"],
            check_every=early_stopping["check_every"],
        )]

    t0 = time.perf_counter()
    trainer = GPTrainer(**trainer_kwargs)
    results = trainer.train()
    train_time = time.perf_counter() - t0

    valid = [r for r in results if r.get("loss") is not None and not r.get("error")]
    if not valid:
        raise RuntimeError("every initialization failed")
    loss = min(r["loss"] for r in valid)

    stopped_epoch = None
    for sc in trainer_kwargs["stop_conditions"]:
        if getattr(sc, "stopped_epoch", None) is not None:
            stopped_epoch = sc.stopped_epoch

    scores = {}
    for n_mc in mc_sample_values:
        out = evaluate_gpc_model(model, test_x, test_y, num_mc_samples=n_mc)
        scores[n_mc] = {
            "accuracy": float(out.get("accuracy", float("nan"))),
            "ece": float(out.get("ece", float("nan"))),
        }

    return {"loss": float(loss), "train_time": train_time,
            "stopped_epoch": stopped_epoch, "scores": scores}


def run_tabpfn(X_train_pool, y_train_pool, X_test, y_test, idx_train,
               version_key, seed):
    random.seed(seed)
    np.random.seed(seed)

    X_tr, X_te = scale_inputs(X_train_pool[idx_train], X_test)
    y_tr = y_train_pool[idx_train]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if version_key == "v25":
        clf = TabPFNClassifier(device=device)
    else:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)

    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)
    train_time = time.perf_counter() - t0

    accuracy = float((probs.argmax(axis=1) == y_test).mean())
    ece = compute_ece(torch.tensor(probs, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long), n_bins=10)
    return {"accuracy": accuracy, "ece": float(ece), "train_time": train_time}


CONFIG_COLUMNS = [
    "kernel", "alpha_epsilon", "num_inits", "num_mc_samples",
    "adam_lr", "adam_epochs", "val_fraction", "val_patience", "val_check_every",
]


def _validate_sweep(sweep):
    required = {"kernels", "alpha_epsilon", "train_sizes", "seeds", "num_inits",
                "num_mc_samples", "adam", "early_stopping", "run_tabpfn"}
    missing = required - set(sweep)
    if missing:
        raise ValueError(f"SWEEP is missing {sorted(missing)}")
    for key in required - {"run_tabpfn"}:
        if not isinstance(sweep[key], (list, tuple)) or len(sweep[key]) == 0:
            raise ValueError(f"SWEEP['{key}'] must be a non-empty list")
    for adam in sweep["adam"]:
        if not {"lr", "num_epochs"} <= set(adam):
            raise ValueError("each SWEEP['adam'] entry needs 'lr' and 'num_epochs'")
    for es in sweep["early_stopping"]:
        if es is None:
            continue
        if not {"val_fraction", "patience", "check_every"} <= set(es):
            raise ValueError("each SWEEP['early_stopping'] entry needs "
                             "'val_fraction', 'patience' and 'check_every', "
                             "or must be None to disable it")


def _fit_configs(sweep):
    return [
        {"kernel": k, "alpha_epsilon": a, "num_inits": ni,
         "adam": adam, "early_stopping": es}
        for k, a, ni, adam, es in itertools.product(
            sweep["kernels"], sweep["alpha_epsilon"], sweep["num_inits"],
            sweep["adam"], sweep["early_stopping"])
    ]


def _config_row(fit_cfg, n_mc):
    es = fit_cfg["early_stopping"] or {}
    return {
        "kernel": fit_cfg["kernel"],
        "alpha_epsilon": fit_cfg["alpha_epsilon"],
        "num_inits": fit_cfg["num_inits"],
        "num_mc_samples": n_mc,
        "adam_lr": fit_cfg["adam"]["lr"],
        "adam_epochs": fit_cfg["adam"]["num_epochs"],
        "val_fraction": es.get("val_fraction", 0.0),
        "val_patience": es.get("patience", float("nan")),
        "val_check_every": es.get("check_every", float("nan")),
    }


def _seed_data(spec, seed, loaded, train_sizes, val_fraction):
    universe = train_universe_size(train_sizes, val_fraction)

    if spec.test_mode == "external":
        X_pool, y_pool, X_test, y_test = loaded
        pools = stratified_nested_train_pools(y_pool, train_sizes, universe, seed)
        return X_pool, y_pool, X_test, y_test, pools

    X, y = loaded
    pools = stratified_nested_train_pools(y, train_sizes, universe, seed)
    mask = np.ones(len(y), dtype=bool)
    mask[pools[universe]] = False
    idx_cand = np.where(mask)[0]
    if spec.test_mode == "subsample":
        idx_cand = stratified_test_sample(y, idx_cand, spec.test_size, seed)
    print(f"  Test size: {len(idx_cand)}, counts: {np.bincount(y[idx_cand])}")
    return X, y, X[idx_cand], y[idx_cand], pools


# Write the results CSV and seed-aggregated summary
def save_results(rows, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)

    raw_path = os.path.join(output_dir, f"{dataset_name}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    group_cols = ["model"] + CONFIG_COLUMNS + ["train_size"]
    present = [c for c in group_cols if c in df.columns]
    summary = []
    for keys, grp in df.groupby(present, dropna=False):
        row = dict(zip(present, keys if isinstance(keys, tuple) else (keys,),
                       strict=True))
        row.update({
            "final_train_nll_mean": grp["loss"].mean(),
            "final_train_nll_std": grp["loss"].std(),
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(),
            "ece_mean": grp["ece"].mean(),
            "ece_std": grp["ece"].std(),
            "train_time_mean": grp["train_time"].mean(),
            "train_time_std": grp["train_time"].std(),
            "stopped_epoch_mean": grp["stopped_epoch"].dropna().mean(),
            "n_seeds": int(grp["seed"].nunique()),
        })
        summary.append(row)

    df_summary = pd.DataFrame(summary).sort_values(present)
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    return raw_path, summary_path


def run_sweep(spec, sweep, loader_cfg=None):
    _validate_sweep(sweep)

    train_sizes = sorted(sweep["train_sizes"])
    fit_configs = _fit_configs(sweep)
    mc_values = sorted(sweep["num_mc_samples"])
    n_runs = (len(fit_configs) * len(train_sizes) * len(sweep["seeds"]))

    print(f"\n{'=' * 70}\nDataset: {spec.name}  (output_dir={spec.output_dir})")
    print(f"{len(fit_configs)} fit config(s) x {len(train_sizes)} train size(s) "
          f"x {len(sweep['seeds'])} seed(s) = {n_runs} fits")
    print(f"{'=' * 70}")

    loaded = spec.loader(loader_cfg or {})
    input_dim = loaded[0].shape[1]

    max_val_fraction = max(
        (es or {}).get("val_fraction", 0.0) for es in sweep["early_stopping"])

    rows = []
    for seed in sweep["seeds"]:
        print(f"\n=== Seed {seed} ===")
        X_pool, y_pool, X_test, y_test, pools = _seed_data(
            spec, seed, loaded, train_sizes, max_val_fraction)
        idx_universe = pools[train_universe_size(train_sizes, max_val_fraction)]
        tabpfn_cache = {}

        for n in train_sizes:
            idx_train = pools[n]

            for fit_cfg in fit_configs:
                es = fit_cfg["early_stopping"]
                val_fraction = (es or {}).get("val_fraction", 0.0)
                idx_val = make_validation_split(
                    idx_train, idx_universe, n, val_fraction, seed,
                    dataset_name=spec.name)

                desc = (f"  N={n:4d}  kernel={fit_cfg['kernel']}  "
                        f"alpha={fit_cfg['alpha_epsilon']}  "
                        f"inits={fit_cfg['num_inits']}  seed={seed}")
                print(desc, end=" ... ", flush=True)

                try:
                    out = run_gp(
                        X_pool, y_pool, X_test, y_test, input_dim,
                        idx_train, idx_val, seed,
                        kernel=fit_cfg["kernel"],
                        alpha_epsilon=fit_cfg["alpha_epsilon"],
                        num_inits=fit_cfg["num_inits"],
                        adam=fit_cfg["adam"],
                        early_stopping=es,
                        mc_sample_values=mc_values,
                    )
                    stop_str = (f"  stopped@{out['stopped_epoch']}"
                                if out["stopped_epoch"] else "")
                    head = out["scores"][mc_values[0]]
                    print(f"loss={out['loss']:.4f}  acc={head['accuracy']:.4f}  "
                          f"ece={head['ece']:.4f}  t={out['train_time']:.1f}s{stop_str}")
                    for n_mc in mc_values:
                        rows.append({
                            "model": "GP+",
                            **_config_row(fit_cfg, n_mc),
                            "seed": seed, "train_size": n,
                            "loss": out["loss"],
                            "accuracy": out["scores"][n_mc]["accuracy"],
                            "ece": out["scores"][n_mc]["ece"],
                            "train_time": out["train_time"],
                            "stopped_epoch": out["stopped_epoch"],
                        })
                # A failed config must not end the sweep.
                except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    print(f"FAILED: {e}")
                    traceback.print_exc()
                    for n_mc in mc_values:
                        rows.append({
                            "model": "GP+",
                            **_config_row(fit_cfg, n_mc),
                            "seed": seed, "train_size": n,
                            "loss": float("nan"), "accuracy": float("nan"),
                            "ece": float("nan"), "train_time": float("nan"),
                            "stopped_epoch": None,
                        })

            if sweep["run_tabpfn"] and TABPFN_AVAILABLE:
                for vkey, model_name in TABPFN_MODEL_NAMES.items():
                    if vkey == "v2" and not TABPFN_V2_AVAILABLE:
                        continue
                    if (n, vkey) not in tabpfn_cache:
                        print(f"  N={n:4d}  {model_name}  seed={seed}",
                              end=" ... ", flush=True)
                        try:
                            res = run_tabpfn(X_pool, y_pool, X_test, y_test,
                                             idx_train, vkey, seed)
                            print(f"acc={res['accuracy']:.4f}  ece={res['ece']:.4f}  "
                                  f"t={res['train_time']:.1f}s")
                        # A failed baseline must not end the sweep.
                        except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                            res = {"accuracy": float("nan"), "ece": float("nan"),
                                   "train_time": float("nan")}
                            print(f"FAILED: {e}")
                            traceback.print_exc()
                        tabpfn_cache[(n, vkey)] = res
                    res = tabpfn_cache[(n, vkey)]
                    rows.append({
                        "model": model_name,
                        **dict.fromkeys(CONFIG_COLUMNS),
                        "seed": seed, "train_size": n,
                        "loss": float("nan"),
                        "accuracy": res["accuracy"], "ece": res["ece"],
                        "train_time": res["train_time"], "stopped_epoch": None,
                    })

    return save_results(rows, spec.name, spec.output_dir)
