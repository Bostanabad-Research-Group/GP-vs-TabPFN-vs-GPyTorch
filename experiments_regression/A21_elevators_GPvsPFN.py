import torch
import json
from pathlib import Path
import time

import gpplus
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from tabpfn import TabPFNRegressor

from load_experimental_data import load_elevators_data
import defaults
from run_metadata import experiment_data_info, pfn_model_info


def elevators_GPvsPFN(
    num_runs=defaults.NUM_RUNS,
    train_size=10,  # total training size is train_size * input dimensions
    train_pool_size=None,  # if None, computed as total_available - test_pool_size
    test_pool_size=3000,
    split_seed=defaults.SEED,
    num_inits=defaults.TRAINER_NUM_INITS,
    num_epochs=defaults.TRAINER_NUM_EPOCHS,
    lr=defaults.TRAINER_LR,
    convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs=defaults.TRAINER_MIN_EPOCHS,
    min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
    cholesky_jitter=defaults.TRAINER_CHOLESKY_JITTER,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device=defaults.TRAINER_GP_DEVICE,
    amp_device=defaults.TRAINER_AMP_DEVICE,
    save_path="./results/elevators",
    title=None,
    standardize_X=defaults.STANDARDIZE_X,
    standardize_y=defaults.STANDARDIZE_Y,
    x_standardize_method=defaults.X_STANDARDIZE_METHOD,
    seed=defaults.SEED,
    seed_trainer=defaults.SEED_TRAINER,
    gp_dtype=defaults.DTYPE_GP,
    pfn_dtype=defaults.DTYPE_PFN,
    trainer_info=True,
    run_models=None,  # None=both, 'gp', or 'pfn'
    log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
    preprocess_pfn=defaults.PREPROCESS_PFN,
    warnings_ignore=defaults.WARNINGS_IGNORE,
):
    if warnings_ignore:
        import warnings
        warnings.filterwarnings("ignore")

    if run_models == "pfn":
        num_inits = 0

    print(f"GP Device: {gp_device}")
    print(f"TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device, random_state=seed)

    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None

    set_seed(seed)
    X_all, y_all = load_elevators_data(print_info=True)
    input_dims = X_all.shape[1]
    train_samples = train_size * input_dims
    total_available = X_all.shape[0]

    if test_pool_size < 0:
        raise ValueError("test_pool_size must be non-negative.")
    if train_pool_size is None:
        train_pool_size = total_available - test_pool_size
    if train_pool_size < 0:
        raise ValueError(
            f"Computed train_pool_size is negative ({train_pool_size}). "
            "Reduce test_pool_size."
        )

    total_pool = train_pool_size + test_pool_size
    if total_pool > total_available:
        raise ValueError(
            f"train_pool_size + test_pool_size = {total_pool}, but dataset has "
            f"{total_available} rows."
        )
    if train_samples > train_pool_size:
        raise ValueError(
            f"train_size * input_dims = {train_size} * {input_dims} = {train_samples} "
            f"cannot exceed train_pool_size ({train_pool_size})."
        )

    total_required_train = num_runs * train_samples
    use_disjoint_train_splits = total_required_train <= train_pool_size

    if title is None:
        title = (
            f"elevators_{input_dims}Dx_{train_size}Dn_"
            f"{num_inits}inits_split{split_seed}_x{num_runs}"
        )
    else:
        title = (
            f"elevators_{title}_{input_dims}Dx_{train_size}Dn_"
            f"{num_inits}inits_split{split_seed}_x{num_runs}"
        )

    split_generator = torch.Generator()
    split_generator.manual_seed(split_seed)
    split_perm = torch.randperm(total_available, generator=split_generator)
    selected_pool = split_perm[:total_pool]
    test_pool_idx = selected_pool[:test_pool_size]
    train_pool_idx = selected_pool[test_pool_size:]

    X_test_all = X_all[test_pool_idx]
    y_test_all = y_all[test_pool_idx]
    X_pool = X_all[train_pool_idx]
    y_pool = y_all[train_pool_idx]

    print("=" * 10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("=" * 10)
    print(
        f"Fixed split -> train pool: {len(X_pool)}, test: {len(X_test_all)}. "
        f"Per-run train subset: train_size * input_dims = {train_size} * {input_dims} = {train_samples}"
    )
    if use_disjoint_train_splits:
        print("Train sampling mode: disjoint across runs (no overlap)")
    else:
        print(
            "Train sampling mode: independent per run (overlap allowed) "
            f"because total required train points ({total_required_train}) "
            f"exceed train pool ({train_pool_size})"
        )

    # All features are continuous (integer-like storage must not become one-hot).
    # Inferring qual_dict on a split can also mark columns categorical with levels
    # that appear only in the test pool, which breaks encoding on X_pool alone.
    qual_dict: dict = {}
    cont_cols = list(range(input_dims))
    cat_cols = []
    source_cols = []

    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []

    total_start_time = time.time()
    disjoint_run_indices_2d = None
    if use_disjoint_train_splits:
        global_generator = torch.Generator()
        global_generator.manual_seed(seed)
        global_perm = torch.randperm(train_pool_size, generator=global_generator)
        disjoint_run_indices_2d = global_perm[: num_runs * train_samples].reshape(num_runs, train_samples)

    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        if use_disjoint_train_splits:
            run_train_idx = disjoint_run_indices_2d[i]
        else:
            run_generator = torch.Generator()
            run_generator.manual_seed(run_seed)
            run_perm = torch.randperm(train_pool_size, generator=run_generator)
            run_train_idx = run_perm[:train_samples]

        X_train = X_pool[run_train_idx].detach().clone().to(dtype=gp_dtype)
        y_train = y_pool[run_train_idx].detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        X_train_raw_for_pfn = X_train.detach().clone()
        X_test_raw_for_pfn = X_test.detach().clone()

        if standardize_X:
            if x_standardize_method == 0:
                Xscaler = gpplus.utils.StandardScaler()
                X_scaling_type = "StandardScaler (Gaussian)"
            elif x_standardize_method == 1:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                X_scaling_type = "UniformScaler [0, 1]"
            elif x_standardize_method == 2:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                X_scaling_type = "UniformScaler [-1, 1]"
            else:
                raise ValueError(
                    f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}"
                )
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
        else:
            X_scaling_type = "None"

        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        if run_models in [None, "gp"]:
            print(f"\n--- {title} GP Training ---")
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=defaults.SF_kernel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )

            gp_metric, _, _, gp_trainer_info = train_eval_gp(
                model,
                X_test,
                y_test,
                num_epochs=num_epochs,
                seed=run_seed,
                num_inits=num_inits,
                lr=lr,
                convergence_patience=convergence_patience,
                min_epochs=min_epochs,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                initializer_class=initializer_class,
                device=gp_device,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,
                trainer_info=trainer_info,
                callbacks=(
                    defaults.get_default_gp_callbacks(
                        optimizer_class,
                        callback_save_path=callback_save_path,
                        log_lbfgs_inner=log_lbfgs_inner,
                    )
                    if (trainer_info or log_lbfgs_inner)
                    else []
                ),
                callback_save_path=callback_save_path,
                cholesky_jitter=cholesky_jitter,
                optimizer_kwargs=optimizer_kwargs,
                log_lbfgs_inner=log_lbfgs_inner,
            )
            GPPlus_metrics.append(gp_metric)

            if gp_trainer_info:
                gp_trainer_info["run"] = i + 1
                gp_trainer_info["metrics"] = gp_metric
                GPTrainer_info.append(gp_trainer_info)

        if run_models in [None, "pfn"]:
            print(f"\n--- {title} TabPFN Training ---")
            if preprocess_pfn:
                pfn_X_train, pfn_X_test = X_train, X_test
                pfn_y_train = y_train_normal if standardize_y else y_train
                pfn_y_test = y_test
                pfn_y_mean, pfn_y_std = y_train_mean, y_train_std
            else:
                pfn_X_train = X_train_raw_for_pfn
                pfn_X_test = X_test_raw_for_pfn
                pfn_y_train = y_train
                pfn_y_test = y_test
                pfn_y_mean, pfn_y_std = None, None

            tabpfn_metric, _, _ = train_eval_PFN(
                pfn_X_train,
                pfn_X_test,
                pfn_y_train,
                pfn_y_test,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                source_cols=source_cols,
                y_train_mean=pfn_y_mean,
                y_train_std=pfn_y_std,
                record_y_train_mean=y_train_mean if standardize_y else None,
                record_y_train_std=y_train_std if standardize_y else None,
            )
            TabPFN_metrics.append(tabpfn_metric)

        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item()),
            }
            shared_experiment_info = experiment_data_info(
                cat_cols=cat_cols,
                cont_cols=cont_cols,
                source_cols=source_cols,
                qual_dict=qual_dict,
                input_dim=X_train.shape[1],
                train_samples=X_train.shape[0],
                test_samples=X_test.shape[0],
                standardize_X=standardize_X,
                standardize_y=standardize_y,
                x_standardize_method=x_standardize_method,
                X_scaling_type=X_scaling_type,
                y_train_mean=y_train_mean,
                y_train_std=y_train_std,
                y_test_mean=y_test_stats["y_test_mean"],
                y_test_std=y_test_stats["y_test_std"],
                num_runs=num_runs,
                seed=seed,
                seed_trainer=seed_trainer,
                preprocess_pfn=preprocess_pfn,
                pfn_dtype=pfn_dtype,
                train_pool_size=train_pool_size,
                test_pool_size=test_pool_size,
                split_seed=split_seed,
            )
            if run_models in [None, "gp"]:
                gp_model_info = {
                    **shared_experiment_info,
                    "model_str": str(model),
                    "dtype": str(gp_dtype),
                    "device": str(gp_device),
                    "num_epochs": num_epochs,
                    "num_inits": num_inits,
                    "lr": lr,
                    "optimizer": optimizer_class.__name__,
                    "convergence_patience": convergence_patience,
                    "initializer": initializer_class.__name__ if initializer_class else None,
                }
            if run_models in [None, "pfn"]:
                tabpfn_model_info = pfn_model_info(
                    regressor, experiment_data=shared_experiment_info
                )

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    TabPFN_summary = (
        analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
        if run_models in [None, "pfn"]
        else None
    )
    GPPlus_summary = (
        analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title)
        if run_models in [None, "gp"]
        else None
    )

    if save_path is not None:
        if run_models is None:
            plot_metrics(
                TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path
            )

        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = run_models if run_models is not None else "gpVpfn"
        combined_data = {}
        if run_models in [None, "gp"]:
            combined_data["gp_data"] = {
                "summary": GPPlus_summary,
                "metrics": GPPlus_metrics,
                "gp_model_info": gp_model_info,
            }
        if run_models in [None, "pfn"]:
            combined_data["tabpfn_data"] = {
                "summary": TabPFN_summary,
                "metrics": TabPFN_metrics,
                "pfn_model_info": tabpfn_model_info,
            }
        _defaults_path = Path(__file__).resolve().parent / "defaults.py"
        if _defaults_path.is_file():
            combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
        (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))

        if trainer_info and GPTrainer_info and run_models in [None, "gp"]:
            trainer_analysis_dir = Path(save_path) / "trainer_analysis"
            trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
            trainer_info_by_run = {
                f"run_{entry.get('run', idx + 1)}": entry for idx, entry in enumerate(GPTrainer_info)
            }
            trainer_info_data = {
                "title": title,
                "num_runs": num_runs,
                "num_inits_per_run": num_inits,
                "trainer_info": trainer_info_by_run,
            }
            trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
            trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))

    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("=" * 60)
    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    elevators_GPvsPFN(
        num_runs=10,
        train_size=20,
        test_pool_size=3000,
        split_seed=42,
        save_path="./results/elevators/temp",
    )
