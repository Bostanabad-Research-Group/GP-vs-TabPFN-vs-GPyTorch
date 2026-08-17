import json
import time
from pathlib import Path

import numpy as np
import defaults
from run_metadata import experiment_data_info, pfn_model_info
import gpytorch
import torch
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from load_experimental_data import (
    generate_tabpfn_1d_chirp_data,
    generate_tabpfn_1d_damped_forrester_data,
    generate_tabpfn_1d_damped_sine_data,
    generate_tabpfn_1d_discontinuity_data,
    generate_tabpfn_1d_forrester_data,
    generate_tabpfn_1d_gramacy_lee_data,
    generate_tabpfn_1d_localized_bump_data,
    generate_tabpfn_1d_sin_cubic_data,
    generate_tabpfn_1d_smooth_multisine_data,
    generate_tabpfn_1d_triangle_wave_data,
    tabpfn_1d_chirp_function,
    tabpfn_1d_damped_forrester_function,
    tabpfn_1d_damped_sine_function,
    tabpfn_1d_discontinuity_function,
    tabpfn_1d_forrester_function,
    tabpfn_1d_gramacy_lee_function,
    tabpfn_1d_localized_bump_function,
    tabpfn_1d_sin_cubic_function,
    tabpfn_1d_smooth_multisine_function,
    tabpfn_1d_triangle_wave_function,
)
from experimental_utils.a22_results_io import save_predictions_npz as persist_predictions_npz
from experimental_utils.plot_tabpfn1d_comparison import (
    save_1d_all_runs_gp_tabpfn_plot,
    save_1d_train_gp_tabpfn_plot,
)
from tabpfn import TabPFNRegressor

import gpplus
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN


# Functions whose true_fn requires an x_bounds kwarg (domain remapping/chirp phase).
_X_BOUNDS_DEPENDENT_FUNCTIONS = {"forrester", "damped_forrester", "chirp", "gramacy_lee"}


def _ensure_ncrps(metric, y_true, y_pred, output_std):
    """Inject Gaussian CRPS/NCRPS into a metric dict when missing."""
    if metric is None or y_pred is None:
        return
    if metric.get("NCRPS") is not None:
        return
    try:
        from gpplus.utils.metrics_functions import compute_crps_gaussian

        yt = y_true.detach().cpu().numpy().ravel() if hasattr(y_true, "detach") else np.asarray(y_true).ravel()
        y_std = float(yt.std())
        if y_std <= 0:
            return

        std = output_std
        if std is None and metric.get("NIS_width") is not None:
            sigma = (float(metric["NIS_width"]) * y_std) / (2.0 * 1.96)
            std = np.full(yt.shape[0], sigma, dtype=np.float64)

        if std is None:
            return

        crps = float(compute_crps_gaussian(y_true, y_pred, std))
        metric["CRPS"] = crps
        metric["NCRPS"] = crps / y_std
    except Exception as e:
        print(f"NCRPS injection failed: {e}")

REGRESSION_1D_FUNCTIONS = {
    "forrester": {
        "generate_data": generate_tabpfn_1d_forrester_data,
        "true_fn": tabpfn_1d_forrester_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Forrester benchmark",
    },
    "damped_forrester": {
        "generate_data": generate_tabpfn_1d_damped_forrester_data,
        "true_fn": tabpfn_1d_damped_forrester_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "5*damped sine + Forrester",
    },
    "sin_cubic": {
        "generate_data": generate_tabpfn_1d_sin_cubic_data,
        "true_fn": tabpfn_1d_sin_cubic_function,
        "default_x_bounds": [0.0, 1.0],
        "description": "1/4*sin(6*pi*x) + 6*x^3 - 7*x^2 + x + 0.5",
    },
    "smooth_multisine": {
        "generate_data": generate_tabpfn_1d_smooth_multisine_data,
        "true_fn": tabpfn_1d_smooth_multisine_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "sin(2*pi*x) + 0.5*sin(6*pi*x) (GP-friendly baseline)",
    },
    "chirp": {
        "generate_data": generate_tabpfn_1d_chirp_data,
        "true_fn": tabpfn_1d_chirp_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Linear chirp (non-stationary frequency)",
    },
    "discontinuity": {
        "generate_data": generate_tabpfn_1d_discontinuity_data,
        "true_fn": tabpfn_1d_discontinuity_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "0.6*sin(2*pi*x) + Heaviside jump at x=0",
    },
    "localized_bump": {
        "generate_data": generate_tabpfn_1d_localized_bump_data,
        "true_fn": tabpfn_1d_localized_bump_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Flat + narrow Gaussian spike exp(-(x/0.04)^2)",
    },
    "triangle_wave": {
        "generate_data": generate_tabpfn_1d_triangle_wave_data,
        "true_fn": tabpfn_1d_triangle_wave_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Triangle wave (periodic kinks, period 0.4)",
    },
    "gramacy_lee": {
        "generate_data": generate_tabpfn_1d_gramacy_lee_data,
        "true_fn": tabpfn_1d_gramacy_lee_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Gramacy & Lee: sin(10*pi*z)/(2z) + (z-1)^4, z in [0.5,2.5]",
    },
    "damped_sine": {
        "generate_data": generate_tabpfn_1d_damped_sine_data,
        "true_fn": tabpfn_1d_damped_sine_function,
        "default_x_bounds": [-0.5, 0.5],
        "description": "exp(-6*|x|)*sin(10*pi*x) (amplitude modulated)",
    },
}


def regression_1D_GPvsPFN(
        function_name="damped_forrester",
        num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10,
        dimensions=1,
        x_bounds=None,
        test_x_bounds=None,
        test_outside_margin=0.0,
        num_inits=defaults.TRAINER_NUM_INITS,
        num_epochs=defaults.TRAINER_NUM_EPOCHS,
        lr=defaults.TRAINER_LR,
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_epochs=defaults.TRAINER_MIN_EPOCHS,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path="./results/A22_regression_1D",
        title=None,
        standardize_X=defaults.STANDARDIZE_X,
        standardize_y=defaults.STANDARDIZE_Y,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,
        noise_train=0.0,
        noise_test=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype=defaults.DTYPE_GP,
        pfn_dtype=defaults.DTYPE_PFN,
        trainer_info=True,
        run_models=None,
        log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
        preprocess_pfn=defaults.PREPROCESS_PFN,
        plot_1d_comparison=True,
        save_predictions_npz=True,
        warnings_ignore=defaults.WARNINGS_IGNORE,
        tabpfn_kwargs=None,
    ):

    if warnings_ignore:
        import warnings
        warnings.filterwarnings("ignore")

    if run_models == "pfn":
        num_inits = 0

    if function_name not in REGRESSION_1D_FUNCTIONS:
        raise ValueError(
            f"Unknown function_name={function_name!r}; "
            f"choose from {list(REGRESSION_1D_FUNCTIONS)}"
        )
    fn_cfg = REGRESSION_1D_FUNCTIONS[function_name]
    generate_data_fn = fn_cfg["generate_data"]
    true_fn = fn_cfg["true_fn"]
    fn_description = fn_cfg["description"]

    if x_bounds is None:
        x_bounds = list(fn_cfg["default_x_bounds"])
    x_bounds_tuple = (float(x_bounds[0]), float(x_bounds[1]))

    if function_name in _X_BOUNDS_DEPENDENT_FUNCTIONS:
        def eval_true_fn(X: torch.Tensor) -> torch.Tensor:
            return true_fn(X, x_bounds=x_bounds_tuple)
    else:
        eval_true_fn = true_fn
    if test_x_bounds is None:
        test_x_bounds = [x_bounds[0] - test_outside_margin, x_bounds[1] + test_outside_margin]
    if dimensions != 1:
        raise ValueError("This experiment is 1D only; use dimensions=1.")

    if title is None:
        title = f"A22_{function_name}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else:
        title = f"A22_{function_name}_{title}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"

    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    _pfn_kw = dict(device=amp_device, random_state=seed)
    if tabpfn_kwargs:
        _pfn_kw.update(tabpfn_kwargs)
    regressor = TabPFNRegressor(**_pfn_kw)
    if tabpfn_kwargs:
        print(f" TabPFN kwargs: {tabpfn_kwargs}")
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None

    set_seed(seed)

    num_runs_gen = max(num_runs, 20)
    train_per_run = train_size * dimensions
    total_train = num_runs_gen * train_per_run
    total_samples = num_test + total_train

    print(
        f"Generating {total_samples} unique Sobol samples for 1D {function_name}: {fn_description}\n"
        f"\tTest samples: {num_test} / Train samples: {total_train}"
    )

    X_train_all, y_train_all, X_test_all, y_test_all = generate_data_fn(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("=" * 10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("=" * 10)

    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []
    all_runs_plot_data: list[dict] = []

    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)

    x_test_1d = X_test_all[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_true_1d = eval_true_fn(X_test_all.to(dtype=torch.float64)).detach().cpu().numpy().ravel()

    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        run_train_indices = train_indices_2d[i]
        X_train = X_train_all[run_train_indices]
        y_train = y_train_all[run_train_indices]

        x_train_plot = X_train[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
        y_train_plot = y_train.detach().cpu().to(dtype=torch.float64).numpy().ravel()
        y_pred_gp_run = None
        y_pred_tabpfn_run = None
        y_std_gp_run = None
        y_std_tabpfn_run = None

        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
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
                raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
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
            if (i == 0) or (i == num_runs - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                print(model)

            gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
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
                optimizer_kwargs=optimizer_kwargs,
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
                log_lbfgs_inner=log_lbfgs_inner,
            )

            _ensure_ncrps(gp_metric, y_test, y_pred_gp, output_std_gp)
            GPPlus_metrics.append(gp_metric)

            if gp_trainer_info:
                gp_trainer_info["run"] = i + 1
                gp_trainer_info["metrics"] = gp_metric
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nGP Results (Run {i+1}/{num_runs})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
            y_pred_gp_run = y_pred_gp
            y_std_gp_run = output_std_gp

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

            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
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
            _ensure_ncrps(tabpfn_metric, y_test, y_pred_tabpfn, output_std_tabpfn)
            TabPFN_metrics.append(tabpfn_metric)

            print(f"\nTabPFN Results (Run {i+1}/{num_runs})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
            y_pred_tabpfn_run = y_pred_tabpfn
            y_std_tabpfn_run = output_std_tabpfn

        if plot_1d_comparison and save_path is not None:
            try:
                out_plot_dir = Path(save_path) / "plots" / "prediction_runs"
                fp = save_1d_train_gp_tabpfn_plot(
                    x_train_plot,
                    y_train_plot,
                    x_test_1d,
                    y_pred_gp_run,
                    y_pred_tabpfn_run,
                    y_std_gp_run,
                    y_std_tabpfn_run,
                    out_plot_dir,
                    title=title,
                    run_index=i + 1,
                    y_true_test=y_true_1d,
                    file_suffix=f"ntr{noise_train}_nte{noise_test}",
                )
                print(f"Saved 1D comparison plot: {fp}")
            except Exception as _plot_e:
                print(f"1D prediction plot failed (run {i+1}): {_plot_e}")

        if plot_1d_comparison:
            run_record = {
                "x_train": x_train_plot,
                "y_train": y_train_plot,
                "y_pred_gp": y_pred_gp_run,
                "y_pred_tabpfn": y_pred_tabpfn_run,
                "y_std_gp": y_std_gp_run,
                "y_std_tabpfn": y_std_tabpfn_run,
            }
            all_runs_plot_data.append(run_record)

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
                test_samples=num_test,
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
                noise_train=noise_train,
                noise_test=noise_test,
                noise_type=noise_type,
                preprocess_pfn=preprocess_pfn,
                pfn_dtype=pfn_dtype,
                dimensions=dimensions,
                x_bounds=x_bounds,
                function_name=function_name,
                function_description=fn_description,
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

    if plot_1d_comparison and save_path is not None and all_runs_plot_data:
        try:
            out_plot_dir = Path(save_path) / "plots" / "prediction_runs"
            fp_all = save_1d_all_runs_gp_tabpfn_plot(
                all_runs_plot_data,
                x_test_1d,
                out_plot_dir,
                title=title,
                y_true_test=y_true_1d,
                file_suffix=f"ntr{noise_train}_nte{noise_test}",
            )
            print(f"Saved all-runs 1D comparison plot: {fp_all}")
        except Exception as _plot_all_e:
            print(f"All-runs 1D prediction plot failed: {_plot_all_e}")

    if save_predictions_npz and save_path is not None and all_runs_plot_data:
        try:
            npz_path = persist_predictions_npz(
                Path(save_path),
                x_test=x_test_1d,
                y_true_test=y_true_1d,
                runs=all_runs_plot_data,
                title=title,
                function_name=function_name,
                noise_train=noise_train,
                noise_test=noise_test,
            )
            print(f"Saved predictions for replay: {npz_path}")
        except Exception as _npz_e:
            print(f"Saving predictions.npz failed: {_npz_e}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if run_models in [None, "pfn"] else None
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, "gp"] else None

    if save_path is not None:
        if run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if run_models is not None:
                file_prefix = run_models
            else:
                file_prefix = "gpVpfn"

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
        except Exception:
            pass

        if trainer_info and GPTrainer_info and run_models in [None, "gp"]:
            try:
                trainer_analysis_dir = Path(save_path) / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)

                trainer_info_by_run = {
                    f"run_{entry.get('run', i + 1)}": entry
                    for i, entry in enumerate(GPTrainer_info)
                }
                trainer_info_data = {
                    "title": title,
                    "num_runs": num_runs,
                    "num_inits_per_run": num_inits,
                    "trainer_info": trainer_info_by_run,
                }

                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")

                try:
                    from experimental_utils.plot_trainer_analysis_hyperparams import extract_runs_and_chosen

                    all_inits, chosen_list = extract_runs_and_chosen(trainer_info_data)
                    if chosen_list and "gp_data" in combined_data:
                        gp_section = combined_data.get("gp_data") or {}
                        gp_metrics_list = gp_section.get("metrics")
                        if isinstance(gp_metrics_list, list):
                            for i_run, (metric_rec, chosen_rec) in enumerate(zip(gp_metrics_list, chosen_list)):
                                loss_val = chosen_rec.get("loss")
                                if loss_val is not None:
                                    metric_rec["loss_final"] = float(loss_val)
                        combined_path = out_dir / f"{file_prefix}_{title}.json"
                        try:
                            combined_path.write_text(json.dumps(combined_data, indent=2))
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Augmenting gp_data with chosen final loss failed: {e}")
                try:
                    from experimental_utils.plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data

                    plot_trainer_analysis_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except Exception as plot_e:
                    print(f"Trainer analysis plotting skipped: {plot_e}")
                try:
                    from experimental_utils.plot_epoch_metrics import plot_iter_metrics_from_data

                    plot_iter_metrics_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except ValueError:
                    pass
                except Exception as e:
                    print(f"Epoch metrics plotting skipped: {e}")

            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback

                traceback.print_exc()
    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("=" * 60)
    print(
        f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of inits: {num_inits}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\tX_scaling_type: {X_scaling_type}\n\ty_standardize: {standardize_y}"
    )
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\truns: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    functions_to_run = tuple(REGRESSION_1D_FUNCTIONS.keys())
    for fn_name in functions_to_run:
        print("\n" + "#" * 70)
        print(f"Running A22 GP vs PFN: {fn_name}")
        print("#" * 70)
        regression_1D_GPvsPFN(
            function_name=fn_name,
            num_runs=10,
            train_size=20,
            dimensions=1,
            save_path=f"./results/A22_regression_1D/{fn_name}",
        )


if __name__ == "__main__":
    functions_to_run = tuple(REGRESSION_1D_FUNCTIONS.keys())
    for fn_name in functions_to_run:
        print("\n" + "#" * 70)
        print(f"Running A22 GP vs PFN: {fn_name}")
        print("#" * 70)
        regression_1D_GPvsPFN(
            function_name=fn_name,
            num_runs=10,
            train_size=100,
            dimensions=1,
            save_path=f"./results/A22_regression_1D/{fn_name}",
        )
