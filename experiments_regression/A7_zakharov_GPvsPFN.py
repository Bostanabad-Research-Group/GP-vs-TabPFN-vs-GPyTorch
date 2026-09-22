import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
import matplotlib.pyplot as plt
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from load_experimental_data import generate_zakharov_data
import defaults
from run_metadata import experiment_data_info, pfn_model_info

# import warnings
# warnings.filterwarnings("ignore")
def zakharov_GPvsPFN(num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10,  # total training size is train_size * number of X input dimensions
        dimensions=5,
        x_bounds=[-5, 10],
        num_inits=defaults.TRAINER_NUM_INITS,
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        min_epochs=defaults.TRAINER_MIN_EPOCHS,
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/zakharov',
        title=None,
        standardize_X=defaults.STANDARDIZE_X,
        standardize_y=defaults.STANDARDIZE_Y,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        standardize_y_log_scale=False,
        log_y_epsilon=1e-8,
        log_y_C=None,
        noise_train=0.0,
        noise_test=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = defaults.DTYPE_GP,
        pfn_dtype = defaults.DTYPE_PFN,
        trainer_info=True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
        log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
        preprocess_pfn=defaults.PREPROCESS_PFN,
        plot_train_y_hist=True,
        hist_bins=30,
        kernel_type=None,  # None=default SF kernel, 'Gaussian', 'PowerExponential', 'Matern'
        loss_type="nll",  # "nll" (default MLL) or "loo" (leave-one-out pseudo likelihood)
    ):

    if run_models == 'pfn':
        num_inits = 0

    if title is None:
        title = f"Zakharov_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else:
        title = f"Zakharov_{title}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = defaults.make_tabpfn_regressor(amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None

    # Generate data
    set_seed(seed)
    
    train_per_run = train_size * dimensions  # train_size * dimensions for Zakharov
    num_runs_gen = max(num_runs, 20)
    n_train_generate = num_runs_gen * train_per_run
    total_samples = num_test + n_train_generate
    print(
        f"Generating {total_samples} unique Sobol samples for {dimensions}D Zakharov function\n\t"
        f"Test samples: {num_test} / Train pool: {n_train_generate} "
        f"({train_per_run} points per run across {num_runs} runs)"
    )

    # Generate train and test data in one call
    X_train_all, y_train_all, X_test_all, y_test_all = generate_zakharov_data(
        n_train=n_train_generate,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed
    )
    if plot_train_y_hist and save_path is not None:
        try:
            plot_dir = Path(plot_save_path)
            plot_dir.mkdir(parents=True, exist_ok=True)
            hist_path = plot_dir / f"y_train_hist_{title}.png"
            y_vals = y_train_all.detach().cpu().reshape(-1).numpy()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(y_vals, bins=hist_bins, alpha=0.85, color="steelblue", edgecolor="black")
            ax.set_title(f"{title}\nZakharov training y histogram")
            ax.set_xlabel("y_train")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved training y histogram: {hist_path}")
        except Exception as e:
            print(f"Saving training y histogram failed: {e}")
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across runs

    torch.manual_seed(seed)
    all_indices = torch.randperm(n_train_generate)
    train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)

    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        run_train_indices = train_indices_2d[i]
        X_train = X_train_all[run_train_indices]
        y_train = y_train_all[run_train_indices]

        # GP preprocessing; PFN uses scaled or raw arrays per preprocess_pfn
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        X_train_raw_for_pfn = X_train.detach().clone()
        X_test_raw_for_pfn = X_test.detach().clone()
        y_train_raw_for_pfn = y_train.detach().clone()
        y_test_raw_for_pfn = y_test.detach().clone()
        # Determine X scaling type
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

        if standardize_y_log_scale:
            # Step 1: shift all y by C = 3 * std(y_train) before the log transform.
            computed_log_y_C = float((3.0 * y_train.std()).item())
            Yscaler = gpplus.utils.LogScaler(epsilon=log_y_epsilon, C=computed_log_y_C)
        else:
            Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)
        log_scale_C = Yscaler.C if standardize_y_log_scale else None

        if standardize_y_log_scale:
            y_train_log_space = torch.log(y_train + log_scale_C)
            print(f"\n--- Run {i+1}/{num_runs} y_train statistics (log-scaled) ---")
            print(f"  LogScaler C: {log_scale_C}")
            print(f"  Log-scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Log-scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Log-scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Log-scaled Max: {y_train_normal.max().item():.6f}")
            print(f"  Log-space min (before standardization): {y_train_log_space.min().item():.6f}")
            print(f"  Log-space max (before standardization): {y_train_log_space.max().item():.6f}")

        if kernel_type == "PowerExponential":
            kernel_mod = gpplus.kernels.LogScaleKernel(
                gpplus.kernels.PowerExponentialKernel(ard_num_dims=dimensions)
            )
        elif kernel_type == "Gaussian":
            kernel_mod = gpplus.kernels.LogScaleKernel(
                gpplus.kernels.GaussianKernel(ard_num_dims=dimensions)
            )
        elif kernel_type == "Matern":
            kernel_mod = gpplus.kernels.LogScaleKernel(
                gpplus.kernels.MaternKernel(nu=2.5, ard_num_dims=dimensions)
            )
        else:
            kernel_mod = defaults.SF_kernel

        # =============================================================================
        # GP Section 
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} GP Training ---")
            
            # Create GP model (default kernel like SF wing)
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kernel_mod,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_runs - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                if standardize_y_log_scale:
                    print(f"LogScaler C: {log_scale_C}")
                print(model)

            # Create trainer
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
                y_train_mean=y_train_mean,
                y_train_std=y_train_std,
                standardize_y_log_scale=standardize_y_log_scale,
                log_scale_C=log_scale_C,
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
                loss_type=loss_type,
            )
            GPPlus_metrics.append(gp_metric)
            
            # Accumulate trainer info if available
            if gp_trainer_info:
                # Add fold information to trainer log
                gp_trainer_info["fold"] = i + 1
                gp_trainer_info["metrics"] = gp_metric  # Include metrics for this fold
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nGP Results (Fold {i+1}/{num_runs})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_models in [None, 'pfn']:
            print(f"\n--- {title} TabPFN Training ---")
            if preprocess_pfn:
                pfn_X_train, pfn_X_test = X_train, X_test
                pfn_y_train = y_train_normal if standardize_y else y_train
                pfn_y_test = y_test
                pfn_y_mean, pfn_y_std = y_train_mean, y_train_std
                pfn_log_scale = standardize_y_log_scale
                pfn_log_scale_C = log_scale_C
            else:
                pfn_X_train = X_train_raw_for_pfn
                pfn_X_test = X_test_raw_for_pfn
                pfn_y_train = y_train_raw_for_pfn
                pfn_y_test = y_test_raw_for_pfn
                pfn_y_mean, pfn_y_std = None, None
                pfn_log_scale = False
                pfn_log_scale_C = None

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
                standardize_y_log_scale=pfn_log_scale,
                log_scale_C=pfn_log_scale_C,
                record_y_train_mean=y_train_mean,
                record_y_train_std=y_train_std,
            )
            TabPFN_metrics.append(tabpfn_metric)

            # Print results for this run
            print(f"\nTabPFN Results (Run {i+1}/{num_runs})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
        
        # Collect model info from first fold
        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
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
                standardize_y_log_scale=standardize_y_log_scale,
                preprocess_pfn=preprocess_pfn,
                pfn_dtype=pfn_dtype,
                log_y_epsilon=log_y_epsilon,
                log_y_C=log_y_C,
                log_y_C_rule="C = 3 * std(y_train)",
                log_scale_C=log_scale_C,
                x_bounds=x_bounds,
                dimensions=dimensions,
                kernel_type=kernel_type,
                loss_type=loss_type,
            )
            if run_models in [None, 'gp']:
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
            if run_models in [None, 'pfn']:
                tabpfn_model_info = pfn_model_info(
                    regressor, experiment_data=shared_experiment_info
                )
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if run_models in [None, 'pfn'] else None
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, 'gp'] else None
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        if run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Determine file prefix based on run_models
            if run_models is not None:
                file_prefix = run_models
            else:
                file_prefix = "gpVpfn"
            
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {}
            if run_models in [None, 'gp']:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info
                }
            if run_models in [None, 'pfn']:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info
                }
            # Append defaults.py source at end of JSON for reproducibility
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
        
        # Save trainer info if trainer_info is enabled
        if trainer_info and GPTrainer_info and run_models in [None, 'gp']:
            try:
                # Create trainer_analysis directory (same level as plots)
                trainer_analysis_dir = Path(save_path) / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
                
                # Save raw trainer info (just the parameter info)
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
                
                # Save trainer info JSON (always use "gp" prefix for GP trainer info)
                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                try:
                    from experimental_utils.plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data
                    plot_trainer_analysis_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except Exception as plot_e:
                    print(f"Trainer analysis plotting skipped: {plot_e}")
                try:
                    from experimental_utils.plot_epoch_metrics import plot_iter_metrics_from_data
                    plot_iter_metrics_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except ValueError:
                    pass  # no epoch_metrics in data
                except Exception as e:
                    print(f"Epoch metrics plotting skipped: {e}")
                
            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of inits: {num_inits}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\tX_scaling_type: {X_scaling_type}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\truns: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    from A7_zakharov_GPvsPFN_gpytorch import zakharov_GPvsPFN as zakharov_GPvsPFN_gpytorch

    # Warm-up run (no save artifacts).
    zakharov_GPvsPFN(
        title="WARMUP",
        num_runs=1,
        train_size=2,
        dimensions=20,
        num_test=500,
        num_inits=1,
        num_epochs=1,
        noise_train=0.002,
        noise_test=0.002,
        standardize_y_log_scale=True,
        save_path=None,
        run_models="gp",
    )

    for train_size in (5, 20):
        for noise in (0.002, 0.08):
            common = dict(
                num_runs=defaults.NUM_RUNS,
                train_size=train_size,
                dimensions=20,
                num_inits=defaults.TRAINER_NUM_INITS,
                noise_train=noise,
                noise_test=noise,
            )

            # GP+ with log-scaled y (LogScaler)
            # zakharov_GPvsPFN(
            #     **common,
            #     standardize_y_log_scale=True,
            #     save_path="./results/zakharov_log_scale",
            #     run_models="gp",
            # )

            # # GP+ with Power Exponential kernel (GP+ PE in IDETC tables)
            # zakharov_GPvsPFN(
            #     **common,
            #     standardize_y_log_scale=True,
            #     kernel_type="PowerExponential",
            #     save_path="./results/zakharov_power_exponential_log_scale",
            #     run_models="gp",
            # )

            # # GP+ trained with leave-one-out pseudo likelihood
            # zakharov_GPvsPFN(
            #     **common,
            #     standardize_y_log_scale=True,
            #     loss_type="loo",
            #     save_path="./results/zakharov_loo_log_scale",
            #     run_models="gp",
            # )

            # Pure GPyTorch baseline (Gaussian kernel) with log-scaled y (LogScaler)
            zakharov_GPvsPFN_gpytorch(
                num_runs=common["num_runs"],
                train_size=common["train_size"],
                dimensions=common["dimensions"],
                num_inits=common["num_inits"],
                noise_train=noise,
                noise_test=noise,
                standardize_y_log_scale=True,
                save_path="./results/zakharov_gpytorch_log_scale",
            )

    # zakharov_GPvsPFN(num_runs=1, train_size=20, dimensions=20, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=40, dimensions=20, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=80, dimensions=20, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=10, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=20, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=40, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')
    # zakharov_GPvsPFN(num_runs=1, train_size=80, dimensions=40, num_inits=4, save_path='./results/zakharov/temp')












