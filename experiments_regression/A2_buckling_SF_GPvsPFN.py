import torch
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN
from gpplus.likelihoods import LogGaussianLikelihood
from gpytorch.priors import NormalPrior
# from gpytorch.means import ZeroMean
from load_experimental_data import generate_mf_buckling_data_with_folds
import defaults
from run_metadata import experiment_data_info, pfn_model_info

# import warnings
# warnings.filterwarnings("ignore")
def buckling_SF_GPvsPFN(num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions (4)
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
        save_path='./results/A2_buckling',
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
        gp_dtype=defaults.DTYPE_GP,
        pfn_dtype=defaults.DTYPE_PFN,
        trainer_info=True,
        MF_kernel=True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
        log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
        preprocess_pfn=defaults.PREPROCESS_PFN,
        warnings_ignore=defaults.WARNINGS_IGNORE,
        single_dataset=False,
        # If True: one stratified train fold (same X_train, y_train) for every run; run_seed still varies.
        # If False: legacy — one fold per run from a larger total train pool (max(num_runs, 20) folds).
        kernel_type=None,  # None=default MF kernel, 'Gaussian', 'PowerExponential', 'Matern'
        loss_type="nll",  # "nll" (default MLL) or "loo" (leave-one-out pseudo likelihood)
    ):
    if warnings_ignore:
        import warnings
        warnings.filterwarnings("ignore")

    if run_models == 'pfn':
        num_inits = 0
    
    if title is None:
        title = f"buckling_SF_{train_size}Dn_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else: 
        title = f"buckling_SF_{title}_{train_size}Dn_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    
    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = defaults.make_tabpfn_regressor(amp_device, seed) if run_models in [None, 'pfn'] else None
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None
    
    # Calculate total samples (4D problem). Folds are built inside generate_mf_buckling_data_with_folds.
    train_per_run = train_size * 4
    if single_dataset:
        num_runs_gen = 1
        total_train = train_per_run
        total_samples = num_test + total_train
        print(
            f"Generating {total_samples} unique Sobol samples\n\t"
            f"Test samples: {num_test} / Train samples: {total_train} "
            f"(single_dataset=True: same stratified train fold for all {num_runs} runs)"
        )
    else:
        num_runs_gen = max(num_runs, 20)
        total_train = num_runs_gen * train_per_run
        total_samples = num_test + total_train
        print(
            f"Generating {total_samples} unique Sobol samples\n\t"
            f"Test samples: {num_test} / Train samples: {total_train} "
            f"(single_dataset=False: disjoint folds, {train_per_run} points per run)"
        )

    X_train_runs, y_train_runs, X_test_all, y_test_all = generate_mf_buckling_data_with_folds(
        train_samples_per_source=[total_train, 0],
        test_samples_per_source=[num_test, 0],
        num_runs=num_runs_gen,
        train_noise=[noise_train, 0.0],
        test_noise=[noise_test, 0.0],
        noise_type=noise_type,
        seed=seed,
    )

    # Drop the 5th (source) column since SF uses only s0
    for i in range(len(X_train_runs)):
        if X_train_runs[i].shape[1] == 5:
            X_train_runs[i] = X_train_runs[i][:, :4]
    if X_test_all.shape[1] == 5:
        X_test_all = X_test_all[:, :4]
    
    # Combine all train runs for TabPFN
    X_train_all = torch.cat(X_train_runs, dim=0)
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_test_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # X_enc_train_all, _, _, _ = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    print('cont_cols:', cont_cols)
    print('cat_cols:', cat_cols)
    print('source_cols:', source_cols)
    # Encode each run individually for GP training
    X_train_runs_enc = []
    for run_data in X_train_runs:
        run_enc, _, _, _ = encode_qual_data(run_data, qual_dict=qual_dict, source_col=None)
        X_train_runs_enc.append(run_enc)
    
    print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across runs

    # Debug: Print categorical distributions for first few folds
    print(f"\n{'='*20} PRE-STRATIFIED RUNS VERIFICATION {'='*20}")
    for run in range(min(3, len(X_train_runs))):
        run_data = X_train_runs[run]
        
        print(f"\nRun {run + 1} categorical distributions:")
        # Check I distribution (column 3)
        for i in range(3):
            count = (run_data[:, 3] == i).sum().item()
            print(f"  I={i}: {count} samples")
        
        # Check E distribution (column 1) 
        for i in range(2):
            count = (run_data[:, 1] == i).sum().item()
            print(f"  E={i}: {count} samples")
            
        # Check K distribution (column 2)
        for i in range(4):
            count = (run_data[:, 2] == i).sum().item()
            print(f"  K={i}: {count} samples")
        
        # Print actual samples (E, K, I) for verification
        E_col, K_col, I_col = 1, 2, 3
        samples_eki = [
            (int(run_data[j, E_col].item()), int(run_data[j, K_col].item()), int(run_data[j, I_col].item()))
            for j in range(run_data.shape[0])
        ]
        print(f"  Samples (E, K, I): {samples_eki}")
    
    print(f"{'='*60}")
        
    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        # Use pre-generated fold (same fold every run if single_dataset)
        run_idx = 0 if single_dataset else i
        X_train = X_train_runs_enc[run_idx]
        y_train = y_train_runs[run_idx]
        
        # Prepare data (standardization) - ALWAYS DO THIS for both GP and PFN
        # Convert to torch dtype and optionally standardize X
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        # TabPFN: same train/test rows as GP but raw 4-column inputs (no one-hot encoding)
        X_train_raw_for_pfn = X_train_runs[run_idx].detach().clone().to(dtype=gp_dtype)
        X_test_raw_for_pfn = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train_raw_for_pfn = y_train.detach().clone()
        y_test_raw_for_pfn = y_test.detach().clone()
        # Determine X scaling type
        X_scaling_type = "None"
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

        # Normalize the GP data
        # Print unscaled y_train statistics
        print(f"\n--- Run {i+1}/{num_runs} y_train statistics (unscaled) ---")
        print(f"  Mean: {y_train.mean().item():.6f}")
        print(f"  Std: {y_train.std().item():.6f}")
        print(f"  Median: {y_train.median().item():.6f}")
        print(f"  Min: {y_train.min().item():.6f}")
        print(f"  Max: {y_train.max().item():.6f}")
        
        if standardize_y_log_scale:
            computed_log_y_C = float((3.0 * y_train.std()).item())
            Yscaler = gpplus.utils.LogScaler(epsilon=log_y_epsilon, C=computed_log_y_C)
        else:
            Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)
        # Get C from LogScaler if using log scale to ensure it matches
        log_scale_C = Yscaler.C if standardize_y_log_scale else None
        
        # Print scaled y_train statistics
        if standardize_y_log_scale:
            # Compute log-space values before standardization for display
            y_train_log_space = torch.log(y_train + log_scale_C)
            print(f"\n--- Run {i+1}/{num_runs} y_train statistics (log-scaled) ---")
            print(f"  LogScaler C: {log_scale_C}")
            print(f"  Log-scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Log-scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Log-scaled Median: {y_train_normal.median().item():.6f}")
            print(f"  Log-scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Log-scaled Max: {y_train_normal.max().item():.6f}")
            print(f"  Log-space mean (before standardization): {y_train_mean.item() if y_train_mean.numel() == 1 else y_train_mean.squeeze().item():.6f}")
            print(f"  Log-space std (before standardization): {y_train_std.item() if y_train_std.numel() == 1 else y_train_std.squeeze().item():.6f}")
            print(f"  Log-space median (before standardization): {y_train.median().item() if y_train.median().numel() == 1 else y_train.median().squeeze().item():.6f}")
            print(f"  Log-space min (before standardization): {y_train_log_space.min().item():.6f}")
            print(f"  Log-space max (before standardization): {y_train_log_space.max().item():.6f}")
        else:
            print(f"\n--- Run {i+1}/{num_runs} y_train statistics (standard-scaled) ---")
            print(f"  Scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Scaled Median: {y_train_normal.median().item():.6f}")
            print(f"  Scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Scaled Max: {y_train_normal.max().item():.6f}")
        
        # =============================================================================
        # GP Section 
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} GP Training ---")

            input_dim = X_train.shape[1]
            if kernel_type == "PowerExponential":
                kernel = gpplus.kernels.LogScaleKernel(
                    gpplus.kernels.PowerExponentialKernel(ard_num_dims=input_dim)
                )
            elif kernel_type == "Gaussian":
                kernel = gpplus.kernels.LogScaleKernel(
                    gpplus.kernels.GaussianKernel(ard_num_dims=input_dim)
                )
            # elif kernel_type == "Matern":
            #     kernel = gpplus.kernels.LogScaleKernel(
            #         gpplus.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
            #     )
            elif MF_kernel:
            #     kernel = defaults.MF_kernel(
            #         cont_cols=cont_cols,
            #         cat_cols=cat_cols,
            #         source_cols=source_cols,
            #     )
                grouped_cat = [x for sub in cat_cols for x in sub]
                kernel = gpplus.kernels.LogScaleKernel(
                    gpplus.kernels.MVMFKernel(
                    cont_cols=cont_cols,
                    # cat_cols=cat_cols,
                    cat_cols=grouped_cat,
                    )  
                )
            else:
                kernel = defaults.SF_kernel

            # Prior on log-scale noise (raw_noise in [-7, -1], same scale as init)
            # buckling_likelihood = LogGaussianLikelihood(
            #     noise_prior=NormalPrior(loc=-3.5, scale=0.5),  # mass in [-7, -1]; -2.5 for 0.05 noise, -4.5 for 0.005
            # )
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kernel,
                mean_module=defaults.SF_mean,
                # mean_module=ZeroMean(),
                # likelihood=buckling_likelihood,
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
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
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
                # Add run information to trainer log
                gp_trainer_info["run"] = i + 1
                gp_trainer_info["metrics"] = gp_metric  # Include metrics for this run
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nGP Results (run {i+1}/{num_runs})")
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
                y_train_mean=pfn_y_mean,
                y_train_std=pfn_y_std,
                standardize_y_log_scale=pfn_log_scale,
                log_scale_C=pfn_log_scale_C,
                source_cols=source_cols,
                record_y_train_mean=y_train_mean if standardize_y else None,
                record_y_train_std=y_train_std if standardize_y else None,
            )
            TabPFN_metrics.append(tabpfn_metric)

            # Print results for this run
            print(f"\nTabPFN Results (run {i+1}/{num_runs})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
        
        # Collect model info from first run
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
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
                train_samples=int(train_per_run),
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
                    "loss_type": loss_type,
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
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, 'gp'] else None
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if run_models in [None,'pfn'] else None
    
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
                    "pfn_model_info": tabpfn_model_info,
                }
            # Append defaults.py source at end of JSON for reproducibility
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
        
        # Save trainer info if trainer_info is enabled
        if trainer_info and GPTrainer_info:
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
                
                # Save trainer info JSON
                trainer_info_file = trainer_analysis_dir / f"gpVpfn_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
                # Generate hyperparameter plots in trainer_analysis/plots
                try:
                    from experimental_utils.plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data
                    plots_dir = trainer_analysis_dir / "plots"
                    plot_trainer_analysis_from_data(trainer_info_data, plots_dir)
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
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of runs: {num_inits}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\truns: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN as buckling_SF_GPvsPFN_gpytorch

    # Warm-up run (no save artifacts).
    buckling_SF_GPvsPFN(
        title="WARMUP",
        num_runs=1,
        train_size=2,
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
                num_test=5000,
                num_inits=defaults.TRAINER_NUM_INITS,
                num_epochs=defaults.TRAINER_NUM_EPOCHS,
                noise_train=noise,
                noise_test=noise,
            )

            # GP+ with log-scaled y (LogScaler): pass standardize_y_log_scale=True
            # buckling_SF_GPvsPFN(
            #     **common,
            #     standardize_y_log_scale=True,
            #     save_path="./results/A2_buckling_log_scale",
            #     run_models="gp",
            # )

            # GP+ with Power Exponential kernel (GP+ PE in IDETC tables)
            # buckling_SF_GPvsPFN(
            #     **common,
            #     standardize_y_log_scale=True,
            #     kernel_type="PowerExponential",
            #     save_path="./results/A2_buckling_power_exponential_log_scale",
            #     run_models="gp",
            # )

            # GP+ trained with leave-one-out pseudo likelihood
            buckling_SF_GPvsPFN(
                **common,
                standardize_y_log_scale=True,
                loss_type="loo",
                save_path="./results/A2_buckling_loo_log_scale",
                run_models="gp",
            )

            # Pure GPyTorch baseline
            # buckling_SF_GPvsPFN_gpytorch(
            #     num_runs=common["num_runs"],
            #     train_size=common["train_size"],
            #     num_inits=common["num_inits"],
            #     num_epochs=common["num_epochs"],
            #     noise_train=noise,
            #     noise_test=noise,
            #     save_path="./results/A2_buckling_gpytorch",
            # )



