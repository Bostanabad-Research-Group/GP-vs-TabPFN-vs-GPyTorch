import torch
import json
import numpy as np
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
import gpytorch
import linear_operator
from linear_operator.utils.errors import NotPSDError, NanError
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics, compute_metrics
from gpplus.utils import set_seed, train_eval_PFN
from gpplus.training.eval import evaluate_gp_model
# from gpplus.tabpfn.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from load_experimental_data import generate_mf_buckling_data_with_folds
import defaults_gpytorch as defaults
from gpytorch_train_eval import train_eval_gp_gpytorch_default

# import warnings
# warnings.filterwarnings("ignore")

class ExactGPModel(gpytorch.models.ExactGP):
    """Pure gpytorch ExactGP model for regression."""
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def buckling_SF_GPvsPFN(num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10, # total training size is train_size * number of X input dimensions (4)
        num_inits=defaults.TRAINER_NUM_INITS, 
        num_epochs=defaults.TRAINER_NUM_EPOCHS, 
        lr=defaults.TRAINER_LR, 
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/buckling',
        title=None,
        standardize_X=True,
        standardize_y=True,
        x_standardize_method=2,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        standardize_y_log_scale=False,
        log_y_epsilon=1e-8,
        log_y_C=None,  # kept for API parity with A2 GP+; C is always 3 * std(y_train)
        noise_train=0.0,
        noise_test=0.0,
        noise_type='gaussian',
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        gp_dtype = getattr(defaults, 'DTYPE_GP', torch.float64),
        pfn_dtype = getattr(defaults, 'DTYPE_PFN', torch.float32),
    ):
    if title is None:
        title = f"buckling_gpytorch_{train_size}D_{num_epochs}epochs_{num_inits}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"buckling_gpytorch{title}_{train_size}D_{num_epochs}epochs_{num_inits}runs_{lr}_noiseTest{noise_test}_noiseTrain{noise_train}"
    
    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    # regressor = VanillaDirectTabPFNRegressor(device=amp_device)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None
    
    # Match GP+ pool: max(num_runs, 20) stratified folds (4D buckling)
    train_per_run = train_size * 4
    num_runs_gen = max(num_runs, 20)
    total_train = num_runs_gen * train_per_run
    total_samples = num_test + total_train

    # Generate all unique Sobol samples at once for single-fidelity (use only s0)
    print(
        f"Generating {total_samples} unique Sobol samples\n\t"
        f"Test samples: {num_test} / Train pool: {total_train} "
        f"(disjoint folds, {train_per_run} points per run)"
    )
    X_train_folds, y_train_folds, X_test_all, y_test_all = generate_mf_buckling_data_with_folds(
        train_samples_per_source=[total_train, 0],
        test_samples_per_source=[num_test, 0],
        num_runs=num_runs_gen,
        train_noise=[noise_train, 0.0],
        test_noise=[noise_test, 0.0],
        noise_type=noise_type,
        seed=seed,
    )
    # Drop the 5th (source) column since SF uses only s0
    for i in range(len(X_train_folds)):
        if X_train_folds[i].shape[1] == 5:
            X_train_folds[i] = X_train_folds[i][:, :4]
    if X_test_all.shape[1] == 5:
        X_test_all = X_test_all[:, :4]
    
    # Combine all train folds for TabPFN
    X_train_all = torch.cat(X_train_folds, dim=0)
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison (GPyTorch Defaults)")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    X_enc_test_all, cont_cols, cat_cols, source_cols = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # X_enc_train_all, _, _, _ = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    
    # Encode each fold individually for GP training
    X_train_folds_enc = []
    for fold_data in X_train_folds:
        fold_enc, _, _, _ = encode_qual_data(fold_data, qual_dict=qual_dict, source_col=None)
        X_train_folds_enc.append(fold_enc)
    
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []

    # Debug: Print categorical distributions for each fold
    print(f"\n{'='*20} PRE-STRATIFIED FOLDS VERIFICATION {'='*20}")
    for fold in range(min(3, num_runs)):  # Show first 3 folds
        fold_data = X_train_folds[fold]
        
        print(f"\nFold {fold + 1} categorical distributions:")
        # Check I distribution (column 3)
        for i in range(3):
            count = (fold_data[:, 3] == i).sum().item()
            print(f"  I={i}: {count} samples")
        
        # Check E distribution (column 1) 
        for i in range(2):
            count = (fold_data[:, 1] == i).sum().item()
            print(f"  E={i}: {count} samples")
            
        # Check K distribution (column 2)
        for i in range(4):
            count = (fold_data[:, 2] == i).sum().item()
            print(f"  K={i}: {count} samples")
    
    print(f"{'='*60}")
        
    total_start_time = time.time()
    for i in range(num_runs):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_runs}: {fold_seed} {'='*20}")

        # Use pre-generated fold
        X_train = X_train_folds_enc[i]
        y_train = y_train_folds[i]
        
        # =============================================================================
        # GP Section 
        # =============================================================================
        print(f"\n--- {title} GP Training (GPyTorch Defaults) ---")
        
        # Convert to torch dtype and optionally standardize X
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_enc_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
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
        else:
            X_scaling_type = "None"

        # Same y scaling as A2_buckling_SF_GPvsPFN / A7_zakharov: LogScaler with C = 3 * std(y_train).
        if standardize_y_log_scale:
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
            print(f"\n--- Fold {i+1}/{num_runs} y_train statistics (log-scaled) ---")
            print(f"  LogScaler C: {log_scale_C}")
            print(f"  Log-scaled Mean: {y_train_normal.mean().item():.6f}")
            print(f"  Log-scaled Std: {y_train_normal.std().item():.6f}")
            print(f"  Log-scaled Min: {y_train_normal.min().item():.6f}")
            print(f"  Log-scaled Max: {y_train_normal.max().item():.6f}")
            print(f"  Log-space min (before standardization): {y_train_log_space.min().item():.6f}")
            print(f"  Log-space max (before standardization): {y_train_log_space.max().item():.6f}")

        # Create GP model using pure gpytorch components
        # Use gpytorch defaults: GaussianLikelihood, ScaleKernel(RBFKernel), ConstantMean
        input_dim = X_train.shape[-1]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )
        
        # Ensure components use the correct dtype
        likelihood = likelihood.to(dtype=gp_dtype)
        mean_module = mean_module.to(dtype=gp_dtype)
        covar_module = covar_module.to(dtype=gp_dtype)
        
        model = ExactGPModel(
            X_train,
            y_train_normal if standardize_y else y_train,
            likelihood,
            mean_module,
            covar_module
        )
        if (i == 0) or (i == num_runs - 1):
            print(f"X_train: {X_train.shape}")
            print(f"X_test: {X_test.shape}")
            print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
            if standardize_y_log_scale:
                print(f"LogScaler C: {log_scale_C}")
            print(model)

        # Train with gpytorch defaults
        gp_metric, y_pred_gp, output_std_gp = train_eval_gp_gpytorch_default(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            num_inits=num_inits,
            seed=fold_seed,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            convergence_patience=convergence_patience,
            optimizer_class=optimizer_class,
            lr=lr,
            # Inverse matches GP+ train_eval_gp default: exp(mu) - C (median of log-normal).
            standardize_y_log_scale=standardize_y_log_scale,
            log_scale_C=log_scale_C,
            log_y_point_inverse="median",
        )
        GPPlus_metrics.append(gp_metric)
        if log_scale_C is not None:
            gp_metric["log_scale_C"] = float(log_scale_C)

        print(f"\nGP Results (Fold {i+1}/{num_runs})")
        # Check if this fold had failures
        if gp_metric.get("all_runs_failed", False):
            print(f"  WARNING: All training runs failed for this fold!")
        elif gp_metric.get("all_metrics_nan", False):
            print(f"  WARNING: All metrics are NaN for this fold!")
        elif "evaluation_error" in gp_metric:
            print(f"  WARNING: Evaluation error: {gp_metric['evaluation_error']}")
        
        for k, v in gp_metric.items():
            if k not in ["evaluation_error", "all_runs_failed", "all_metrics_nan"]:
                if isinstance(v, (int, float)):
                    if np.isnan(v):
                        print(f"  {k}: NaN")
                    else:
                        print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        # print(f"\n--- {title} TabPFN Training ---")
        
        # tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
        #     X_train,
        #     X_test,
        #     y_train_normal if standardize_y else y_train,
        #     y_test,
        #     amp_device=amp_device,
        #     amp_dtype=pfn_dtype,
        #     regressor=regressor,
        #     source_cols=source_cols,
        #     y_train_mean=y_train_mean if standardize_y else None,
        #     y_train_std=y_train_std if standardize_y else None,
        # )
        # TabPFN_metrics.append(tabpfn_metric)

        # # Print results for this fold
        # print(f"\nTabPFN Results (Fold {i+1}/{num_runs})")
        # for k, v in tabpfn_metric.items():
        #     print(f"  {k}: {v:.4f}")
        
        # Collect model info from first fold
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }
            
            gp_model_info = {
                "model_str": str(model),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": int(train_per_run),
                "test_samples": num_test,
                "y_train_mean": float(y_train_mean.item()),
                "y_train_std": float(y_train_std.item()),
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "standardize_y_log_scale": standardize_y_log_scale,
                "log_scale_C": float(log_scale_C) if log_scale_C is not None else None,
                "log_y_C_rule": "C = 3 * std(y_train)" if standardize_y_log_scale else None,
                "log_y_point_inverse": "median" if standardize_y_log_scale else None,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_inits": num_inits,
                "lr": lr if lr is not None else (0.001 if optimizer_class == torch.optim.Adam else 0.1),
                "optimizer": optimizer_class.__name__ if optimizer_class is not None else "LBFGS",
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class is not None else None,
                **y_test_stats,
                "num_runs": num_runs,
                "seed": seed,
                "seed_trainer": seed_trainer,
            }
            # tabpfn_model_info = {
            #     "model_path": regressor.model_path,
            #     "fit_mode": regressor.fit_mode,
            #     "device": str(regressor.device_),
            #     "inference_precision": regressor.inference_precision,
            #     "random_state": regressor.random_state,
            #     "use_autocast": regressor.use_autocast_,
            #     "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
            # }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Check for folds with all NaN metrics
    failed_folds = []
    for idx, metric in enumerate(GPPlus_metrics):
        if metric.get("all_runs_failed", False) or metric.get("all_metrics_nan", False):
            failed_folds.append(idx + 1)
    
    if failed_folds:
        print(f"\nWARNING: {len(failed_folds)} fold(s) had complete failures: {failed_folds}")
        print(f"   These folds will have NaN metrics in the summary.")
        print(f"   Consider investigating data scaling, initialization, or increasing jitter.")
    
    # Summaries via analyze_metrics
    # TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title)
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        # plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {
                "gp_data": {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info
                },
                # "tabpfn_data": {
                #     "summary": TabPFN_summary,
                #     "metrics": TabPFN_metrics,
                #     "pfn_model_info": tabpfn_model_info
                # },
            }
            (out_dir / f"gpVpfn_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {num_runs} folds: {time.time() - total_start_time:.2f}s")
    print("="*60)
    opt_name = optimizer_class.__name__ if optimizer_class is not None else "None"
    print(f"Trainer details (GPyTorch): \n\tnumber of epochs: {num_epochs}\n\tlearning rate: {lr}\n\toptimizer: {opt_name}\n\tdevice: {gp_device}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test)} test samples, {len(X_train)} train samples\n\tfolds: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Quick warmup (imports / first train) — do not save.
    print("Warmup (no save)...")
    buckling_SF_GPvsPFN(
        num_runs=1,
        train_size=5,
        num_test=50,
        num_inits=1,
        noise_train=0.002,
        noise_test=0.002,
        standardize_y_log_scale=False,
        save_path=None,
    )

    save_dir = "./results_IDETC/10_runs_gpytorch_corrected/buckling"
    for train_size in (5, 20):
        for noise in (0.002, 0.08):
            buckling_SF_GPvsPFN(
                num_runs=defaults.NUM_RUNS,
                train_size=train_size,
                num_test=5000,
                num_inits=defaults.TRAINER_NUM_INITS,
                noise_train=noise,
                noise_test=noise,
                save_path=save_dir,
            )

    # Match 10_runs_gpytorch_orig/buckling grid, with GP+ LogScaler (C = 3 * std(y_train)).
    save_dir = "./results_logscale_study/10_runs_gpytorch_logscale/buckling"
    for train_size in (5, 20):
        for noise in (0.002, 0.08):
            buckling_SF_GPvsPFN(
                num_runs=defaults.NUM_RUNS,
                train_size=train_size,
                num_test=5000,
                num_inits=defaults.TRAINER_NUM_INITS,
                noise_train=noise,
                noise_test=noise,
                standardize_y_log_scale=True,
                save_path=save_dir,
            )
 
