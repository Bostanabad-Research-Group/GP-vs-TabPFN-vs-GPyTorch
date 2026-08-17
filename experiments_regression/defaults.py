import gpplus
import torch
from gpplus.training.callbacks import (
    FinalParameterStorageCallback,
    IterationParameterCallback,
    EpochParameterCallback,
    LBFGSInnerMetricsCallbackV3,
)
from gpplus.training.optimizers import LBFGSScipy

WARNINGS_IGNORE = True

X_STANDARDIZE_METHOD = 2  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
STANDARDIZE_X = True
STANDARDIZE_Y = True

SF_kernel = None
SF_mean = None
SF_likelihood = None

NUM_RUNS = 10 # Number of training sets to run for each problem

TRAINER_LR = None # Using LBFGS
TRAINER_MIN_EPOCHS = 0  # Do not consider early stopping until at least this many epochs
# One outer epoch for LBFGS/LBFGSScipy: a single optimizer.step() runs up to max_iter
# inner iterations per initialization/restart (GPTrainer also forces 1 for LBFGS-like).
TRAINER_NUM_EPOCHS = 1
TRAINER_NUM_INITS = 16 # Number of initializations to run for each training set
TRAINER_CONVERGENCE_PATIENCE = 10
TRAINER_CHOLESKY_JITTER = 1e-6
TRAINER_MIN_LOSS_CHANGE = 1e-7
TRAINER_OPTIMIZER_CLASS = gpplus.training.optimizers.LBFGSScipy
# Match LBFGSScipy.__init__ defaults (convergence via tolerance_grad / tolerance_change).
TRAINER_OPTIMIZER_KWARGS = {
    "max_iter": 2000,
    "max_eval": 5000,
    "tolerance_grad": 1e-5,
    "tolerance_change": 1e-9,
    "history_size": 10,
}

TRAINER_INITIALIZER_CLASS = gpplus.training.parameter_initializer.DefaultParameterInitializer
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32
NOISE_TYPE = "gaussian" # "gaussian" or "uniform" or "student_t"


PREPROCESS_PFN = False

TRAINER_INFO = True # Will log detailed trainer information if 'TRAINER_INFO = True'
PLOT_METRICS = True # Will plot GPvsPFN RRMSE and NIS results if 'PLOT_METRICS = True'
TRAINER_LOG_LBFGS_INNER = True # Only will log if 'TRAINER_INFO = True'

SEED = 42
SEED_TRAINER = None

def get_default_gp_callbacks(
    optimizer_class,
    callback_save_path: str | None = None,
    log_lbfgs_inner: bool = True,
    lbfgs_inner_extra_metrics: list | None = None,
):
    """
    Default GP callbacks for final experiments (A#).

    This mirrors the original train_eval3 defaults, but is now configured explicitly
    from experiments_final instead of being hard-wired into train_eval3.
    """
    callbacks: list = [FinalParameterStorageCallback(save_file=None, verbose=False)]

    if optimizer_class is not None:
        # Only set epoch callback save path when logging to disk is desired.
        if callback_save_path is not None:
            epoch_save_file = f"{callback_save_path}/epoch_parameters.json"
        else:
            epoch_save_file = None

        # LBFGSScipy: per-iteration parameter logging + optional inner metrics
        if optimizer_class is LBFGSScipy or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy)
        ):
            callbacks.append(
                IterationParameterCallback(
                    save_file=None,
                    verbose=False,
                    save_every_n_iterations=20,
                )
            )
            if log_lbfgs_inner:
                callbacks.append(
                    LBFGSInnerMetricsCallbackV3(
                        log_record_every_n_iters=10,
                        log_metrics_every_n_iters=10,
                        log_nll=True,
                        log_nis=True,
                        log_loo=True,
                        log_kf=True,
                        log_residual_mse=True,
                        extra_metrics=lbfgs_inner_extra_metrics or [],
                    )
                )
        # Adam: per-epoch parameter logging
        elif optimizer_class is torch.optim.Adam or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.Adam)
        ):
            callbacks.append(
                EpochParameterCallback(
                    save_file=epoch_save_file,
                    verbose=False,
                    save_every_n_epochs=20,
                )
            )

    return callbacks

def MF_kernel(
    cont_cols=None,
    cat_cols=None,
    source_cols=None,
    cont_kernel=None,
    cat_kernel=None,
    source_kernel=None,
    cat_encoder=None,
    source_encoder=None,
    z_dim=2,
    fix_lengthscale_cat=False,
    fix_lengthscale_source=False,
    **kwargs
):
    """Factory function that creates and wraps the MF kernel."""
    return gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            source_cols=source_cols,
            cont_kernel=cont_kernel,
            cat_kernel=cat_kernel,
            source_kernel=source_kernel,
            cat_encoder=cat_encoder,
            source_encoder=source_encoder,
            z_dim=z_dim,
            fix_lengthscale_cat=fix_lengthscale_cat,
            fix_lengthscale_source=fix_lengthscale_source,
            **kwargs
        )
    )

# For current experiments, we are not using the following MF methods
# MF_mean = gpplus.means.MultiMean
# MF_likelihood = gpplus.likelihoods.MultiLikelihood
# MF_STANDARDIZATION_METHOD = 2 # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently
