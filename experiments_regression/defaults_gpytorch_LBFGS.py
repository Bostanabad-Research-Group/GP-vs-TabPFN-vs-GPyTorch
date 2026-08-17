import torch
from gpplus.training.optimizers import LBFGSScipy

# Note: These defaults are for gpytorch-based training scripts.
# The shared training helper lives in `gpytorch_train_eval.py`
# as `train_eval_gp_gpytorch_default`.

# TabPFN input preprocessing: True = same X/y scaling as GP; False = raw encoded X and y (SI paper style)
PREPROCESS_PFN = False

SF_kernel = None
SF_mean = None
SF_likelihood = None

# MF settings (not typically used in gpytorch scripts, but kept for consistency)
MF_mean = None  # gpytorch uses ConstantMean
MF_likelihood = None  # gpytorch uses GaussianLikelihood
def MF_kernel(*args, **kwargs):
    """Not used in gpytorch scripts - they use ScaleKernel(RBFKernel)"""
    return None

MF_STANDARDIZATION_METHOD = 2

NUM_RUNS = 10
TRAINER_LR = None  # LBFGSScipy does not use lr
# One outer epoch per init: a single LBFGSScipy.step() with up to max_iter inner iters.
TRAINER_NUM_EPOCHS = 1
TRAINER_NUM_INITS = 16
TRAINER_CONVERGENCE_PATIENCE = None  # unused with 1-epoch LBFGS (scipy tolerances apply)
TRAINER_MIN_LOSS_CHANGE = 1e-7
TRAINER_OPTIMIZER_CLASS = LBFGSScipy
TRAINER_INITIALIZER_CLASS = None  # Not used in gpytorch scripts
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
# Parallel multi-init workers on CPU (None = cpu_count - 2, matching gpplus).
TRAINER_N_JOBS = None

SEED = 42
SEED_TRAINER = None

DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32

# LBFGSScipy kwargs — match gpplus.training.optimizers.LBFGSScipy.__init__ defaults.
LBFGS_MAX_ITER = 2000
LBFGS_MAX_EVAL = 5000
LBFGS_TOLERANCE_GRAD = 1e-5
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE = 10
LBFGS_LR = 1  # unused by LBFGSScipy; kept for torch.optim.LBFGS compatibility

TRAINER_OPTIMIZER_KWARGS = {
    "max_iter": LBFGS_MAX_ITER,
    "max_eval": LBFGS_MAX_EVAL,
    "tolerance_grad": LBFGS_TOLERANCE_GRAD,
    "tolerance_change": LBFGS_TOLERANCE_CHANGE,
    "history_size": LBFGS_HISTORY_SIZE,
}

# Adam optimizer kwargs (lr / epochs come from TRAINER_LR / TRAINER_NUM_EPOCHS above).
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
