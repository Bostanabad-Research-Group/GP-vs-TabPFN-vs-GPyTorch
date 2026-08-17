import torch

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
TRAINER_LR = 0.1
TRAINER_NUM_EPOCHS = 10000
TRAINER_NUM_INITS = 16
TRAINER_CONVERGENCE_PATIENCE = 20
# Absolute MLL improvement needed to count as progress (resets patience).
# Adam crawls slowly; 1e-3 stops much earlier than 1e-4 (~hundreds vs ~thousands of epochs).
TRAINER_MIN_LOSS_CHANGE = 1e-3
# TRAINER_OPTIMIZER_CLASS = torch.optim.LBFGS
TRAINER_OPTIMIZER_CLASS = torch.optim.Adam
TRAINER_INITIALIZER_CLASS = None  # Not used in gpytorch scripts
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
# Parallel multi-init workers on CPU (None = cpu_count - 2, matching gpplus).
TRAINER_N_JOBS = None

SEED = 42
SEED_TRAINER = None

DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32

# LBFGS-only (used only when TRAINER_OPTIMIZER_CLASS is LBFGS).
# Use torch-like small max_iter + many outer epochs (orig gpytorch best_epoch ~6-8).
# max_iter=2000 *per outer step* with epochs=100 is extremely slow (and worse under parallel).
LBFGS_MAX_ITER = 20
LBFGS_MAX_EVAL = 25
LBFGS_TOLERANCE_GRAD = 1e-5
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE = 10
LBFGS_LR = 1  # Line-search step size for torch.optim.LBFGS

# Adam optimizer kwargs (lr / epochs come from TRAINER_LR / TRAINER_NUM_EPOCHS above).
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
