import torch
from gpplus.training.optimizers import LBFGSScipy

# GPyTorch regression settings.
# The paper uses L-BFGS. Set USE_ADAM = True to switch this file to Adam.
# The problem scripts import this module as ``defaults_gpytorch``.

USE_ADAM = False

PREPROCESS_PFN = False

SF_kernel = None
SF_mean = None
SF_likelihood = None

MF_mean = None
MF_likelihood = None


def MF_kernel(*args, **kwargs):
    """Not used by the GPyTorch scripts. They build ScaleKernel(RBFKernel) themselves."""
    return None


MF_STANDARDIZATION_METHOD = 2

NUM_RUNS = 10
TRAINER_NUM_INITS = 16
TRAINER_INITIALIZER_CLASS = None
TRAINER_GP_DEVICE = "cpu"
TRAINER_AMP_DEVICE = "cuda"
TRAINER_N_JOBS = None

SEED = 42
SEED_TRAINER = None

DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32

LBFGS_MAX_ITER = 2000
LBFGS_MAX_EVAL = 5000
LBFGS_TOLERANCE_GRAD = 1e-5
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE = 10
LBFGS_LR = 1

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0

if USE_ADAM:
    TRAINER_LR = 0.1
    TRAINER_NUM_EPOCHS = 10000
    TRAINER_CONVERGENCE_PATIENCE = 20
    TRAINER_MIN_LOSS_CHANGE = 1e-3
    TRAINER_OPTIMIZER_CLASS = torch.optim.Adam
    TRAINER_OPTIMIZER_KWARGS = None
else:
    # One outer step per restart. LBFGSScipy runs up to max_iter inner iterations.
    TRAINER_LR = None
    TRAINER_NUM_EPOCHS = 1
    TRAINER_CONVERGENCE_PATIENCE = None
    TRAINER_MIN_LOSS_CHANGE = 1e-7
    TRAINER_OPTIMIZER_CLASS = LBFGSScipy
    TRAINER_OPTIMIZER_KWARGS = {
        "max_iter": LBFGS_MAX_ITER,
        "max_eval": LBFGS_MAX_EVAL,
        "tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "history_size": LBFGS_HISTORY_SIZE,
    }
