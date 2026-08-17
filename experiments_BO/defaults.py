import gpplus
import torch

X_STANDARDIZE_METHOD = 2 # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
STANDARDIZE_X = True
STANDARDIZE_Y = True

SF_kernel = None
SF_mean = None
SF_likelihood = None

NUM_RUNS = 1 # Number of training sets to run for each problem
TRAINER_LR = None # Using LBFGS
TRAINER_MIN_EPOCHS = 0  # Do not consider early stopping until at least this many epochs
TRAINER_NUM_EPOCHS = 1
TRAINER_NUM_INITS = 16 # Number of initializations to run for each training set
TRAINER_CONVERGENCE_PATIENCE = 10
# TRAINER_CHOLESKY_JITTER = 0
TRAINER_CHOLESKY_JITTER = 1e-6
TRAINER_MIN_LOSS_CHANGE = 1e-7
# TRAINER_OPTIMIZER_CLASS = torch.optim.Adam
TRAINER_OPTIMIZER_CLASS = gpplus.training.optimizers.LBFGSScipy
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 20, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #1
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 15000, "max_eval": 15000, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #2
TRAINER_OPTIMIZER_KWARGS = {"max_iter": 2000, "max_eval": 2500, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #2
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 20}   # 0 
TRAINER_LOG_LBFGS_INNER = False

TRAINER_INITIALIZER_CLASS = gpplus.training.parameter_initializer.DefaultParameterInitializer
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32
NOISE_TYPE = 'gaussian'

# BO experiment defaults
BO_NUM_RUNS = 1 # Number of training datasets to run BO on for each problem
BO_ACQUISITION = 'EI'  # "EI" or "TS"
GP_OPTIMIZE_AF = True  # If True, optimize GP acquisition function; else use simple sampling
BO_N_AF_OPT = 64 # number of initialization points to optimize from in the GP model's AF (gp only method)
BO_N_AF_SAMPLE = 5000 # number of sample points to evaluate in the AF (simple method)
BO_MAX_ITER = 30
BO_PATIENCE_NO_IMPROVE = 10
BO_GI_PFN = False  # If True, use gradient-informed (GITBO) PFN; else vanilla PFN + Sobol 5k

TRAINER_INFO = False
BO_FULL_INFO = True
BO_TEST_METRICS = True
RUN_MODELS = None  # "gp" for BO; None reserved for future PFN


SEED = 42
SEED_TRAINER = None
