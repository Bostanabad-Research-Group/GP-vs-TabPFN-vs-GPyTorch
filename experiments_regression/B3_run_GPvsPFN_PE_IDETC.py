from A1_wing_SF_GPvsPFN import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN
from A4_ackley_GPvsPFN import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN import zakharov_GPvsPFN
from A8_griewank_GPvsPFN import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN import dixon_price_GPvsPFN
import gpplus
import defaults

kernel_type = "PowerExponential"

def set_pe_sf_kernel(dims):
    defaults.SF_kernel = gpplus.kernels.LogScaleKernel(
        gpplus.kernels.PowerExponentialKernel(ard_num_dims=dims)
    )

# folder = r"C:/Users/kianb/Repos/DATA BIN/results_IDETC/IDETC"
folder = "results_IDETC"
date = "10_runs_logging_full_PE"

save_path_wing = f"./{folder}/{date}/wing/"
save_path_buckling = f"./{folder}/{date}/buckling/"
save_path_borehole = f"./{folder}/{date}/borehole/"
save_path_ackley = f"./{folder}/{date}/ackley/"
save_path_ackley_V2 = f"./{folder}/{date}/ackleyV2/"
save_path_rosenbrock = f"./{folder}/{date}/rosenbrock/"
save_path_rastrigin = f"./{folder}/{date}/rastrigin/"
save_path_zakharov = f"./{folder}/{date}/zakharov/"
save_path_griewank = f"./{folder}/{date}/griewank/"
save_path_dixon_price = f"./{folder}/{date}/dixon_price/"

num_test = 5000
# run_models = None
run_models = 'gp'
# run_models = 'pfn'
loss_type = "nll"
# title = "2.0"
# title = 'V3'
title = None
# title = 'gtol_1e-7'
# title = '3'
num_inits = 16

num_runs = 10

set_pe_sf_kernel(10)
wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=5, save_path=None, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models)
# wing_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# buckling_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# borehole_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # # %% Wing ------------------------------------------------------------------------------------------------
set_pe_sf_kernel(10)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # # # %% Buckling ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # # # %% Borehole ------------------------------------------------------------------------------------------------
set_pe_sf_kernel(8)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% 20 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
set_pe_sf_kernel(20)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Rosenbrock -------------------------------------------------------------
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# %% Zakharov -------------------------------------------------------------
zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)

# %% Griewank -------------------------------------------------------------
set_pe_sf_kernel(20)
griewank_GPvsPFN(title="10runs", num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
griewank_GPvsPFN(title="10runs", num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
griewank_GPvsPFN(title="10runs", num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
griewank_GPvsPFN(title="10runs", num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Dixon Price ------------------------------------------------------------- 
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# %% 40 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
set_pe_sf_kernel(40)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Rosenbrock -------------------------------------------------------------
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Zakharov -------------------------------------------------------------
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# %% Dixon Price -------------------------------------------------------------  
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)

# # %% 80 Dx Problems ------------------------------------------------------------------------------------------------
# # %% Ackley -------------------------------------------------------------
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # %% Rosenbrock -------------------------------------------------------------
set_pe_sf_kernel(80)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)
rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type, kernel_type=kernel_type)

# # # %% Zakharov -------------------------------------------------------------
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)

# # # %% Dixon Price -------------------------------------------------------------  
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits, run_models=run_models, loss_type=loss_type)
