from A1_wing_SF_GPvsPFN_gpytorch import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN_gpytorch import borehole_SF_GPvsPFN
from A4_Ackley_GPvsPFN_gpytorch import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN_gpytorch import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN_gpytorch import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN_gpytorch import zakharov_GPvsPFN
from A8_griewank_GPvsPFN_gpytorch import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN_gpytorch import dixon_price_GPvsPFN

folder = "results_March13"
date = "10_runs_gpytorch"
# date = "test96"

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
# title = "2.5"
# title = 'V3'
title = None


num_runs = 10

wing_SF_GPvsPFN(title="warmup", num_runs=2, train_size=10, save_path=None, noise_train=0.002, noise_test=0.002, num_test=num_test)
# wing_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test)
# buckling_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test)
# borehole_SF_GPvsPFN(title="test", num_runs=2, train_size=10, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test)

# # # %% Wing ------------------------------------------------------------------------------------------------
# wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test)
# wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test)
# wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test)
# wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Buckling ------------------------------------------------------------------------------------------------
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test)
# buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Borehole ------------------------------------------------------------------------------------------------
# borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test)
# borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test)
borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% 20 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Rosenbrock -------------------------------------------------------------
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)

# %% Zakharov -------------------------------------------------------------
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)

# %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Dixon Price -------------------------------------------------------------  
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)

# %% 40 Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley -------------------------------------------------------------
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)

# %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Rosenbrock -------------------------------------------------------------
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Zakharov -------------------------------------------------------------
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)

# %% Dixon Price -------------------------------------------------------------  
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% 80 Dx Problems ------------------------------------------------------------------------------------------------
# # %% Ackley -------------------------------------------------------------
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)
# ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # %% Rastrigin --------------------------------------------------------------
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Rosenbrock -------------------------------------------------------------
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test)
# rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Zakharov -------------------------------------------------------------
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)
# zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Griewank -------------------------------------------------------------
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)
# griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test)

# # # %% Dixon Price -------------------------------------------------------------  
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)
# dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test)
