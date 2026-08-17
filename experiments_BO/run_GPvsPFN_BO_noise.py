from B1_wing_SF_GPvsPFN import wing_SF_GPvsPFN_BO
from B2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN_BO
from B3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN_BO
from B4_ackley_GPvsPFN import ackley_GPvsPFN_BO
from B5_rastrigin_GPvsPFN import rastrigin_GPvsPFN_BO
from B6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN_BO
from B7_zakharov_GPvsPFN import zakharov_GPvsPFN_BO
from B8_griewank_GPvsPFN import griewank_GPvsPFN_BO
from B9_dixon_price_GPvsPFN import dixon_price_GPvsPFN_BO

folder = "results_BO"
date = "IDETC_check"
# date = "GP+_check"
# date = "GP+"
# date = "GP+_no_AF_optimize"
# date = "PFN_V2.0"
# date = "PFN_V2.5"
# date = "PFN_V2.5_GI"
# date = "PFN_V2.5_check"
# date = "PFN_V2.0_GI"
# date = "PFN_V2.0_GI_TS"
# date = "PFN_V2.0_TS"

save_path_wing = f"./{folder}/{date}/B1_wing/"
save_path_buckling = f"./{folder}/{date}/B2_buckling/"
save_path_borehole = f"./{folder}/{date}/B3_borehole/"
save_path_ackley = f"./{folder}/{date}/B4_ackley/"
save_path_rastrigin = f"./{folder}/{date}/B5_rastrigin/"
save_path_rosenbrock = f"./{folder}/{date}/B6_rosenbrock/"
save_path_zakharov = f"./{folder}/{date}/B7_zakharov/"
save_path_griewank = f"./{folder}/{date}/B8_griewank/"
save_path_dixon_price = f"./{folder}/{date}/B9_dixon_price/"

num_runs = 10 # 10 is default

num_inits = 16
start_size = 5 # 5 is default

max_iter = 30 # 30 is default
patience_no_improve = 10 # 10 is default

# title = '2.0'
title = None
# run_models = 'pfn'
# run_models = 'gp'
run_models = None

low_noise = 0.002
high_noise = 0.08

# %% Wing ------------------------------------------------------------------------------------------------
wing_SF_GPvsPFN_BO(title="warmup", num_runs=2, num_inits=num_inits, start_size=start_size, max_iter=2, patience_no_improve=1, save_path=None, run_models=run_models)
# wing_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_wing, run_models=run_models)
wing_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_wing, run_models=run_models)
# # # # # %% Buckling ------------------------------------------------------------------------------------------------
# buckling_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_buckling, run_models=run_models)
buckling_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_buckling, run_models=run_models)
# # # %% Borehole ------------------------------------------------------------------------------------------------
# borehole_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_borehole, run_models=run_models)
borehole_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_borehole, run_models=run_models)
# # %% 20Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Ackley ------------------------------------------------------------------------------------------------
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_ackley, run_models=run_models)
# # # %% Zakharov ------------------------------------------------------------------------------------------------
# zakharov_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
zakharov_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
# # %% Griewank ------------------------------------------------------------------------------------------------
# griewank_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_griewank, run_models=run_models)
griewank_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_griewank, run_models=run_models)
# # %% 40Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Ackley ------------------------------------------------------------------------------------------------
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_ackley, run_models=run_models)
ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_ackley, run_models=run_models) #continue normal gp here
# %% Dixon Price ------------------------------------------------------------------------------------------------
# dixon_price_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
dixon_price_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
# %% 80Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Rosenbrock ------------------------------------------------------------------------------------------------
# rosenbrock_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)
# rosenbrock_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)

# run_models = 'gp'
# date = "GP+"

# # %% Wing ------------------------------------------------------------------------------------------------
# wing_SF_GPvsPFN_BO(title="warmup", num_runs=2, num_inits=num_inits, start_size=start_size, max_iter=2, patience_no_improve=1, save_path=None, run_models=run_models)
# wing_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_wing, run_models=run_models)
# wing_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_wing, run_models=run_models)
# # # # # %% Buckling ------------------------------------------------------------------------------------------------
# buckling_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_buckling, run_models=run_models)
# buckling_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_buckling, run_models=run_models)
# # # %% Borehole ------------------------------------------------------------------------------------------------
# borehole_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_borehole, run_models=run_models)
# borehole_SF_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_borehole, run_models=run_models)
# # %% 20Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Ackley ------------------------------------------------------------------------------------------------
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_ackley, run_models=run_models)
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_ackley, run_models=run_models)
# # # %% Zakharov ------------------------------------------------------------------------------------------------
# zakharov_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
# zakharov_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_zakharov, run_models=run_models)
# # %% Griewank ------------------------------------------------------------------------------------------------
# griewank_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_griewank, run_models=run_models)
# griewank_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=20, save_path=save_path_griewank, run_models=run_models)
# # %% 40Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Ackley ------------------------------------------------------------------------------------------------
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_ackley, run_models=run_models)
# ackley_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_ackley, run_models=run_models) #continue normal gp here
# # %% Dixon Price ------------------------------------------------------------------------------------------------
# dixon_price_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
# dixon_price_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=40, save_path=save_path_dixon_price, run_models=run_models)
# # %% 80Dx Problems ------------------------------------------------------------------------------------------------
# # # %% Rosenbrock ------------------------------------------------------------------------------------------------
# rosenbrock_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=low_noise, noise_test=low_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)
# rosenbrock_GPvsPFN_BO(title=title, num_runs=num_runs, num_inits=num_inits, start_size=start_size, noise_train=high_noise, noise_test=high_noise, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=80, save_path=save_path_rosenbrock, run_models=run_models)