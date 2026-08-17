from B1_wing_SF_GPvsPFN import wing_SF_GPvsPFN_BO
from B2_buckling_SF_GPvsPFN import buckling_SF_GPvsPFN_BO
from B3_borehole_SF_GPvsPFN import borehole_SF_GPvsPFN_BO
from B4_ackley_GPvsPFN import ackley_GPvsPFN_BO
from B5_rastrigin_GPvsPFN import rastrigin_GPvsPFN_BO
from B6_rosenbrock_GPvsPFN import rosenbrock_GPvsPFN_BO
from B7_zakharov_GPvsPFN import zakharov_GPvsPFN_BO
from B8_griewank_GPvsPFN import griewank_GPvsPFN_BO
from B9_dixon_price_GPvsPFN import dixon_price_GPvsPFN_BO

folder = "results"
date = "check"

save_path_wing = f"./{folder}/{date}/B1_wing/"
save_path_buckling = f"./{folder}/{date}/B2_buckling/"
save_path_borehole = f"./{folder}/{date}/B3_borehole/"
save_path_ackley = f"./{folder}/{date}/B4_ackley/"
save_path_rastrigin = f"./{folder}/{date}/B5_rastrigin/"
save_path_rosenbrock = f"./{folder}/{date}/B6_rosenbrock/"
save_path_zakharov = f"./{folder}/{date}/B7_zakharov/"
save_path_griewank = f"./{folder}/{date}/B8_griewank/"
save_path_dixon_price = f"./{folder}/{date}/B9_dixon_price/"

num_runs = 2

num_inits = 16
start_size = 5

max_iter = 3
patience_no_improve = 0

# run_models = 'gp'
run_models = 'pfn'

# %% Wing ------------------------------------------------------------------------------------------------
# wing_SF_GPvsPFN_BO(title="warmup", num_runs=1, num_inits=num_inits, start_size=start_size, max_iter=1, patience_no_improve=0, save_path=None, run_models=run_models)
wing_SF_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_wing, run_models=run_models)

# # %% Buckling ------------------------------------------------------------------------------------------------
buckling_SF_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_buckling, run_models=run_models)

# %% Borehole ------------------------------------------------------------------------------------------------
borehole_SF_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, save_path=save_path_borehole, run_models=run_models)

# %% 5Dx Problems ------------------------------------------------------------------------------------------------
# %% Ackley ------------------------------------------------------------------------------------------------
ackley_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_ackley, run_models=run_models)

# %% Rastrigin ------------------------------------------------------------------------------------------------
rastrigin_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_rastrigin, run_models=run_models)

# # %% Rosenbrock ------------------------------------------------------------------------------------------------
rosenbrock_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_rosenbrock, run_models=run_models)

# # %% Zakharov ------------------------------------------------------------------------------------------------
zakharov_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_zakharov, run_models=run_models)

# %% Griewank ------------------------------------------------------------------------------------------------
griewank_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_griewank, run_models=run_models)

# %% Dixon Price ------------------------------------------------------------------------------------------------
dixon_price_GPvsPFN_BO(num_runs=num_runs, num_inits=num_inits, start_size=start_size, max_iter=max_iter, patience_no_improve=patience_no_improve, dimensions=5, save_path=save_path_dixon_price, run_models=run_models)