"""Run the full IDETC gpytorch suite with Adam defaults (lr=0.1, 10000 epochs).

Matches the problem/setup matrix in results_IDETC/10_runs_gpytorch_orig.
Saves to results_IDETC/10_runs_gpytorch_corrected_adam.
"""
from __future__ import annotations

import sys

# A*_gpytorch.py and gpytorch_train_eval import `defaults_gpytorch`.
# Patch before those imports so Adam settings (10000 epochs, lr=0.1) apply.
import defaults_gpytorch_adam as defaults_gpytorch

sys.modules["defaults_gpytorch"] = defaults_gpytorch

from A1_wing_SF_GPvsPFN_gpytorch import wing_SF_GPvsPFN
from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN
from A3_borehole_SF_GPvsPFN_gpytorch import borehole_SF_GPvsPFN
from A4_Ackley_GPvsPFN_gpytorch import ackley_GPvsPFN
from A5_rastrigin_GPvsPFN_gpytorch import rastrigin_GPvsPFN
from A6_rosenbrock_GPvsPFN_gpytorch import rosenbrock_GPvsPFN
from A7_zakharov_GPvsPFN_gpytorch import zakharov_GPvsPFN
from A8_griewank_GPvsPFN_gpytorch import griewank_GPvsPFN
from A9_dixon_price_GPvsPFN_gpytorch import dixon_price_GPvsPFN


def main() -> None:
    folder = "results_IDETC"
    date = "10_runs_gpytorch_corrected_adam"

    save_path_wing = f"./{folder}/{date}/wing/"
    save_path_buckling = f"./{folder}/{date}/buckling/"
    save_path_borehole = f"./{folder}/{date}/borehole/"
    save_path_ackley = f"./{folder}/{date}/ackley/"
    save_path_rosenbrock = f"./{folder}/{date}/rosenbrock/"
    save_path_rastrigin = f"./{folder}/{date}/rastrigin/"
    save_path_zakharov = f"./{folder}/{date}/zakharov/"
    save_path_griewank = f"./{folder}/{date}/griewank/"
    save_path_dixon_price = f"./{folder}/{date}/dixon_price/"

    num_test = 5000
    title = None
    num_inits = defaults_gpytorch.TRAINER_NUM_INITS
    num_runs = defaults_gpytorch.NUM_RUNS

    print(
        f"[B4 adam] optimizer={defaults_gpytorch.TRAINER_OPTIMIZER_CLASS.__name__}, "
        f"lr={defaults_gpytorch.TRAINER_LR}, epochs={defaults_gpytorch.TRAINER_NUM_EPOCHS}, "
        f"num_inits={num_inits}, num_runs={num_runs}, save=./{folder}/{date}/"
    )

    # Warmup (JIT / CUDA / TabPFN init) — not saved
    wing_SF_GPvsPFN(
        title="warmup",
        num_runs=2,
        train_size=5,
        save_path=None,
        noise_train=0.002,
        noise_test=0.002,
        num_test=num_test,
        num_inits=num_inits,
    )

    # %% Wing ------------------------------------------------------------------------------------------------
    wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    wing_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_wing, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Buckling ------------------------------------------------------------------------------------------------
    buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    buckling_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_buckling, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Borehole ------------------------------------------------------------------------------------------------
    borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=5, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    borehole_SF_GPvsPFN(title=title, num_runs=num_runs, train_size=20, save_path=save_path_borehole, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% 20 Dx Problems ------------------------------------------------------------------------------------------------
    # %% Ackley -------------------------------------------------------------
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Zakharov -------------------------------------------------------------
    zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    zakharov_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_zakharov, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Griewank -------------------------------------------------------------
    griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    griewank_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=20, save_path=save_path_griewank, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% 40 Dx Problems ------------------------------------------------------------------------------------------------
    # %% Ackley -------------------------------------------------------------
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    ackley_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_ackley, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Rastrigin --------------------------------------------------------------
    rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    rastrigin_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_rastrigin, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% Dixon Price -------------------------------------------------------------
    dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    dixon_price_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=40, save_path=save_path_dixon_price, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    # %% 80 Dx Problems ------------------------------------------------------------------------------------------------
    # %% Rosenbrock -------------------------------------------------------------
    rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.002, noise_test=0.002, num_test=num_test, num_inits=num_inits)
    rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=5, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)
    rosenbrock_GPvsPFN(title=title, num_runs=num_runs, train_size=20, dimensions=80, save_path=save_path_rosenbrock, noise_train=0.08, noise_test=0.08, num_test=num_test, num_inits=num_inits)

    print(f"[B4 adam] done. Results in ./{folder}/{date}/")


if __name__ == "__main__":
    main()
