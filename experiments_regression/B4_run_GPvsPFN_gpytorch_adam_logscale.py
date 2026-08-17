"""Run buckling + zakharov gpytorch log-scale suite with Adam defaults.

Saves to results_logscale_study/10_runs_gpytorch_corrected_adam_logscale.
"""
from __future__ import annotations

import sys

import defaults_gpytorch_adam as defaults_gpytorch

sys.modules["defaults_gpytorch"] = defaults_gpytorch

from A2_buckling_SF_GPvsPFN_gpytorch import buckling_SF_GPvsPFN
from A7_zakharov_GPvsPFN_gpytorch import zakharov_GPvsPFN


def main() -> None:
    folder = "results_logscale_study"
    date = "10_runs_gpytorch_corrected_adam_logscale"

    save_path_buckling = f"./{folder}/{date}/buckling/"
    save_path_zakharov = f"./{folder}/{date}/zakharov/"

    num_test = 5000
    title = None
    num_inits = defaults_gpytorch.TRAINER_NUM_INITS
    num_runs = defaults_gpytorch.NUM_RUNS

    print(
        f"[B4 adam logscale] optimizer={defaults_gpytorch.TRAINER_OPTIMIZER_CLASS.__name__}, "
        f"lr={defaults_gpytorch.TRAINER_LR}, epochs={defaults_gpytorch.TRAINER_NUM_EPOCHS}, "
        f"num_inits={num_inits}, num_runs={num_runs}, save=./{folder}/{date}/"
    )

    # Warmup — not saved
    buckling_SF_GPvsPFN(
        title="warmup",
        num_runs=1,
        train_size=5,
        save_path=None,
        noise_train=0.002,
        noise_test=0.002,
        num_test=50,
        num_inits=1,
        standardize_y_log_scale=True,
    )

    # Buckling (4D): 5/20 x 0.002/0.08
    for train_size in (5, 20):
        for noise in (0.002, 0.08):
            buckling_SF_GPvsPFN(
                title=title,
                num_runs=num_runs,
                train_size=train_size,
                save_path=save_path_buckling,
                noise_train=noise,
                noise_test=noise,
                num_test=num_test,
                num_inits=num_inits,
                standardize_y_log_scale=True,
            )

    # Zakharov (20D): 5/20 x 0.002/0.08
    for train_size in (5, 20):
        for noise in (0.002, 0.08):
            zakharov_GPvsPFN(
                title=title,
                num_runs=num_runs,
                train_size=train_size,
                dimensions=20,
                save_path=save_path_zakharov,
                noise_train=noise,
                noise_test=noise,
                num_test=num_test,
                num_inits=num_inits,
                standardize_y_log_scale=True,
            )

    print(f"[B4 adam logscale] done. Results in ./{folder}/{date}/")


if __name__ == "__main__":
    main()
