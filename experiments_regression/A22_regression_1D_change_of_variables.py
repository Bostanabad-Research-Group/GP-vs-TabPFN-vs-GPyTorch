"""
A22 — 1D regression benchmarks: LML change-of-variables study (omega vs outputscale).

For each TabPFN-style 1D benchmark, plots the GP hyperparameter search space in
transformed coordinates (omega vs log10 outputscale) with zero nugget and no
training-data noise. Runs at n=20 and n=100 Sobol training points.

No GP training — LML surface evaluation only.

Run:
  python A22_regression_1D_change_of_variables.py
"""

from __future__ import annotations

import time
from pathlib import Path

import defaults
import gpplus
import torch
from experimental_utils.lml_surface_omega_outputscale import run_lml_1d_omega_outputscale_study
from load_experimental_data import (
    generate_tabpfn_1d_chirp_data,
    generate_tabpfn_1d_damped_forrester_data,
    generate_tabpfn_1d_damped_sine_data,
    generate_tabpfn_1d_discontinuity_data,
    generate_tabpfn_1d_forrester_data,
    generate_tabpfn_1d_gramacy_lee_data,
    generate_tabpfn_1d_localized_bump_data,
    generate_tabpfn_1d_sin_cubic_data,
    generate_tabpfn_1d_smooth_multisine_data,
    generate_tabpfn_1d_triangle_wave_data,
)

SAVE_ROOT = Path("./results/A22_regression_1D_change_of_variables_401")
TRAIN_SIZES = (20, 100)
N_GRID = 401
OMEGA_ZOOM_HALF_WIDTH = 0.15
S_ZOOM_HALF_WIDTH = 0.15
LENGTHSCALE_ZOOM_HALF_WIDTH = 0.15
LML_CLIP_BELOW_BEST = 10.0

REGRESSION_1D_FUNCTIONS = {
    "forrester": {
        "generate_data": generate_tabpfn_1d_forrester_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Forrester benchmark",
    },
    "damped_forrester": {
        "generate_data": generate_tabpfn_1d_damped_forrester_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "5*damped sine + Forrester",
    },
    "sin_cubic": {
        "generate_data": generate_tabpfn_1d_sin_cubic_data,
        "default_x_bounds": [0.0, 1.0],
        "description": "1/4*sin(6*pi*x) + 6*x^3 - 7*x^2 + x + 0.5",
    },
    "smooth_multisine": {
        "generate_data": generate_tabpfn_1d_smooth_multisine_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "sin(2*pi*x) + 0.5*sin(6*pi*x)",
    },
    "chirp": {
        "generate_data": generate_tabpfn_1d_chirp_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Linear chirp (non-stationary frequency)",
    },
    "discontinuity": {
        "generate_data": generate_tabpfn_1d_discontinuity_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "0.6*sin(2*pi*x) + Heaviside jump at x=0",
    },
    "localized_bump": {
        "generate_data": generate_tabpfn_1d_localized_bump_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Flat + narrow Gaussian spike",
    },
    "triangle_wave": {
        "generate_data": generate_tabpfn_1d_triangle_wave_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Triangle wave (periodic kinks)",
    },
    "gramacy_lee": {
        "generate_data": generate_tabpfn_1d_gramacy_lee_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "Gramacy & Lee sin(10*pi*z)/(2z) + (z-1)^4",
    },
    "damped_sine": {
        "generate_data": generate_tabpfn_1d_damped_sine_data,
        "default_x_bounds": [-0.5, 0.5],
        "description": "exp(-6*|x|)*sin(10*pi*x)",
    },
}


def _preprocess_training_data(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    standardize_x: bool,
    standardize_y: bool,
    x_standardize_method: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = X_train.detach().clone().to(dtype=torch.float64)
    y = y_train.detach().clone().to(dtype=torch.float64)

    if standardize_x:
        if x_standardize_method == 0:
            Xscaler = gpplus.utils.StandardScaler()
        elif x_standardize_method == 1:
            Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
        elif x_standardize_method == 2:
            Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
        else:
            raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
        Xscaler.fit(X)
        X = Xscaler.transform(X)

    if standardize_y:
        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y)
        y = Yscaler.transform(y)
    else:
        y = y - y.mean()

    return X, y


def run_lml_change_of_variables_study(
    function_name: str,
    *,
    train_n: int,
    x_bounds: list[float] | None = None,
    seed: int = defaults.SEED,
    save_root: Path | str = SAVE_ROOT,
    standardize_x: bool = defaults.STANDARDIZE_X,
    standardize_y: bool = defaults.STANDARDIZE_Y,
    x_standardize_method: int = defaults.X_STANDARDIZE_METHOD,
) -> dict:
    if function_name not in REGRESSION_1D_FUNCTIONS:
        raise ValueError(
            f"Unknown function_name={function_name!r}; choose from {list(REGRESSION_1D_FUNCTIONS)}"
        )

    fn_cfg = REGRESSION_1D_FUNCTIONS[function_name]
    generate_data_fn = fn_cfg["generate_data"]
    description = fn_cfg["description"]
    if x_bounds is None:
        x_bounds = list(fn_cfg["default_x_bounds"])

    print(f"\n--- {function_name} (n={train_n}) — {description} ---")
    print(f"  x_bounds={x_bounds}, seed={seed}, noise=0")

    X_train, y_train, _, _ = generate_data_fn(
        n_train=train_n,
        n_test=1,
        dimensions=1,
        x_bounds=x_bounds,
        train_noise=0.0,
        test_noise=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=seed,
    )

    x_used, y_used = _preprocess_training_data(
        X_train,
        y_train,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
        x_standardize_method=x_standardize_method,
    )

    save_path = Path(save_root) / function_name / f"n{train_n}"
    result = run_lml_1d_omega_outputscale_study(
        save_path=save_path,
        problem_name=function_name,
        train_x=x_used,
        y_used=y_used,
        train_n=train_n,
        nugget=0.0,
        n_grid=N_GRID,
        cholesky_jitter=0.0,
        profile_constant_mean=False,
        normalize_lml_by_n=True,
        lml_clip_below_best=LML_CLIP_BELOW_BEST,
        zoom_lml_around_coarse_best=True,
        omega_zoom_half_width=OMEGA_ZOOM_HALF_WIDTH,
        s_zoom_half_width=S_ZOOM_HALF_WIDTH,
        lengthscale_zoom_half_width=LENGTHSCALE_ZOOM_HALF_WIDTH,
        panel_title=description,
    )

    for path in result["files"]:
        print(f"  Saved {path}")

    opt = result.get("grid_optimum")
    if opt is not None:
        print(
            f"  Grid max: omega={opt['omega']:.4g}, "
            f"outputscale={opt['outputscale']:.4g}, lml={opt['lml']:.4f}"
        )
    print(f"  Elapsed {result['timing_seconds']:.1f}s")

    return result


def run_all(
    *,
    functions: tuple[str, ...] | None = None,
    train_sizes: tuple[int, ...] = TRAIN_SIZES,
    save_root: Path | str = SAVE_ROOT,
    seed: int = defaults.SEED,
) -> None:
    if functions is None:
        functions = tuple(REGRESSION_1D_FUNCTIONS.keys())

    total_start = time.time()
    for fn_name in functions:
        for train_n in train_sizes:
            run_lml_change_of_variables_study(
                fn_name,
                train_n=train_n,
                seed=seed,
                save_root=save_root,
            )

    print(f"\nDone — {len(functions)} problems × {len(train_sizes)} train sizes")
    print(f"Figures under {Path(save_root).resolve()}/")
    print(f"Total elapsed: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    if defaults.WARNINGS_IGNORE:
        import warnings

        warnings.filterwarnings("ignore")

    print("=" * 72)
    print("A22 LML change-of-variables study — all 1D benchmarks")
    print("=" * 72)
    run_all(functions=("damped_forrester", "sin_cubic"))
