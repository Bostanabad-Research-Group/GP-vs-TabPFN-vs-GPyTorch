"""Shared GPyTorch training/eval utilities for GPvsPFN experiments.

This module exists to avoid duplicating the large `train_eval_gp_gpytorch_default` helper
across each `*_gpytorch.py` experiment script.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from typing import Any, Callable

import numpy as np
import torch
import gpytorch
import linear_operator
from joblib import Parallel, delayed, parallel_config

try:
    from linear_operator.utils.errors import NotPSDError, NanError
except Exception:  # pragma: no cover
    from linear_operator.utils.errors import NotPSDError  # type: ignore

    NanError = NotPSDError  # type: ignore

from gpplus.training.eval import evaluate_gp_model
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils.metrics_functions import compute_metrics

# Allow this helper to work when running from GPvsPFN/experiments (local import).
try:
    import defaults_gpytorch as defaults
except ModuleNotFoundError:  # pragma: no cover
    from experiments import defaults_gpytorch as defaults  # type: ignore


def _is_lbfgs_scipy(optimizer_class) -> bool:
    return optimizer_class is LBFGSScipy or (
        isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy)
    )


def _is_lbfgs_like(optimizer_class) -> bool:
    return (
        _is_lbfgs_scipy(optimizer_class)
        or optimizer_class is torch.optim.LBFGS
        or (
            isinstance(optimizer_class, type)
            and issubclass(optimizer_class, torch.optim.LBFGS)
        )
    )


def _lbfgs_optimizer_kwargs() -> dict[str, Any]:
    """Resolve LBFGS / LBFGSScipy kwargs from defaults (public-repo style 2k/5k)."""
    kwargs = getattr(defaults, "TRAINER_OPTIMIZER_KWARGS", None)
    if isinstance(kwargs, dict) and kwargs:
        return dict(kwargs)
    return {
        "max_iter": int(getattr(defaults, "LBFGS_MAX_ITER", 2000)),
        "max_eval": int(getattr(defaults, "LBFGS_MAX_EVAL", 5000)),
        "tolerance_grad": float(getattr(defaults, "LBFGS_TOLERANCE_GRAD", 1e-5)),
        "tolerance_change": float(getattr(defaults, "LBFGS_TOLERANCE_CHANGE", 1e-9)),
        "history_size": int(getattr(defaults, "LBFGS_HISTORY_SIZE", 10)),
    }


def _reinit_gp_hyperparams(model, rng: torch.Generator) -> None:
    """Overwrite GP hyperparameters with N(0, 0.1) draws (matches prior sequential path)."""
    with torch.no_grad():
        if hasattr(model.likelihood, "raw_noise"):
            model.likelihood.raw_noise.data.normal_(mean=0, std=0.1, generator=rng)
        if hasattr(model.mean_module, "constant"):
            model.mean_module.constant.data.normal_(mean=0, std=0.1, generator=rng)
        if hasattr(model.covar_module, "base_kernel"):
            if hasattr(model.covar_module.base_kernel, "raw_lengthscale"):
                model.covar_module.base_kernel.raw_lengthscale.data.normal_(
                    mean=0, std=0.1, generator=rng
                )
            elif hasattr(model.covar_module.base_kernel, "lengthscale"):
                model.covar_module.base_kernel.lengthscale.data.normal_(
                    mean=1.0, std=0.1, generator=rng
                )
        if hasattr(model.covar_module, "raw_outputscale"):
            model.covar_module.raw_outputscale.data.normal_(mean=0, std=0.1, generator=rng)


def _train_one_gpytorch_init(
    model_template,
    start_state: dict,
    run_idx: int,
    num_inits: int,
    num_epochs: int,
    optimizer_class,
    run_lr: float,
    convergence_patience: int | None,
    min_loss_change: float,
    device: str,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train a single random-init (joblib-worker safe). Returns loss / state / jitter."""
    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(start_state)
    model.train()
    model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    is_lbfgs_scipy = _is_lbfgs_scipy(optimizer_class)
    is_torch_lbfgs = optimizer_class is torch.optim.LBFGS or (
        isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.LBFGS)
    )
    is_lbfgs = is_lbfgs_scipy or is_torch_lbfgs
    lbfgs_kwargs = _lbfgs_optimizer_kwargs() if is_lbfgs else {}

    # Track best LBFGS inner iteration (scipy callback / torch has no per-iter hook).
    best_iter_state = {"best_iter": None, "best_loss": float("inf")}

    def _lbfgs_iteration_callback(iteration=None, loss=None, flat_params=None, **_kwargs):
        if iteration is None or loss is None:
            return
        try:
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
        except (TypeError, ValueError):
            return
        if loss_val < best_iter_state["best_loss"]:
            best_iter_state["best_loss"] = loss_val
            best_iter_state["best_iter"] = int(iteration)

    if is_lbfgs_scipy:
        opt_kwargs = dict(lbfgs_kwargs)
        opt_kwargs["iteration_callback"] = _lbfgs_iteration_callback
        optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    elif is_torch_lbfgs:
        optimizer = optimizer_class(
            model.parameters(),
            lr=run_lr if run_lr is not None else 1.0,
            **lbfgs_kwargs,
        )
    elif optimizer_class == torch.optim.Adam:
        optimizer = optimizer_class(
            model.parameters(),
            lr=run_lr,
            betas=getattr(defaults, "ADAM_BETAS", (0.9, 0.999)),
            eps=getattr(defaults, "ADAM_EPS", 1e-8),
            weight_decay=getattr(defaults, "ADAM_WEIGHT_DECAY", 0.0),
        )
    else:
        optimizer = optimizer_class(model.parameters(), lr=run_lr)

    def closure():
        optimizer.zero_grad()
        try:
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(1e6, dtype=loss.dtype, device=loss.device, requires_grad=True)
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0.0
            return loss
        except NotPSDError:
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            return torch.tensor(1e6, dtype=model_dtype, device=model_device, requires_grad=True)
        except (RuntimeError, ValueError) as e:
            err = str(e).lower()
            if "nan" in err or "notpsd" in err or "not p.d." in err:
                model_dtype = next(model.parameters()).dtype
                model_device = next(model.parameters()).device
                return torch.tensor(1e6, dtype=model_dtype, device=model_device, requires_grad=True)
            raise

    run_best_loss = float("inf")
    run_best_iter = None
    run_best_state_dict = None
    no_improvement_epochs = 0
    previous_loss = None
    early_stop_triggered = False
    jitter = 1e-6
    max_jitter = 1e-3
    run_jitter = jitter
    messages: list[str] = []

    def _log(msg: str) -> None:
        if quiet:
            messages.append(msg)
        else:
            print(msg)

    # LBFGS-like: one outer step per init (inner iters handled by optimizer).
    epochs_to_run = 1 if is_lbfgs else num_epochs

    for epoch in range(epochs_to_run):
        loss_val = None
        epoch_successful = False
        skip_run = False

        while not epoch_successful and not skip_run:
            with (
                gpytorch.settings.cholesky_jitter(jitter),
                linear_operator.settings.cholesky_jitter(
                    float_value=jitter, double_value=jitter
                ),
            ):
                try:
                    if is_lbfgs:
                        loss = optimizer.step(closure)
                        if is_lbfgs_scipy and hasattr(optimizer, "_last_loss"):
                            loss = optimizer._last_loss
                    else:
                        loss = closure()
                        optimizer.step()

                    if torch.isnan(loss) or torch.isinf(loss):
                        _log(
                            f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                            f"NaN/Inf loss detected. Skipping this run."
                        )
                        skip_run = True
                        break

                    loss_val = loss.item() if hasattr(loss, "item") else float(loss)
                    epoch_successful = True
                except NotPSDError:
                    if jitter < max_jitter:
                        jitter = min(jitter * 10, max_jitter)
                        run_jitter = jitter
                        _log(
                            f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                            f"NotPSDError detected. Increasing jitter to {jitter:.1e}."
                        )
                        epoch_successful = False
                    else:
                        _log(
                            f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                            f"NotPSDError persists even with jitter={jitter:.1e}. Skipping this run."
                        )
                        skip_run = True
                        break
                except (RuntimeError, ValueError) as e:
                    error_str = str(e).lower()
                    if "nan" in error_str or "nanerror" in error_str:
                        _log(
                            f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                            f"NaN error detected: {e}. Skipping this run."
                        )
                        skip_run = True
                        break
                    if (
                        "notpsd" in error_str
                        or "not p.d." in error_str
                        or "not positive definite" in error_str
                    ):
                        if jitter < max_jitter:
                            jitter = min(jitter * 10, max_jitter)
                            run_jitter = jitter
                            _log(
                                f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                                f"NotPSD error detected. Increasing jitter to {jitter:.1e}."
                            )
                            epoch_successful = False
                            continue
                        _log(
                            f"  Run {run_idx + 1}/{num_inits}, Epoch {epoch + 1}/{epochs_to_run}: "
                            f"NotPSD error persists even with jitter={jitter:.1e}. Skipping this run."
                        )
                        skip_run = True
                        break
                    raise

        if skip_run or loss_val is None:
            break

        if loss_val < run_best_loss:
            significant_improvement = (run_best_loss - loss_val) >= min_loss_change
            run_best_loss = loss_val
            run_best_state_dict = copy.deepcopy(model.state_dict())
            if is_lbfgs:
                if best_iter_state["best_iter"] is not None:
                    run_best_iter = best_iter_state["best_iter"]
                elif is_lbfgs_scipy:
                    run_best_iter = int(getattr(optimizer, "_n_iter", 0)) or None
                else:
                    # torch.optim.LBFGS: no per-iter callback; report max_iter budget used as n/a → n_iter unknown
                    run_best_iter = None
            else:
                run_best_iter = epoch
            if significant_improvement:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
        else:
            no_improvement_epochs += 1

        # Outer-loop early stopping only for multi-epoch (non-LBFGS) optimizers.
        if is_lbfgs:
            break

        early_stop_reason = ""
        if convergence_patience is not None and no_improvement_epochs >= convergence_patience:
            early_stop_triggered = True
            early_stop_reason = (
                f"No improvement >= {min_loss_change:.1e} for {convergence_patience} epochs"
            )

        if previous_loss is not None:
            loss_change = abs(previous_loss - loss_val)
            if loss_change < min_loss_change:
                early_stop_triggered = True
                if early_stop_reason:
                    early_stop_reason += (
                        f" OR absolute loss change below {min_loss_change:.1e}"
                    )
                else:
                    early_stop_reason = f"absolute loss change below {min_loss_change:.1e}"

        if early_stop_triggered:
            if run_best_state_dict is not None:
                model.load_state_dict(run_best_state_dict)
            _log(
                f"  Early stopping at epoch {epoch + 1}: {early_stop_reason}. "
                f"Best loss: {run_best_loss:.6f}"
            )
            break

        previous_loss = loss_val

    if run_best_state_dict is not None:
        model.load_state_dict(run_best_state_dict)
        state_out = copy.deepcopy(run_best_state_dict)
    else:
        state_out = None

    if run_best_iter is None and is_lbfgs_scipy:
        run_best_iter = int(getattr(optimizer, "_n_iter", 0)) or None

    return {
        "run_index": run_idx,
        "loss": float(run_best_loss),
        "best_iter": run_best_iter,
        "state_dict": state_out,
        "jitter": run_jitter,
        "messages": messages,
    }


def train_eval_gp_gpytorch_default(
    model,
    X_test: torch.Tensor,
    y_test,
    num_epochs: int,
    num_inits: int = 1,
    seed: int | None = None,    
    device: str = "cpu",
    y_train_mean: torch.Tensor | None = None,
    y_train_std: torch.Tensor | None = None,
    convergence_patience: int | None = None,
    # Absolute MLL improvement required to reset patience / for plateau stop.
    # 1e-7 is too small for Adam (microscopic steps reset patience for ~10k epochs).
    min_loss_change: float | None = None,
    optimizer_class=None,
    lr: float | None = None,
    standardize_y_log_scale: bool = False,
    y_train_min: float | None = None,
    log_scale_C: float | None = None,  # C used in log(y + C). Required when standardize_y_log_scale=True.
    log_y_point_inverse: str = "median",  # "median": exp(mu)-C; "mean": exp(mu+sigma^2/2)-C
    iteration_callbacks: list[Callable[..., None]] | None = None,
    n_jobs: int | None = None,
):
    """
    Train a GP model using gpytorch components (ExactMarginalLogLikelihood).

    - Supports multiple runs with different initializations, selecting the best run by loss.
    - CPU multi-init training is parallelized with joblib (same idea as gpplus).
    - Supports LBFGS (with closure) and standard optimizers (Adam/SGD/etc.).
    - Adds robust NaN/NotPSD handling via jitter escalation.
    - Optional iteration_callbacks: only used when n_jobs == 1 (series mode).

    Returns:
        gp_metric: dict of computed metrics
        y_pred: numpy array of predictions (denormalized if mean/std provided)
        output_std: numpy array of predictive std (denormalized if mean/std provided)
    """

    # Move model and data to device (keep y_test on CPU for metric computation)
    model = model.to(device)
    X_test = X_test.to(device)
    # Keep y_test on CPU - it's in original scale and used only for metrics
    y_test_cpu = y_test.detach().clone().cpu() if isinstance(y_test, torch.Tensor) else y_test

    original_state = copy.deepcopy(model.state_dict())

    # Track best run
    best_loss = float("inf")
    best_state_dict = None
    best_run_index = None
    best_iter = None
    run_results: list[dict[str, Any]] = []

    # Track final jitter value used during training (for evaluation)
    final_jitter = 1e-6  # Default gpytorch jitter

    # Track best loss per run (for diagnostics / logging).
    best_loss_per_run: list[float] = [float("inf")] * num_inits

    # Training time tracking
    t_train_start = time.time()
    logging_time = 0.0  # time spent on printing / callbacks during training (not optimizer work)

    # Multiple runs with different initializations
    # Use seed to create reproducible random state for each run
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Resolve optimizer once. LBFGS/LBFGSScipy: one outer epoch per init (up to max_iter inside).
    if optimizer_class is None:
        optimizer_class = getattr(defaults, "TRAINER_OPTIMIZER_CLASS", LBFGSScipy)
    if num_epochs is None:
        num_epochs = int(getattr(defaults, "TRAINER_NUM_EPOCHS", 1))
    else:
        num_epochs = int(num_epochs)
    if _is_lbfgs_like(optimizer_class):
        num_epochs = 1

    if min_loss_change is None:
        min_loss_change = float(getattr(defaults, "TRAINER_MIN_LOSS_CHANGE", 1e-7))
    else:
        min_loss_change = float(min_loss_change)

    run_lr = lr
    if run_lr is None:
        if _is_lbfgs_like(optimizer_class):
            run_lr = getattr(defaults, "LBFGS_LR", 1.0)
        else:
            run_lr = getattr(defaults, "TRAINER_LR", 0.1)

    # Build start states in the main process so RNG matches the old sequential path.
    start_states: list[dict] = []
    model.load_state_dict(original_state)
    for run_idx in range(num_inits):
        if run_idx > 0:
            _reinit_gp_hyperparams(model, rng)
        start_states.append(copy.deepcopy(model.state_dict()))

    # Parallelize multi-init on CPU (gpplus-style). Force series if callbacks need ordering.
    if n_jobs is None:
        n_jobs = getattr(defaults, "TRAINER_N_JOBS", None)
    if iteration_callbacks:
        max_jobs = 1
    elif str(device).startswith("cuda"):
        max_jobs = 1  # ExactGP multi-init on one GPU stays series
    else:
        requested = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 1) - 2)
        max_jobs = min(num_inits, max(1, int(requested)))

    quiet = max_jobs > 1
    if max_jobs > 1:
        print(
            f"  Training {num_inits} inits with {max_jobs} parallel CPU workers "
            f"(joblib loky, 1 BLAS thread/worker)."
        )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            init_results = Parallel(n_jobs=max_jobs, verbose=0)(
                delayed(_train_one_gpytorch_init)(
                    model,
                    start_states[run_idx],
                    run_idx,
                    num_inits,
                    num_epochs,
                    optimizer_class,
                    run_lr,
                    convergence_patience,
                    min_loss_change,
                    device,
                    quiet=quiet,
                )
                for run_idx in range(num_inits)
            )
    else:
        init_results = [
            _train_one_gpytorch_init(
                model,
                start_states[run_idx],
                run_idx,
                num_inits,
                num_epochs,
                optimizer_class,
                run_lr,
                convergence_patience,
                min_loss_change,
                device,
                quiet=False,
            )
            for run_idx in range(num_inits)
        ]

    for result in init_results:
        for msg in result.get("messages") or []:
            print(msg)
        run_idx = int(result["run_index"])
        final_loss = float(result["loss"])
        best_loss_per_run[run_idx] = final_loss
        run_results.append(
            {
                "run_index": run_idx,
                "loss": final_loss,
                "best_iter": result.get("best_iter"),
                "state_dict": result["state_dict"],
            }
        )
        if result["state_dict"] is not None and final_loss < best_loss:
            best_loss = final_loss
            best_state_dict = copy.deepcopy(result["state_dict"])
            best_run_index = run_idx
            best_iter = result.get("best_iter")
            final_jitter = float(result["jitter"])

    total_training_wall_time = time.time() - t_train_start
    training_time = total_training_wall_time - logging_time

    # Check if any run was successful
    if best_state_dict is None:
        print(f"  ERROR: All {num_inits} training runs failed due to numerical instability.")
        print("  This indicates severe numerical issues. Possible causes:")
        print("    - Data scaling problems")
        print("    - Incompatible hyperparameter initialization")
        print(f"    - Insufficient jitter (max tried: {1e-3})")
        print("    - Model/data mismatch")
        # Return dummy metrics with NaN values - skip compute_metrics since it can't handle NaN
        y_pred_np = np.full(len(y_test_cpu), np.nan)
        output_std_np = np.full(len(y_test_cpu), np.nan)
        # Create metrics dict manually with NaN values
        gp_metric = {
            "Total_Time": total_training_wall_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": 0.0,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,  # Record the jitter that was attempted
            "evaluation_error": f"All {num_inits} training runs failed - no valid model",
            "all_runs_failed": True,
            "all_metrics_nan": True,
            "best_loss_per_run": best_loss_per_run,
        }
        return gp_metric, y_pred_np, output_std_np

    # Load best model state
    model.load_state_dict(best_state_dict)
    model.eval()
    model.likelihood.eval()

    # Validate model parameters before evaluation
    has_nan_inf = False
    param_info: list[str] = []

    # Check likelihood noise
    if hasattr(model.likelihood, "raw_noise"):
        noise_val = model.likelihood.raw_noise.detach()
        if torch.isnan(noise_val).any() or torch.isinf(noise_val).any():
            has_nan_inf = True
            param_info.append(f"likelihood.raw_noise: {noise_val}")

    # Check outputscale
    if hasattr(model.covar_module, "raw_outputscale"):
        outputscale_val = model.covar_module.raw_outputscale.detach()
        if torch.isnan(outputscale_val).any() or torch.isinf(outputscale_val).any():
            has_nan_inf = True
            param_info.append(f"covar_module.raw_outputscale: {outputscale_val}")

    # Check lengthscales
    if hasattr(model.covar_module, "base_kernel") and hasattr(
        model.covar_module.base_kernel, "raw_lengthscale"
    ):
        lengthscale_val = model.covar_module.base_kernel.raw_lengthscale.detach()
        if torch.isnan(lengthscale_val).any() or torch.isinf(lengthscale_val).any():
            has_nan_inf = True
            param_info.append(f"base_kernel.raw_lengthscale: {lengthscale_val}")

    if has_nan_inf:
        print(
            f"  WARNING: Model has invalid hyperparameters (NaN/Inf detected): {param_info}"
        )
        print("  This indicates numerical instability. Skipping evaluation for this run.")
        y_pred_np = np.full(len(y_test_cpu), np.nan)
        output_std_np = np.full(len(y_test_cpu), np.nan)
        gp_metric = {
            "Total_Time": training_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": 0.0,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,
            "evaluation_error": f"NaN/Inf in parameters: {param_info}",
            "all_metrics_nan": True,
        }
        return gp_metric, y_pred_np, output_std_np

    # Evaluation with error handling and jitter settings
    t_pred_start = time.time()
    try:
        eval_jitter = final_jitter
        with (
            gpytorch.settings.cholesky_jitter(eval_jitter),
            linear_operator.settings.cholesky_jitter(
                float_value=eval_jitter, double_value=eval_jitter
            ),
        ):
            y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
    except (NanError, NotPSDError) as e:
        print(f"  WARNING: Evaluation failed due to numerical instability: {e}")
        print("  Attempting evaluation with increased jitter...")
        try:
            with (
                gpytorch.settings.cholesky_jitter(1e-3),
                linear_operator.settings.cholesky_jitter(
                    float_value=1e-3, double_value=1e-3
                ),
            ):
                y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
                final_jitter = 1e-3
        except Exception as e2:
            print(f"  ERROR: Evaluation failed even with maximum jitter: {e2}")
            y_pred_np = np.full(len(y_test_cpu), np.nan)
            output_std_np = np.full(len(y_test_cpu), np.nan)
            gp_metric = {
                "Total_Time": training_time,
                "Training_Time": training_time,
                "Logging_Time": logging_time,
                "Prediction_Time": 0.0,
                "RRMSE": np.nan,
                "RMSE": np.nan,
                "MSE": np.nan,
                "NIS": np.nan,
                "NIS_width": np.nan,
                "NIS_outside": np.nan,
                "jitter": final_jitter,
                "evaluation_error": str(e2),
                "all_metrics_nan": True,
            }
            return gp_metric, y_pred_np, output_std_np
    except (RuntimeError, ValueError) as e:
        error_str = str(e).lower()
        if (
            "nan" in error_str
            or "notpsd" in error_str
            or "not p.d." in error_str
            or "not positive definite" in error_str
        ):
            print(f"  WARNING: Evaluation failed due to numerical instability: {e}")
            print("  Attempting evaluation with increased jitter...")
            try:
                with (
                    gpytorch.settings.cholesky_jitter(1e-3),
                    linear_operator.settings.cholesky_jitter(
                        float_value=1e-3, double_value=1e-3
                    ),
                ):
                    y_pred, _, _, output_std = evaluate_gp_model(model, X_test)
                    final_jitter = 1e-3
            except Exception as e2:
                print(f"  ERROR: Evaluation failed even with maximum jitter: {e2}")
                y_pred_np = np.full(len(y_test_cpu), np.nan)
                output_std_np = np.full(len(y_test_cpu), np.nan)
                gp_metric = {
                    "Total_Time": training_time,
                    "Training_Time": training_time,
                    "Logging_Time": logging_time,
                    "Prediction_Time": 0.0,
                    "RRMSE": np.nan,
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "NIS": np.nan,
                    "NIS_width": np.nan,
                    "NIS_outside": np.nan,
                    "jitter": final_jitter,
                    "evaluation_error": str(e2),
                    "all_metrics_nan": True,
                }
                return gp_metric, y_pred_np, output_std_np
        else:
            raise

    prediction_time = time.time() - t_pred_start

    # Denormalize if needed
    if y_train_mean is not None and y_train_std is not None:
        if standardize_y_log_scale:
            if log_scale_C is None:
                raise ValueError(
                    "train_eval_gp_gpytorch_default: standardize_y_log_scale=True requires log_scale_C."
                )
            # y_pred/output_std are in standardized log space: undo standardization, then exp.
            log_y_pred = (y_pred * y_train_std) + y_train_mean
            log_y_std = output_std * y_train_std
            max_log_val = 700.0 if log_y_pred.dtype == torch.float32 else 1000.0
            log_y_pred = torch.clamp(log_y_pred, min=-max_log_val, max=max_log_val)
            exp_log_y = torch.exp(log_y_pred)
            if log_y_point_inverse == "mean":
                half_var = 0.5 * (log_y_std**2)
                y_pred = exp_log_y * torch.exp(half_var) - float(log_scale_C)
            else:  # "median"
                y_pred = exp_log_y - float(log_scale_C)
            # Delta-method std in original scale: d/dz exp(z) = exp(z)
            output_std = exp_log_y * log_y_std
        else:
            y_pred = (y_pred * y_train_std) + y_train_mean
            output_std = output_std * y_train_std

    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    output_std_np = output_std.detach().cpu().numpy().reshape(-1)

    # Check if predictions are all NaN before computing metrics
    if np.all(np.isnan(y_pred_np)):
        print("  WARNING: All predictions are NaN. Skipping metric computation.")
        gp_metric = {
            "Total_Time": training_time + logging_time + prediction_time,
            "Training_Time": training_time,
            "Logging_Time": logging_time,
            "Prediction_Time": prediction_time,
            "RRMSE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "NIS": np.nan,
            "NIS_width": np.nan,
            "NIS_outside": np.nan,
            "jitter": final_jitter,
            "evaluation_error": "All predictions are NaN",
            "all_metrics_nan": True,
        }
    else:
        gp_metric = compute_metrics(
            y_test_cpu,
            y_pred_np,
            output_std_np,
            training_time=training_time,
            prediction_time=prediction_time,
        )

    # Check if all metrics are NaN (indicates complete failure)
    excluded_keys = [
        "training_time",
        "prediction_time",
        "num_epochs",
        "best_iter",
        "best_epoch",
        "evaluation_error",
        "all_runs_failed",
        "all_metrics_nan",
    ]
    metric_values = [
        v
        for k, v in gp_metric.items()
        if k not in excluded_keys and isinstance(v, (int, float))
    ]
    if len(metric_values) > 0 and all(np.isnan(v) for v in metric_values):
        print(
            "  WARNING: All computed metrics are NaN. This indicates evaluation failed completely."
        )
        gp_metric["evaluation_error"] = "All metrics are NaN - evaluation failed"
        gp_metric["all_metrics_nan"] = True

    # Attach best loss per run (outer loop): useful for logging/comparison without full history
    gp_metric["best_loss_per_run"] = best_loss_per_run

    # Extract hyperparameters from the trained model
    gp_metric["jitter"] = final_jitter

    # Extract raw_noise
    try:
        raw_noise = model.likelihood.raw_noise.detach().cpu()
        gp_metric["raw_noise"] = float(raw_noise.item()) if raw_noise.numel() == 1 else float(
            raw_noise.numpy().flatten()[0]
        )
    except Exception:
        gp_metric["raw_noise"] = np.nan

    # Extract noise (transformed) and compute noise_std
    noise_std_original_scale = None
    try:
        noise_variance = model.likelihood.noise.detach().cpu()
        noise_val = float(noise_variance.item()) if noise_variance.numel() == 1 else float(
            noise_variance.numpy().flatten()[0]
        )
        gp_metric["noise"] = noise_val
        noise_std = float(np.sqrt(noise_val))

        # Convert to original output scale if y was standardized
        if y_train_std is not None:
            if isinstance(y_train_std, dict):
                std_to_use = y_train_std[0] if 0 in y_train_std else list(y_train_std.values())[0]
            else:
                std_to_use = y_train_std.item() if hasattr(y_train_std, "item") else y_train_std
            noise_std_original_scale = noise_std * std_to_use
        else:
            noise_std_original_scale = noise_std
        gp_metric["noise_std"] = float(noise_std_original_scale)
    except Exception as e:
        import logging

        logging.warning(f"Could not extract noise: {e}")
        gp_metric["noise"] = np.nan
        gp_metric["noise_std"] = np.nan

    # Extract outputscale
    try:
        outputscale = model.covar_module.outputscale.detach().cpu()
        gp_metric["outputscale"] = float(outputscale.item()) if outputscale.numel() == 1 else float(
            outputscale.numpy().flatten()[0]
        )
    except Exception:
        gp_metric["outputscale"] = np.nan

    # Extract lengthscales (for ARD kernels)
    try:
        if hasattr(model.covar_module, "base_kernel") and hasattr(
            model.covar_module.base_kernel, "lengthscale"
        ):
            lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu()
            for i, ls_val in enumerate(lengthscales.numpy().flatten()):
                gp_metric[f"cont_lengthscale_{i}"] = float(ls_val)
        elif hasattr(model.covar_module, "lengthscale"):
            lengthscale = model.covar_module.lengthscale.detach().cpu()
            if lengthscale.numel() == 1:
                gp_metric["cont_lengthscale_0"] = float(lengthscale.item())
            else:
                for i, ls_val in enumerate(lengthscale.numpy().flatten()):
                    gp_metric[f"cont_lengthscale_{i}"] = float(ls_val)
    except Exception as e:
        import logging

        logging.warning(f"Could not extract lengthscales: {e}")

    # Add training metadata (LBFGS: report best_iter; keep num_epochs=1)
    gp_metric["num_epochs"] = num_epochs
    if best_iter is not None:
        gp_metric["best_iter"] = int(best_iter)
    else:
        gp_metric["best_iter"] = None

    # Add y_train_mean and y_train_std if provided
    if y_train_mean is not None and y_train_std is not None:
        if isinstance(y_train_mean, dict) and isinstance(y_train_std, dict):
            for source_key, mean_val in y_train_mean.items():
                gp_metric[f"y_train_mean_source_{source_key}"] = float(
                    mean_val.item() if hasattr(mean_val, "item") else mean_val
                )
            for source_key, std_val in y_train_std.items():
                gp_metric[f"y_train_std_source_{source_key}"] = float(
                    std_val.item() if hasattr(std_val, "item") else std_val
                )
        else:
            gp_metric["y_train_mean"] = float(
                y_train_mean.item() if hasattr(y_train_mean, "item") else y_train_mean
            )
            gp_metric["y_train_std"] = float(
                y_train_std.item() if hasattr(y_train_std, "item") else y_train_std
            )

    return gp_metric, y_pred_np, output_std_np
