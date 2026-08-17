"""
run_BO: equivalent of train_eval for Bayesian Optimization.

- run_BO_gp: run one BO loop with a GP surrogate (data and bounds are passed in).
- run_BO_pfn: placeholder for future PFN-based BO.

BX scripts (B1–B9) load data and bounds using load_experimental_data, set up
experiment runs, then call run_BO_gp (or run_BO_pfn) from this module. All
config (optimizer, kernel, etc.) is passed in from the BX script (which reads
from experiments_BO/defaults.py); this module does not import defaults.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch.distributions import Normal

import gpplus
from gpplus.training.eval2 import evaluate_gp_model
from gpplus.utils import set_seed, train_eval_gp
from gpplus.utils.metrics_functions import compute_nis, compute_metrics
from gpplus.utils.standard_scaler import StandardScaler, UniformScaler
from GITBO.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from GITBO.gitbo_bo_helpers import sample_dominant_subspace, compute_acquisition_values_gitbo

try:
    from tabpfn import TabPFNRegressor as SklearnTabPFNRegressor
except ImportError:
    SklearnTabPFNRegressor = None

# No defaults import: BX scripts (B1, B4, etc.) pass in all config from experiments_BO/defaults.py.

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


AcquisitionType = Literal["EI", "TS"]


@dataclass
class run_BO_Result:
    """Container for per-run start_size results."""

    best_y_history: List[float]
    x_history: torch.Tensor
    y_history: torch.Tensor
    best_y_clean_history: List[float]
    y_clean_history: Optional[torch.Tensor]
    test_metrics_history: List[Dict[str, float]]
    nis_history: List[Dict[str, float]]
    n_iterations: int
    # Per-iteration: chosen point (raw), AF value, train time (s), AF time (s)
    x_chosen_history: List[torch.Tensor] = field(default_factory=list)
    af_value_history: List[float] = field(default_factory=list)
    y_pred_mean_history: List[float] = field(default_factory=list)
    train_time_history: List[float] = field(default_factory=list)
    af_time_history: List[float] = field(default_factory=list)


# ---------- Scalers (caller passes x_standardize_method) ----------


def _make_X_scaler(
    x_standardize_method: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Any]:
    """Build X scaler (0=Gaussian, 1=Uniform [0,1], 2=Uniform [-1,1]). Caller passes x_standardize_method from defaults."""
    if x_standardize_method == 0:
        return StandardScaler()
    if x_standardize_method == 1:
        return UniformScaler(scale_to_neg_one=False)
    if x_standardize_method == 2:
        return UniformScaler(scale_to_neg_one=True)
    raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")


class SubcolumnScaler:
    """
    Wrapper that applies a base scaler only to a subset of columns (e.g., continuous dims),
    leaving others (e.g., one-hot categoricals) unchanged.
    """

    def __init__(self, base_scaler: Any, cont_cols: List[int]):
        self.base = base_scaler
        self.cont_cols = list(cont_cols)

    def fit(self, X: torch.Tensor) -> None:
        if len(self.cont_cols) == 0:
            return
        self.base.fit(X[:, self.cont_cols])

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if len(self.cont_cols) == 0:
            return X
        X_out = X.clone()
        X_out[:, self.cont_cols] = self.base.transform(X[:, self.cont_cols])
        return X_out

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        if len(self.cont_cols) == 0:
            return X
        X_out = X.clone()
        X_out[:, self.cont_cols] = self.base.inverse_transform(X[:, self.cont_cols])
        return X_out


# ---------- Bounds helpers (match load_experimental_data) ----------


def get_bounds_wing(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Wing 10D bounds as (2, d) tensor. Matches load_experimental_data."""
    l_bound = torch.tensor(
        [150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025],
        dtype=dtype,
        device=device,
    )
    u_bound = torch.tensor(
        [200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08],
        dtype=dtype,
        device=device,
    )
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_ackley(
    dimensions: int,
    x_bounds: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Ackley box bounds (2, d). Matches load_experimental_data."""
    lb, ub = float(x_bounds[0]), float(x_bounds[1])
    l_bound = torch.full((dimensions,), lb, dtype=dtype, device=device)
    u_bound = torch.full((dimensions,), ub, dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_buckling(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Buckling 4D bounds. Matches load_experimental_data."""
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=dtype, device=device)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_borehole(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Borehole 8D bounds. Matches load_experimental_data."""
    l_bound = torch.tensor(
        [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0],
        dtype=dtype,
        device=device,
    )
    u_bound = torch.tensor(
        [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0],
        dtype=dtype,
        device=device,
    )
    return torch.stack([l_bound, u_bound], dim=0)


def get_bounds_analytic(
    dimensions: int,
    x_bounds: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generic box bounds (2, d) for analytic problems (Rastrigin, Rosenbrock, etc.)."""
    lb, ub = float(x_bounds[0]), float(x_bounds[1])
    l_bound = torch.full((dimensions,), lb, dtype=dtype, device=device)
    u_bound = torch.full((dimensions,), ub, dtype=dtype, device=device)
    return torch.stack([l_bound, u_bound], dim=0)


# ---------- Acquisition helpers ----------


def _sample_sobol_in_bounds(
    n: int,
    bounds: torch.Tensor,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    d = bounds.shape[1]
    base = sobol.draw(n).to(dtype=dtype, device=device)
    l, u = bounds[0], bounds[1]
    return l + (u - l) * base


def _sample_mixed_categorical_candidates(
    n_candidates: int,
    cont_bounds: torch.Tensor,
    cat_class_sizes: List[int],
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    trailing_cols: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample n_candidates in mixed continuous + one-hot categorical space.
    Continuous part is drawn in cont_bounds; categorical part cycles through
    all valid combinations (one 1 per group) with roughly even counts, then
    we assign random continuous values per combination and shuffle.

    cont_bounds: (2, n_cont) with cont_bounds[0] lower, cont_bounds[1] upper.
    cat_class_sizes: e.g. [2, 4, 3] for E, K, I one-hot group sizes.
    Returns: (n_candidates, n_cont + sum(cat_class_sizes)) with valid one-hot.
    """
    n_cont = cont_bounds.shape[1]
    n_combos = 1
    for s in cat_class_sizes:
        n_combos *= s

    # How many samples per categorical combination (even split, remainder spread)
    per_combo = n_candidates // n_combos
    remainder = n_candidates % n_combos

    # Build list of (indices for each cat var) for each of n_candidates
    combo_list: List[Tuple[int, ...]] = []
    combo_idx = 0
    for _ in range(n_combos):
        count = per_combo + (1 if combo_idx < remainder else 0)
        # Decode combo_idx into indices for each categorical
        indices = []
        k = combo_idx
        for size in reversed(cat_class_sizes):
            indices.append(k % size)
            k //= size
        indices = list(reversed(indices))  # now (e_idx, k_idx, i_idx) for [2,4,3]
        for _ in range(count):
            combo_list.append(tuple(indices))
        combo_idx += 1

    # Shuffle so we don't have all combo0 then combo1 (randomness helps)
    perm = torch.randperm(len(combo_list), device=device)
    combo_list = [combo_list[i] for i in perm.cpu().tolist()]

    # Continuous part: one Sobol draw per row in [0,1], then scale to cont_bounds
    sobol_draw = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    lb = cont_bounds[0].to(device=device, dtype=dtype)
    ub = cont_bounds[1].to(device=device, dtype=dtype)
    L = lb + sobol_draw[:, :n_cont] * (ub - lb)

    # Build one-hot blocks per categorical group
    cat_parts = []
    offset = 0
    for group_idx, size in enumerate(cat_class_sizes):
        block = torch.zeros((n_candidates, size), dtype=dtype, device=device)
        for row in range(n_candidates):
            block[row, combo_list[row][group_idx]] = 1.0
        cat_parts.append(block)
    cat_block = torch.cat(cat_parts, dim=1)

    out = torch.cat([L, cat_block], dim=1)
    if trailing_cols is not None:
        # trailing_cols: (1, n_trailing); broadcast to (n_candidates, n_trailing)
        trail = trailing_cols.to(device=device, dtype=dtype).expand(n_candidates, -1)
        out = torch.cat([out, trail], dim=1)
    return out


def _compute_ei(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f_signed: float,
) -> torch.Tensor:
    std = torch.clamp(std, min=1e-12)
    best = torch.as_tensor(best_f_signed, dtype=mean.dtype, device=mean.device)
    improvement = mean - best
    z = improvement / std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    pdf = torch.exp(normal.log_prob(z))
    cdf = normal.cdf(z)
    ei = improvement * cdf + std * pdf
    ei = torch.where(std <= 1e-12, torch.zeros_like(ei), ei)
    return ei


def _compute_ts_samples(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.clamp(std, min=0.0)
    return torch.normal(mean, std)


def _predict_gp_mean_std_with_grad(
    model: Any,
    X_raw: torch.Tensor,
    x_scaler: Optional[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict GP posterior mean/std with autograd enabled.
    """
    X_for_model = x_scaler.transform(X_raw) if x_scaler is not None else X_raw
    model.eval()
    if hasattr(model.likelihood, "set_fidelity_indices"):
        model.likelihood.set_fidelity_indices(X_for_model, is_test=True)
    observed_pred = model.likelihood(model(X_for_model))
    mean = observed_pred.mean.reshape(-1)
    std = observed_pred.stddev.reshape(-1)
    return mean, std


def _predict_tabpfn_signed_mean_std(
    regressor: Any,
    X_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict signed standardized mean/std from sklearn TabPFNRegressor.

    Uses output_type="full" to recover uncertainty required by EI/TS.
    Falls back to quantile-derived std if criterion variance is unavailable.
    """
    full_predictions = regressor.predict(
        X_np,
        output_type="full",
        quantiles=[0.025, 0.975],
    )

    mean_np = full_predictions.get("mean", None)
    if mean_np is None:
        raise RuntimeError(
            "TabPFN full predictions did not include 'mean'; cannot compute BO acquisition."
        )
    mean_signed = torch.as_tensor(mean_np, dtype=dtype, device=device).reshape(-1)

    std_signed: Optional[torch.Tensor] = None
    logits = full_predictions.get("logits", None)
    criterion = full_predictions.get("criterion", None)
    if logits is not None and criterion is not None and hasattr(criterion, "variance"):
        logits_t = torch.as_tensor(logits)
        var = criterion.variance(logits_t)
        std_signed = torch.clamp(
            torch.as_tensor(var, dtype=dtype, device=device).reshape(-1),
            min=1e-12,
        ).sqrt()

    if std_signed is None:
        quantiles = full_predictions.get("quantiles", None)
        if isinstance(quantiles, (list, tuple)) and len(quantiles) >= 2:
            q_lo = torch.as_tensor(quantiles[0], dtype=dtype, device=device).reshape(-1)
            q_hi = torch.as_tensor(quantiles[1], dtype=dtype, device=device).reshape(-1)
            # 95% central interval for a normal variable: q0.975 - q0.025 ~= 2*1.96*sigma
            std_signed = torch.clamp((q_hi - q_lo) / (2.0 * 1.959963984540054), min=1e-12)

    if std_signed is None:
        raise RuntimeError(
            "TabPFN full predictions did not provide variance/quantiles; cannot compute EI/TS."
        )

    return mean_signed, std_signed


def _select_next_point_sampling(
    model,
    bounds: torch.Tensor,
    n_candidates: int,
    acquisition: AcquisitionType,
    best_f_signed: float,
    sign: float,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    x_scaler: Optional[Any] = None,
    cat_class_sizes: Optional[List[int]] = None,
    cont_bounds: Optional[torch.Tensor] = None,
    af_trailing_cols: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    if cat_class_sizes is not None and cont_bounds is not None:
        X_cand = _sample_mixed_categorical_candidates(
            n_candidates, cont_bounds, cat_class_sizes, sobol, dtype, device,
            trailing_cols=af_trailing_cols,
        )
    else:
        X_cand = _sample_sobol_in_bounds(n_candidates, bounds, sobol, dtype, device)
    X_for_model = x_scaler.transform(X_cand) if x_scaler is not None else X_cand
    mean, _, _, std = evaluate_gp_model(model, X_for_model)
    mean_signed = sign * mean.squeeze(-1)
    std = std.squeeze(-1)
    if acquisition.upper() == "EI":
        acq_vals = _compute_ei(mean_signed, std, best_f_signed)
    elif acquisition.upper() == "TS":
        acq_vals = _compute_ts_samples(mean_signed, std)
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition}")
    idx = torch.argmax(acq_vals)
    af_val = float(acq_vals[idx].item())
    return X_cand[idx : idx + 1], af_val


def _optimize_ei_with_scipy(
    model,
    bounds: torch.Tensor,
    n_starts: int,
    best_f_signed: float,
    sign: float,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    x_scaler: Optional[Any] = None,
    af_optimizer_method: str = "L-BFGS-B",
    af_optimizer_options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, float]:
    if minimize is None:
        x_next, af_val = _select_next_point_sampling(
            model=model,
            bounds=bounds,
            n_candidates=n_starts,
            acquisition="EI",
            best_f_signed=best_f_signed,
            sign=sign,
            sobol=sobol,
            dtype=dtype,
            device=device,
            x_scaler=x_scaler,
        )
        return x_next, af_val
    d = bounds.shape[1]
    X_starts = _sample_sobol_in_bounds(n_starts, bounds, sobol, dtype, device).cpu().numpy()
    scipy_bounds = [(float(bounds[0, i].cpu().item()), float(bounds[1, i].cpu().item())) for i in range(d)]

    def neg_ei_with_grad_np(x_np: np.ndarray) -> Tuple[float, np.ndarray]:
        x_base = torch.tensor(x_np, dtype=dtype, device=device, requires_grad=True)
        x = x_base.unsqueeze(0)
        try:
            mean, std = _predict_gp_mean_std_with_grad(
                model=model,
                X_raw=x,
                x_scaler=x_scaler,
            )
            mean_signed = sign * mean
            ei = _compute_ei(mean_signed, std, best_f_signed)
            obj = -ei.squeeze()
            obj.backward()
            grad = x_base.grad.detach().cpu().numpy().astype(np.float64)
            obj_val = float(obj.detach().cpu().item())
            if not np.isfinite(obj_val) or not np.all(np.isfinite(grad)):
                return 1e9, np.zeros_like(x_np, dtype=np.float64)
            return obj_val, grad
        except Exception:
            return 1e9, np.zeros_like(x_np, dtype=np.float64)

    best_x = None
    best_ei = -np.inf
    with torch.no_grad():
        X_starts_t = torch.as_tensor(X_starts, dtype=dtype, device=device)
        X_for_model = x_scaler.transform(X_starts_t) if x_scaler is not None else X_starts_t
        mean_s, _, _, std_s = evaluate_gp_model(model, X_for_model)
        mean_s_signed = sign * mean_s.view(-1)
        std_s = std_s.view(-1)
        ei_starts = _compute_ei(mean_s_signed, std_s, best_f_signed)
    k = max(1, n_starts)
    topk_idx = torch.topk(ei_starts, k).indices.cpu().tolist()
    for idx in topk_idx:
        x0 = X_starts[idx]
        res = minimize(
            neg_ei_with_grad_np,
            x0,
            method=af_optimizer_method,
            jac=True,
            bounds=scipy_bounds,
            options=af_optimizer_options,
        )
        if not res.success:
            continue
        ei_val = -res.fun
        if ei_val > best_ei:
            best_ei = ei_val
            best_x = res.x
    if best_x is None:
        idx = int(torch.argmax(ei_starts).item())
        best_x = X_starts[idx]
        best_ei = float(ei_starts[idx].item())
    return torch.as_tensor(best_x, dtype=dtype, device=device).unsqueeze(0), best_ei


def _optimize_ei_mixed_categorical(
    model: Any,
    cont_bounds: torch.Tensor,
    cat_class_sizes: List[int],
    best_f_signed: float,
    sign: float,
    sobol: SobolEngine,
    dtype: torch.dtype,
    device: torch.device,
    x_scaler: Optional[Any] = None,
    n_inits_per_combo: int = 16,
    af_trailing_cols: Optional[torch.Tensor] = None,
    af_optimizer_method: str = "L-BFGS-B",
    af_optimizer_options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Optimize EI over mixed continuous + one-hot categorical space by enumerating
    categorical combinations and optimizing only the continuous variables for each.
    n_inits_per_combo initializations per combination (e.g. 2 → 48 L-BFGS runs for buckling),
    each optimized in L only; best EI across all runs is returned.
    """
    n_cont = cont_bounds.shape[1]
    n_combos = 1
    for s in cat_class_sizes:
        n_combos *= s

    scipy_bounds_cont = [
        (float(cont_bounds[0, i].cpu().item()), float(cont_bounds[1, i].cpu().item()))
        for i in range(n_cont)
    ]

    # Precompute one-hot rows for each combination
    combo_onehots: List[torch.Tensor] = []
    for combo_idx in range(n_combos):
        indices = []
        k = combo_idx
        for size in reversed(cat_class_sizes):
            indices.append(k % size)
            k //= size
        indices = list(reversed(indices))
        parts = []
        for group_idx, size in enumerate(cat_class_sizes):
            row = torch.zeros(size, dtype=dtype, device=device)
            row[indices[group_idx]] = 1.0
            parts.append(row)
        combo_onehots.append(torch.cat(parts))

    def neg_ei_cont_with_grad_np(c_flat: np.ndarray, combo_idx: int) -> Tuple[float, np.ndarray]:
        c_base = torch.tensor(c_flat, dtype=dtype, device=device, requires_grad=True)
        c = c_base.unsqueeze(0) if c_base.dim() == 1 else c_base
        try:
            onehot = combo_onehots[combo_idx]
            x = torch.cat([c, onehot.unsqueeze(0).expand(c.shape[0], -1)], dim=1)
            if af_trailing_cols is not None:
                x = torch.cat([x, af_trailing_cols.to(device=device, dtype=dtype).expand(x.shape[0], -1)], dim=1)
            mean, std = _predict_gp_mean_std_with_grad(
                model=model,
                X_raw=x,
                x_scaler=x_scaler,
            )
            mean_signed = sign * mean
            ei = _compute_ei(mean_signed, std, best_f_signed)
            obj = -ei.squeeze()
            obj.backward()
            grad = c_base.grad.detach().cpu().numpy().astype(np.float64)
            obj_val = float(obj.detach().cpu().item())
            if not np.isfinite(obj_val) or not np.all(np.isfinite(grad)):
                return 1e9, np.zeros_like(c_flat, dtype=np.float64)
            return obj_val, grad
        except Exception:
            return 1e9, np.zeros_like(c_flat, dtype=np.float64)

    best_x_full: Optional[torch.Tensor] = None
    best_ei = -np.inf

    lb = cont_bounds[0].to(device=device, dtype=dtype)
    ub = cont_bounds[1].to(device=device, dtype=dtype)

    for combo_idx in range(n_combos):
        def obj(c: np.ndarray, ci: int = combo_idx) -> Tuple[float, np.ndarray]:
            return neg_ei_cont_with_grad_np(c, ci)

        for init_idx in range(n_inits_per_combo):
            # Initial continuous point: Sobol draw scaled to cont_bounds
            c0 = sobol.draw(1).to(dtype=dtype, device=device).squeeze(0)[:n_cont]
            c0 = (lb + (ub - lb) * c0).cpu().numpy()

            res = minimize(
                obj,
                c0,
                method=af_optimizer_method,
                jac=True,
                bounds=scipy_bounds_cont,
                options=af_optimizer_options,
            )
            c_opt = res.x
            ei_val = -res.fun
            if ei_val > best_ei:
                best_ei = ei_val
                c_t = torch.as_tensor(c_opt, dtype=dtype, device=device)
                best_x_full = torch.cat([c_t, combo_onehots[combo_idx]]).unsqueeze(0)
                if af_trailing_cols is not None:
                    best_x_full = torch.cat(
                        [best_x_full, af_trailing_cols.to(device=device, dtype=dtype)],
                        dim=1,
                    )

    if best_x_full is None:
        # Fallback: first combo, midpoint of cont_bounds
        c_mid = (cont_bounds[0] + cont_bounds[1]) / 2
        best_x_full = torch.cat(
            [c_mid.to(device=device, dtype=dtype), combo_onehots[0]]
        ).unsqueeze(0)
        if af_trailing_cols is not None:
            best_x_full = torch.cat(
                [best_x_full, af_trailing_cols.to(device=device, dtype=dtype)],
                dim=1,
            )
        x_for_model = x_scaler.transform(best_x_full) if x_scaler is not None else best_x_full
        mean, _, _, std = evaluate_gp_model(model, x_for_model)
        best_ei = float(
            _compute_ei(sign * mean.view(-1), std.view(-1), best_f_signed).item()
        )

    return best_x_full, best_ei


# ---------- Main run API ----------


def _resolve_af_optimizer_options(
    *,
    optimizer_kwargs: Optional[Dict[str, Any]],
    af_optimizer_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build SciPy minimize() options for AF optimization.

    Priority:
      1) af_optimizer_kwargs (explicit AF overrides)
      2) optimizer_kwargs (trainer LBFGS kwargs)
      3) safe defaults

    Supports both SciPy key names and trainer-style names:
      - max_iter -> maxiter
      - max_eval -> maxfun
      - tolerance_grad -> gtol
      - tolerance_change -> ftol
      - history_size -> maxcor
    """
    options: Dict[str, Any] = {"maxiter": 50, "disp": False}
    src = af_optimizer_kwargs if af_optimizer_kwargs is not None else optimizer_kwargs
    if not src:
        return options

    # First pass: accept SciPy-native keys directly.
    scipy_keys = {"maxiter", "maxfun", "gtol", "ftol", "maxcor", "eps", "disp"}
    for k, v in src.items():
        if k in scipy_keys:
            options[k] = v

    # Second pass: map trainer-style keys.
    key_map = {
        "max_iter": "maxiter",
        "max_eval": "maxfun",
        "tolerance_grad": "gtol",
        "tolerance_change": "ftol",
        "history_size": "maxcor",
    }
    for old_key, new_key in key_map.items():
        # SciPy-native key in src takes precedence over mapped trainer key.
        if old_key in src and new_key not in src:
            options[new_key] = src[old_key]

    return options


def run_BO_gp(
    *,
    X_init: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    bounds: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    objective_fn_clean: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    minimization_problem: bool = False,
    acquisition: AcquisitionType = "EI",
    gp_optimize_af: bool = True,
    n_AF_opt: int = 16,
    n_AF_sample: int = 5000,
    max_iter: int = 30,
    patience_no_improve: int = 10,
    run_seed: int = 42,
    num_epochs: int = 1,
    num_inits: int = 16,
    lr: Optional[float] = None,
    convergence_patience: int = 10,
    min_epochs: int = 0,
    min_loss_change: float = 1e-7,
    optimizer_class: Any = None,
    optimizer_kwargs: Optional[Dict] = None,
    af_optimizer_method: str = "L-BFGS-B",
    af_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    initializer_class: Any = None,
    gp_device: str = "cpu",
    gp_dtype: torch.dtype = torch.float64,
    standardize_X: bool = True,
    standardize_y: bool = True,
    x_standardize_method: int = 2,
    save_path: Optional[str] = None,
    run_log_filename: Optional[str] = None,
    verbose: bool = True,
    kernel_module: Any = None,
    mean_module: Any = None,
    likelihood: Any = None,
    log_lbfgs_inner: bool = False,
    bo_test_metrics: bool = True,
    cat_class_sizes: Optional[List[int]] = None,
    cont_bounds: Optional[torch.Tensor] = None,
    af_trailing_cols: Optional[torch.Tensor] = None,
    cont_cols: Optional[List[int]] = None,
) -> "run_BO_Result":
    """
    Run one BO loop with a GP surrogate (equivalent of train_eval_gp for BO).

    Caller (BX script) is responsible for loading data and bounds via load_experimental_data.
    When standardize_X/standardize_y are True, X and y are scaled for GP training; bounds and
    objective remain in raw space; chosen points and logs are in raw space.

    For mixed categorical (one-hot) + continuous: pass cat_class_sizes (e.g. [2, 4, 3]) and
    cont_bounds (2 x n_cont). Then: (1) If EI + gp_optimize_af, we optimize EI by enumerating
    categorical combinations (one init per combo) and optimizing only the continuous variables.
    (2) Otherwise we sample AF candidates with valid one-hot and even combo coverage.
    """
    device = torch.device(gp_device)
    dtype = gp_dtype
    if optimizer_kwargs is None:
        raise ValueError("optimizer_kwargs must be passed in by the BX script (e.g. from defaults.TRAINER_OPTIMIZER_KWARGS)")
    set_seed(run_seed)

    if objective_fn_clean is None:
        objective_fn_clean = objective_fn

    X_train_raw = X_init.to(device=device, dtype=dtype)
    y_train_raw = objective_fn(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    y_train_clean_raw = (
        objective_fn_clean(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    )
    X_test_raw = X_test.to(device=device, dtype=dtype)
    y_test_raw = y_test.to(device=device, dtype=dtype).reshape(-1)

    # Fit scalers on initial data
    x_scaler: Optional[Any] = None
    y_scaler: Optional[Any] = None
    if standardize_X:
        base_scaler = _make_X_scaler(x_standardize_method, device, dtype)
        # If cont_cols are provided, only scale those; otherwise scale all columns.
        if cont_cols is not None:
            x_scaler = SubcolumnScaler(base_scaler, cont_cols)
        else:
            x_scaler = base_scaler
        x_scaler.fit(X_train_raw)
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_raw.unsqueeze(1))
        # avoid div-by-zero
        if (y_scaler.std is not None) and (y_scaler.std.squeeze() == 0).all():
            y_scaler.std = torch.ones_like(y_scaler.std, device=device, dtype=dtype)

    X_train = x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
    y_train = y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1) if y_scaler is not None else y_train_raw
    X_test = x_scaler.transform(X_test_raw) if x_scaler is not None else X_test_raw
    y_test = y_test_raw  # train_eval_gp expects raw y_test for metrics; we pass y_train_mean/std for denorm

    y_train_mean = y_scaler.mean.squeeze() if y_scaler is not None else None
    y_train_std = y_scaler.std.squeeze() if y_scaler is not None else None

    sobol = SobolEngine(X_train_raw.shape[1], scramble=True, seed=run_seed)
    # Minimize objective → maximize (-y) → sign = -1. Maximize objective → sign = 1.
    sign = -1.0 if minimization_problem else 1.0
    af_optimizer_options = _resolve_af_optimizer_options(
        optimizer_kwargs=optimizer_kwargs,
        af_optimizer_kwargs=af_optimizer_kwargs,
    )

    best_y_history: List[float] = []
    best_y_clean_history: List[float] = []
    test_metrics_history: List[Dict[str, float]] = []
    nis_history: List[Dict[str, float]] = []
    x_chosen_history: List[torch.Tensor] = []
    af_value_history: List[float] = []
    y_pred_mean_history: List[float] = []
    train_time_history: List[float] = []
    af_time_history: List[float] = []
    no_improve = 0

    for _ in range(max_iter):
        model = gpplus.models.GPR(
            X_train,
            y_train,
            kernel_module=kernel_module,
            mean_module=mean_module,
            likelihood=likelihood,
        )
        callbacks: List = []

        t_train = time.perf_counter()
        gp_metric, y_pred_test, output_std_test, _ = train_eval_gp(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            seed=run_seed,
            num_inits=num_inits,
            lr=lr,
            convergence_patience=convergence_patience,
            min_epochs=min_epochs,
            min_loss_change=min_loss_change,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            source_cols=None,
            trainer_info=False,
            callback_save_path=None,
            callbacks=callbacks,
            log_lbfgs_inner=log_lbfgs_inner,
        )
        train_time_history.append(time.perf_counter() - t_train)

        if bo_test_metrics and isinstance(gp_metric, dict):
            test_metrics_history.append(gp_metric)
        if bo_test_metrics and y_pred_test is not None and output_std_test is not None:
            try:
                nis_dict = compute_nis(
                    y_true=y_test,
                    y_hat=y_pred_test,
                    output_std=output_std_test,
                )
                nis_history.append(nis_dict)
            except Exception:
                nis_history.append({})

        # Best so far: raw for logging, scaled for AF (model predicts in scaled space when standardize_y)
        y_train_raw_cur = (
            y_scaler.inverse_transform(y_train.unsqueeze(1)).squeeze(1)
            if y_scaler is not None
            else y_train
        )
        y_train_clean_raw_cur = y_train_clean_raw
        y_signed_raw = sign * y_train_raw_cur
        best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
        best_raw_val = y_train_raw_cur[best_idx].item()
        best_y_history.append(best_raw_val)

        y_signed_clean = sign * y_train_clean_raw_cur
        _, best_clean_idx = torch.max(y_signed_clean, dim=0)
        best_clean_raw_val = y_train_clean_raw_cur[best_clean_idx].item()
        best_y_clean_history.append(best_clean_raw_val)
        best_f_signed_for_af = float((sign * y_train).max().item())  # scaled space for EI/TS

        if acquisition.upper() not in {"EI", "TS"}:
            raise ValueError(f"Unsupported acquisition function: {acquisition}")

        t_af = time.perf_counter()
        if (
            cat_class_sizes is not None
            and cont_bounds is not None
            and acquisition.upper() == "EI"
            and gp_optimize_af
        ):
            # Mixed categorical: optimize EI per combination (one init per combo, optimize L only)
            x_next_raw, af_val = _optimize_ei_mixed_categorical(
                model=model,
                cont_bounds=cont_bounds,
                cat_class_sizes=cat_class_sizes,
                best_f_signed=best_f_signed_for_af,
                sign=sign,
                sobol=sobol,
                dtype=dtype,
                device=device,
                x_scaler=x_scaler,
                n_inits_per_combo=n_AF_opt,
                af_trailing_cols=af_trailing_cols,
                af_optimizer_method=af_optimizer_method,
                af_optimizer_options=af_optimizer_options,
            )
        elif acquisition.upper() == "EI" and gp_optimize_af:
            x_next_raw, af_val = _optimize_ei_with_scipy(
                model=model,
                bounds=bounds,
                n_starts=n_AF_opt,
                best_f_signed=best_f_signed_for_af,
                sign=sign,
                sobol=sobol,
                dtype=dtype,
                device=device,
                x_scaler=x_scaler,
                af_optimizer_method=af_optimizer_method,
                af_optimizer_options=af_optimizer_options,
            )
        else:
            n_cand = n_AF_sample if n_AF_sample is not None else 5000
            x_next_raw, af_val = _select_next_point_sampling(
                model=model,
                bounds=bounds,
                n_candidates=n_cand,
                acquisition=acquisition.upper(),
                best_f_signed=best_f_signed_for_af,
                sign=sign,
                sobol=sobol,
                dtype=dtype,
                device=device,
                x_scaler=x_scaler,
                cat_class_sizes=cat_class_sizes,
                cont_bounds=cont_bounds,
                af_trailing_cols=af_trailing_cols,
            )
        af_time_history.append(time.perf_counter() - t_af)

        x_chosen_history.append(x_next_raw.detach().cpu().clone())
        af_value_history.append(af_val)
        x_next_for_model = x_scaler.transform(x_next_raw) if x_scaler is not None else x_next_raw
        pred_mean_std, _, _, _ = evaluate_gp_model(model, x_next_for_model)
        pred_mean_std = pred_mean_std.reshape(-1).to(device=device, dtype=dtype)
        if y_scaler is not None:
            pred_mean_raw = y_scaler.inverse_transform(pred_mean_std.unsqueeze(1)).squeeze(1)
        else:
            pred_mean_raw = pred_mean_std
        y_pred_mean_history.append(float(pred_mean_raw[0].item()))

        if verbose:
            iter_idx = len(best_y_history)
            print(
                f"  BO iter {iter_idx:02d} | "
                f"best_y={best_y_history[-1]:.4f} | "
                f"af={af_val:.4f} | "
                f"train_s={train_time_history[-1]:.4f} | "
                f"af_s={af_time_history[-1]:.4f}"
            )
            print("  " + "-" * 70)

        y_next = objective_fn(x_next_raw).to(device=device, dtype=dtype).reshape(-1)
        y_next_clean = (
            objective_fn_clean(x_next_raw).to(device=device, dtype=dtype).reshape(-1)
        )
        new_signed = sign * y_next[0]
        if new_signed > best_signed_val + 1e-8:
            no_improve = 0
        else:
            no_improve += 1

        # Append in raw space for objective and history; scaled for GP
        X_train_raw = torch.cat([X_train_raw, x_next_raw], dim=0)
        y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
        y_train_clean_raw = torch.cat(
            [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
        )
        X_train = x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
        y_train = y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1) if y_scaler is not None else y_train_raw

        # Early stopping on lack of improvement only when patience_no_improve is set and > 0.
        # If patience_no_improve is None or 0, this condition is disabled (use max_iter only).
        if (
            patience_no_improve is not None
            and patience_no_improve > 0
            and no_improve >= patience_no_improve
        ):
            break

    result = run_BO_Result(
        best_y_history=best_y_history,
        best_y_clean_history=best_y_clean_history,
        x_history=X_train_raw.detach().cpu(),
        y_history=y_train_raw.detach().cpu(),
        y_clean_history=y_train_clean_raw.detach().cpu(),
        test_metrics_history=test_metrics_history,
        nis_history=nis_history,
        n_iterations=len(best_y_history),
        x_chosen_history=x_chosen_history,
        af_value_history=af_value_history,
        y_pred_mean_history=y_pred_mean_history,
        train_time_history=train_time_history,
        af_time_history=af_time_history,
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_data: Dict[str, Any] = {
            "run_seed": run_seed,
            "n_iterations": result.n_iterations,
            "x_chosen": [t.tolist() for t in result.x_chosen_history],
            "af_values": result.af_value_history,
            "af_optimizer_method": af_optimizer_method,
            "af_optimizer_options": af_optimizer_options,
            "y_pred_mean_history": result.y_pred_mean_history,
            "train_time_s": result.train_time_history,
            "af_time_s": result.af_time_history,
            "best_y_history": result.best_y_history,
            "best_y_clean_history": result.best_y_clean_history,
            "final_best_y": result.best_y_history[-1]
            if result.best_y_history
            else None,
            "final_best_y_clean": result.best_y_clean_history[-1]
            if result.best_y_clean_history
            else None,
        }
        log_filename = run_log_filename if run_log_filename else "bo_run_log.json"
        if not log_filename.endswith(".json"):
            log_filename += ".json"
        log_file = os.path.join(save_path, log_filename)
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    return result


def _scale_to_unit(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Scale X from bounds to [0,1]."""
    lb = bounds[0].to(X.device, dtype=X.dtype)
    ub = bounds[1].to(X.device, dtype=X.dtype)
    return (X - lb) / (ub - lb + 1e-9)


def _scale_from_unit(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Scale X from [0,1] back to bounds."""
    lb = bounds[0].to(X.device, dtype=X.dtype)
    ub = bounds[1].to(X.device, dtype=X.dtype)
    return lb + X * (ub - lb)


def run_BO_pfn(
    *,
    X_init: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    bounds: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    minimization_problem: bool = False,
    acquisition: AcquisitionType = "EI",
    n_AF_sample: int = 5000,
    max_iter: int = 30,
    patience_no_improve: int = 10,
    run_seed: int = 42,
    gp_device: str = "cpu",
    gp_dtype: torch.dtype = torch.float64,
    save_path: Optional[str] = None,
    run_log_filename: Optional[str] = None,
    verbose: bool = True,
    pfn_device: Optional[str] = None,
    pfn_dtype: torch.dtype = torch.float32,
    bo_test_metrics: bool = True,
    gi_pfn: bool = False,
    objective_fn_clean: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    standardize_X: bool = True,
    standardize_y: bool = True,
    x_standardize_method: int = 2,
    cont_cols: Optional[List[int]] = None,
) -> run_BO_Result:
    """
    Run one BO loop with a PFN surrogate.

    If gi_pfn=True: use gradient-informed (GITBO) method with VanillaDirectTabPFNRegressor.
    If gi_pfn=False: use standard TabPFNRegressor (Sklearn) + 5k Sobol candidates.
    """
    device = torch.device(gp_device)
    dtype = gp_dtype
    if pfn_device is None:
        pfn_device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(run_seed)

    if objective_fn_clean is None:
        objective_fn_clean = objective_fn

    X_train_raw = X_init.to(device=device, dtype=dtype)
    y_train_raw = objective_fn(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    y_train_clean_raw = (
        objective_fn_clean(X_train_raw).to(device=device, dtype=dtype).reshape(-1)
    )
    X_test_raw = X_test.to(device=device, dtype=dtype)
    y_test_raw = y_test.to(device=device, dtype=dtype).reshape(-1)

    # Fit scalers on initial data so PFN uses the same preprocessing as GP.
    x_scaler: Optional[Any] = None
    y_scaler: Optional[Any] = None
    if standardize_X:
        base_scaler = _make_X_scaler(x_standardize_method, device, dtype)
        if cont_cols is not None:
            x_scaler = SubcolumnScaler(base_scaler, cont_cols)
        else:
            x_scaler = base_scaler
        x_scaler.fit(X_train_raw)
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_raw.unsqueeze(1))
        if (y_scaler.std is not None) and (y_scaler.std.squeeze() == 0).all():
            y_scaler.std = torch.ones_like(y_scaler.std, device=device, dtype=dtype)

    X_train = x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
    y_train = (
        y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1)
        if y_scaler is not None
        else y_train_raw
    )

    sobol = SobolEngine(X_train_raw.shape[1], scramble=True, seed=run_seed)
    sign = -1.0 if minimization_problem else 1.0

    best_y_history: List[float] = []
    best_y_clean_history: List[float] = []
    test_metrics_history: List[Dict[str, float]] = []
    nis_history: List[Dict[str, float]] = []
    x_chosen_history: List[torch.Tensor] = []
    af_value_history: List[float] = []
    y_pred_mean_history: List[float] = []
    train_time_history: List[float] = []
    af_time_history: List[float] = []
    no_improve = 0

    bounds_dev = bounds.to(device=device, dtype=dtype)

    if gi_pfn:
        # --- GITBO path: gradient-informed subspace + VanillaDirectTabPFNRegressor ---
        from GITBO.gitbo_bo_helpers import (
            compute_acquisition_values_gitbo,
            sample_dominant_subspace,
        )

        N_PENDING = n_AF_sample if n_AF_sample else 5000
        N_CANDIDATES = 1
        rank_r = 15
        scale = 0.2
        tkwargs = {"device": torch.device(pfn_device), "dtype": pfn_dtype}
        grad_est = None

        for _ in range(max_iter):
            # Best so far in raw space (for logging and early stopping)
            y_signed_raw = sign * y_train_raw
            best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
            best_raw_val = y_train_raw[best_idx].item()
            best_y_history.append(best_raw_val)

            y_signed_clean = sign * y_train_clean_raw
            _, best_clean_idx = torch.max(y_signed_clean, dim=0)
            best_clean_raw_val = y_train_clean_raw[best_clean_idx].item()
            best_y_clean_history.append(best_clean_raw_val)

            if acquisition.upper() not in {"EI", "TS"}:
                raise ValueError(f"Unsupported acquisition for PFN: {acquisition}")

            t_af = time.perf_counter()

            # Use the same standardized X/Y as GP for PFN training.
            trained_X = X_train.to(**tkwargs)
            trained_Y = (sign * y_train).reshape(-1, 1).to(**tkwargs)

            if grad_est is None:
                total_points = N_CANDIDATES * N_PENDING
                # Sample candidates in raw space inside bounds then standardize for PFN.
                X_pen_raw_flat = _sample_sobol_in_bounds(
                    total_points, bounds_dev, sobol, dtype, device
                )
                X_pen_flat = (
                    x_scaler.transform(X_pen_raw_flat)
                    if x_scaler is not None
                    else X_pen_raw_flat
                )
                X_pen = X_pen_flat.to(**tkwargs).view(
                    N_PENDING, N_CANDIDATES, X_train_raw.shape[1]
                )
            else:
                X_pen = sample_dominant_subspace(
                    trained_X,
                    trained_Y,
                    X_train_raw.shape[1],
                    grad_est,
                    sobol,
                    rank_r=rank_r,
                    n_samples=N_PENDING,
                    N_CANDIDATES=N_CANDIDATES,
                    scale=scale,
                    GI_SUBSPACE=True,
                    tkwargs=tkwargs,
                )

            ACQ, _, grad_est = compute_acquisition_values_gitbo(
                acquisition=acquisition,
                DIM=X_train_raw.shape[1],
                N_PENDING=N_PENDING,
                N_CANDIDATES=N_CANDIDATES,
                trained_X=trained_X,
                trained_Y=trained_Y,
                X_pen=X_pen,
                gpu_device=pfn_device,
                tkwargs=tkwargs,
            )

            X_pen_perm = X_pen.permute(1, 0, 2)
            ACQ_perm = ACQ.permute(1, 0)
            best_cand_idx = torch.argmax(ACQ_perm, dim=1)
            best_candidate_std = X_pen_perm[0, best_cand_idx[0], :].unsqueeze(0)
            # Map standardized candidate back to raw space for objective evaluation.
            if x_scaler is not None:
                x_next_raw = x_scaler.inverse_transform(
                    best_candidate_std.to(device=device, dtype=dtype)
                )
            else:
                x_next_raw = best_candidate_std.to(device=device, dtype=dtype)

            af_time_history.append(time.perf_counter() - t_af)
            train_time_history.append(0.0)
            x_chosen_history.append(x_next_raw.detach().cpu().clone())
            af_value_history.append(float(ACQ_perm[0, best_cand_idx[0]].item()))
            # PFN predicts signed standardized y during acquisition; map back to raw y.
            regressor_pred = VanillaDirectTabPFNRegressor(device=pfn_device)
            x_next_std = (
                x_scaler.transform(x_next_raw) if x_scaler is not None else x_next_raw
            )
            X_train_pfn = X_train.to(device=pfn_device, dtype=pfn_dtype)
            x_next_pfn = x_next_std.to(device=pfn_device, dtype=pfn_dtype)
            y_train_signed_pfn = (sign * y_train).to(device=pfn_device, dtype=pfn_dtype)
            X_concat = torch.cat([X_train_pfn.unsqueeze(1), x_next_pfn.unsqueeze(1)], dim=0)
            Y_train_time = y_train_signed_pfn.view(-1, 1)
            Y_pad = torch.zeros(1, 1, device=pfn_device, dtype=pfn_dtype)
            Y_full = torch.cat([Y_train_time, Y_pad], dim=0).unsqueeze(1)
            with torch.no_grad():
                out = regressor_pred.forward(
                    X_concat,
                    Y_full,
                    single_eval_pos=X_train.shape[0],
                )
                logits = out["standard"]
                pred_mean_signed_std = regressor_pred.predict_mean(logits)[X_train.shape[0] :, 0]
            pred_mean_std = (pred_mean_signed_std * sign).to(device=device, dtype=dtype)
            if y_scaler is not None:
                pred_mean_raw = y_scaler.inverse_transform(pred_mean_std.unsqueeze(1)).squeeze(1)
            else:
                pred_mean_raw = pred_mean_std
            y_pred_mean_history.append(float(pred_mean_raw[0].item()))

            if verbose:
                iter_idx = len(best_y_history)
                print(
                    f"  BO iter {iter_idx:02d} | "
                    f"best_y={best_y_history[-1]:.4f} | "
                    f"af={af_value_history[-1]:.4f} | "
                    f"train_s=0.0000 | af_s={af_time_history[-1]:.4f}"
                )
                print("  " + "-" * 70)

            y_next = objective_fn(x_next_raw.to(device=device, dtype=dtype)).to(
                device=device, dtype=dtype
            ).reshape(-1)
            y_next_clean = (
                objective_fn_clean(x_next_raw.to(device=device, dtype=dtype))
                .to(device=device, dtype=dtype)
                .reshape(-1)
            )
            new_signed = sign * y_next[0]
            if new_signed > best_signed_val + 1e-8:
                no_improve = 0
            else:
                no_improve += 1

            X_train_raw = torch.cat(
                [X_train_raw, x_next_raw.to(device=device, dtype=dtype)], dim=0
            )
            y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
            y_train_clean_raw = torch.cat(
                [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
            )

            # Update standardized tensors with fixed scalers.
            X_train = (
                x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
            )
            y_train = (
                y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1)
                if y_scaler is not None
                else y_train_raw
            )

            if (
                patience_no_improve is not None
                and patience_no_improve > 0
                and no_improve >= patience_no_improve
            ):
                break

    else:
        # --- Vanilla PFN path: TabPFNRegressor (Sklearn) + 5k Sobol ---
        if SklearnTabPFNRegressor is None:
            raise ImportError(
                "gi_pfn=False requires tabpfn package. Install with: pip install tabpfn"
            )

        regressor = SklearnTabPFNRegressor(device=pfn_device)
        n_cand = n_AF_sample if n_AF_sample else 5000

        for _ in range(max_iter):
            # Best so far in raw space (for logging and early stopping)
            y_signed_raw = sign * y_train_raw
            best_signed_val, best_idx = torch.max(y_signed_raw, dim=0)
            best_raw_val = y_train_raw[best_idx].item()
            best_y_history.append(best_raw_val)

            y_signed_clean = sign * y_train_clean_raw
            _, best_clean_idx = torch.max(y_signed_clean, dim=0)
            best_clean_raw_val = y_train_clean_raw[best_clean_idx].item()
            best_y_clean_history.append(best_clean_raw_val)
            best_f_signed_for_af = float((sign * y_train).max().item())

            if acquisition.upper() not in {"EI", "TS"}:
                raise ValueError(f"Unsupported acquisition function for PFN: {acquisition}")

            t_af = time.perf_counter()
            X_cand_raw = _sample_sobol_in_bounds(
                n=n_cand,
                bounds=bounds_dev,
                sobol=sobol,
                dtype=dtype,
                device=device,
            )
            # Standardize train and candidate inputs for PFN.
            X_cand = (
                x_scaler.transform(X_cand_raw)
                if x_scaler is not None
                else X_cand_raw
            )

            # TabPFNRegressor fit/predict: fit on train, predict on candidates
            X_train_np = X_train.cpu().numpy()
            y_train_signed_np = (sign * y_train).cpu().numpy()
            X_cand_np = X_cand.cpu().numpy()

            regressor.fit(X_train_np, y_train_signed_np)
            mean_signed_std, std_signed_std = _predict_tabpfn_signed_mean_std(
                regressor=regressor,
                X_np=X_cand_np,
                device=device,
                dtype=dtype,
            )
            if acquisition.upper() == "EI":
                acq_vals = _compute_ei(
                    mean_signed_std,
                    std_signed_std,
                    best_f_signed=best_f_signed_for_af,
                )
            elif acquisition.upper() == "TS":
                acq_vals = _compute_ts_samples(mean_signed_std, std_signed_std)
            else:
                raise ValueError(f"Unsupported acquisition function for PFN: {acquisition}")

            idx = int(torch.argmax(acq_vals).item())
            af_val = float(acq_vals[idx].item())
            x_next_raw = X_cand_raw[idx : idx + 1]
            af_time_history.append(time.perf_counter() - t_af)
            train_time_history.append(0.0)

            x_chosen_history.append(x_next_raw.detach().cpu().clone())
            af_value_history.append(af_val)
            # Store PFN posterior-mean prediction at chosen point (signed, standardized).
            pred_mean_signed_std = mean_signed_std[idx : idx + 1]
            pred_mean_std = pred_mean_signed_std * sign
            if y_scaler is not None:
                pred_mean_raw = y_scaler.inverse_transform(pred_mean_std.unsqueeze(1)).squeeze(1)
            else:
                pred_mean_raw = pred_mean_std
            y_pred_mean_history.append(float(pred_mean_raw[0].item()))

            if verbose:
                iter_idx = len(best_y_history)
                print(
                    f"  BO iter {iter_idx:02d} | "
                    f"best_y={best_y_history[-1]:.4f} | "
                    f"af={af_val:.4f} | "
                    f"train_s=0.0000 | af_s={af_time_history[-1]:.4f}"
                )
                print("  " + "-" * 70)

            y_next = objective_fn(x_next_raw.to(device=device, dtype=dtype)).to(
                device=device, dtype=dtype
            ).reshape(-1)
            y_next_clean = (
                objective_fn_clean(x_next_raw.to(device=device, dtype=dtype))
                .to(device=device, dtype=dtype)
                .reshape(-1)
            )
            new_signed = sign * y_next[0]
            if new_signed > best_signed_val + 1e-8:
                no_improve = 0
            else:
                no_improve += 1

            X_train_raw = torch.cat(
                [X_train_raw, x_next_raw.to(device=device, dtype=dtype)], dim=0
            )
            y_train_raw = torch.cat([y_train_raw, y_next.reshape(-1)], dim=0)
            y_train_clean_raw = torch.cat(
                [y_train_clean_raw, y_next_clean.reshape(-1)], dim=0
            )

            # Update standardized tensors with fixed scalers.
            X_train = (
                x_scaler.transform(X_train_raw) if x_scaler is not None else X_train_raw
            )
            y_train = (
                y_scaler.transform(y_train_raw.unsqueeze(1)).squeeze(1)
                if y_scaler is not None
                else y_train_raw
            )

            if (
                patience_no_improve is not None
                and patience_no_improve > 0
                and no_improve >= patience_no_improve
            ):
                break

    # Final test metrics using PFN surrogate, if enabled.
    if bo_test_metrics:
        try:
            N_train_final = X_train_raw.shape[0]
            N_test = X_test_raw.shape[0]
            t_pred = time.perf_counter()

            if gi_pfn:
                # Use VanillaDirectTabPFNRegressor for test metrics
                regressor_metrics = VanillaDirectTabPFNRegressor(device=pfn_device)
                X_train_pfn = X_train.to(device=pfn_device, dtype=pfn_dtype)
                X_test_std = (
                    x_scaler.transform(X_test_raw)
                    if x_scaler is not None
                    else X_test_raw
                )
                X_test_pfn = X_test_std.to(device=pfn_device, dtype=pfn_dtype)
                X_train_seq = X_train_pfn.unsqueeze(1)
                X_test_seq = X_test_pfn.unsqueeze(1)
                X_concat = torch.cat([X_train_seq, X_test_seq], dim=0)
                y_train_signed_pfn = (sign * y_train).to(
                    device=pfn_device, dtype=pfn_dtype
                )
                Y_train_time = y_train_signed_pfn.view(-1, 1)
                Y_pad = torch.zeros(N_test, 1, device=pfn_device, dtype=pfn_dtype)
                Y_time = torch.cat([Y_train_time, Y_pad], dim=0)
                Y_full = Y_time.unsqueeze(1)
                single_eval_pos = N_train_final
                with torch.no_grad():
                    out = regressor_metrics.forward(X_concat, Y_full, single_eval_pos)
                    logits = out["standard"]
                    mean_all = regressor_metrics.predict_mean(logits)
                    var_all = regressor_metrics.predict_variance(logits)
                    std_all = torch.clamp(var_all, min=1e-8).sqrt()
                # PFN predictions are in standardized-y space; map back to raw like GP.
                mean_test_signed_std = mean_all[single_eval_pos:, 0]
                std_test_std = std_all[single_eval_pos:, 0]
            else:
                # Use TabPFNRegressor predict (no uncertainty from sklearn API)
                X_train_np = X_train.cpu().numpy()
                X_test_std = (
                    x_scaler.transform(X_test_raw)
                    if x_scaler is not None
                    else X_test_raw
                )
                X_test_np = X_test_std.cpu().numpy()
                y_train_signed_np = (sign * y_train).cpu().numpy()
                regressor.fit(X_train_np, y_train_signed_np)
                mean_test_signed_std = torch.tensor(
                    regressor.predict(X_test_np), dtype=pfn_dtype, device=pfn_device
                )
                std_test_std = torch.ones_like(mean_test_signed_std, device=pfn_device) * 1e-6

            pred_time = time.perf_counter() - t_pred

            # Inverse-transform PFN outputs back to raw y space, matching GP behavior.
            # mean/std currently in "signed, standardized" space: sign * y_std.
            y_hat_signed_std = mean_test_signed_std.to(device=device, dtype=dtype)
            std_test_std = std_test_std.to(device=device, dtype=dtype)
            y_hat_std = y_hat_signed_std * sign  # remove sign
            if y_scaler is not None:
                y_hat_test = y_scaler.inverse_transform(
                    y_hat_std.unsqueeze(1)
                ).squeeze(1)
                # Scale predictive std back using training std (broadcast if needed).
                y_train_std_val = y_scaler.std.to(device=device, dtype=dtype).squeeze()
                std_test = std_test_std * y_train_std_val
            else:
                y_hat_test = y_hat_std
                std_test = std_test_std

            metrics_dict = compute_metrics(
                y_true=y_test_raw,
                y_hat=y_hat_test,
                output_std=std_test,
                training_time=0.0,
                prediction_time=pred_time,
            )
            test_metrics_history.append(metrics_dict)

            nis_dict = compute_nis(
                y_true=y_test_raw,
                y_hat=y_hat_test,
                output_std=std_test,
            )
            nis_history.append(nis_dict)
        except Exception:
            # If anything goes wrong, append empty dicts so downstream code can still index [-1].
            test_metrics_history.append({})
            nis_history.append({})

    result = run_BO_Result(
        best_y_history=best_y_history,
        best_y_clean_history=best_y_clean_history,
        x_history=X_train_raw.detach().cpu(),
        y_history=y_train_raw.detach().cpu(),
        y_clean_history=y_train_clean_raw.detach().cpu(),
        test_metrics_history=test_metrics_history,
        nis_history=nis_history,
        n_iterations=len(best_y_history),
        x_chosen_history=x_chosen_history,
        af_value_history=af_value_history,
        y_pred_mean_history=y_pred_mean_history,
        train_time_history=train_time_history,
        af_time_history=af_time_history,
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_data: Dict[str, Any] = {
            "run_seed": run_seed,
            "n_iterations": result.n_iterations,
            "x_chosen": [t.tolist() for t in result.x_chosen_history],
            "af_values": result.af_value_history,
            "y_pred_mean_history": result.y_pred_mean_history,
            "train_time_s": result.train_time_history,
            "af_time_s": result.af_time_history,
            "best_y_history": result.best_y_history,
            "best_y_clean_history": result.best_y_clean_history,
            "final_best_y": result.best_y_history[-1]
            if result.best_y_history
            else None,
            "final_best_y_clean": result.best_y_clean_history[-1]
            if result.best_y_clean_history
            else None,
        }
        log_filename = run_log_filename if run_log_filename else "bo_run_log_pfn.json"
        if not log_filename.endswith(".json"):
            log_filename += ".json"
        log_file = os.path.join(save_path, log_filename)
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    return result
