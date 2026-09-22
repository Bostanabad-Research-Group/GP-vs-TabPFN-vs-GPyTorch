"""Shared experiment/data metadata for GP vs TabPFN JSON exports."""

from __future__ import annotations

from typing import Any


def scalar_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item() if value.numel() == 1 else value.squeeze().item())
    if isinstance(value, dict):
        return {k: scalar_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_value(v) for v in value]
    return value


def experiment_data_info(
    *,
    cat_cols=None,
    cont_cols=None,
    source_cols=None,
    qual_dict=None,
    input_dim: int,
    train_samples: int,
    test_samples: int,
    standardize_X: bool,
    standardize_y: bool,
    x_standardize_method: int,
    X_scaling_type: str,
    y_train_mean,
    y_train_std,
    y_test_mean: float,
    y_test_std: float,
    num_runs: int,
    seed: int,
    seed_trainer,
    noise_train: float | None = None,
    noise_test: float | None = None,
    noise_type: str | None = None,
    standardize_y_log_scale: bool | None = None,
    preprocess_pfn: bool | None = None,
    pfn_dtype=None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Data/experiment fields shared across GP and PFN runs.

    Written to ``pfn_model_info`` even when ``run_models='pfn'`` so PFN-only JSON
    can be cross-checked against GP exports.
    """
    info: dict[str, Any] = {
        "cat_cols": list(cat_cols) if cat_cols is not None else [],
        "cont_cols": list(cont_cols) if cont_cols is not None else [],
        "source_cols": source_cols if source_cols is not None else [],
        "qual_dict": qual_dict if qual_dict is not None else {},
        "input_dim": int(input_dim),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "standardize_X": bool(standardize_X),
        "standardize_y": bool(standardize_y),
        "x_standardize_method": int(x_standardize_method),
        "X_scaling_type": X_scaling_type,
        "y_train_mean": scalar_value(y_train_mean),
        "y_train_std": scalar_value(y_train_std),
        "y_test_mean": float(y_test_mean),
        "y_test_std": float(y_test_std),
        "num_runs": int(num_runs),
        "seed": seed,
        "seed_trainer": seed_trainer,
    }
    if noise_train is not None:
        info["noise_train"] = float(noise_train)
    if noise_test is not None:
        info["noise_test"] = float(noise_test)
    if noise_type is not None:
        info["noise_type"] = noise_type
    if standardize_y_log_scale is not None:
        info["standardize_y_log_scale"] = bool(standardize_y_log_scale)
    if preprocess_pfn is not None:
        info["preprocess_pfn"] = bool(preprocess_pfn)
    if pfn_dtype is not None:
        info["pfn_dtype"] = str(pfn_dtype)
    for key, value in extra.items():
        if value is not None:
            info[key] = scalar_value(value)
    return info


def pfn_model_info(regressor, *, experiment_data: dict[str, Any]) -> dict[str, Any]:
    """TabPFN settings plus shared experiment/data metadata."""
    device = getattr(regressor, "device_", getattr(regressor, "device", None))
    forced_dtype = getattr(regressor, "forced_inference_dtype_", None)
    return {
        **experiment_data,
        "model_path": getattr(regressor, "model_path", None),
        "fit_mode": getattr(regressor, "fit_mode", None),
        "device": str(device) if device is not None else None,
        "inference_precision": getattr(regressor, "inference_precision", None),
        "random_state": getattr(regressor, "random_state", None),
        "use_autocast": getattr(regressor, "use_autocast_", None),
        "forced_inference_dtype": str(forced_dtype) if forced_dtype is not None else None,
    }
