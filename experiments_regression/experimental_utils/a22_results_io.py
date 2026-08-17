"""Load/save A22 1D experiment artifacts and regenerate plots without a full re-run when possible."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def find_gpvpfn_json(result_dir: Path) -> Path | None:
    matches = sorted(result_dir.glob("gpVpfn_*.json"))
    return matches[0] if matches else None


def load_gpvpfn_bundle(result_dir: Path) -> dict:
    path = find_gpvpfn_json(result_dir)
    if path is None:
        raise FileNotFoundError(f"No gpVpfn_*.json in {result_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def predictions_npz_path(result_dir: Path) -> Path:
    return result_dir / "predictions.npz"


def save_predictions_npz(
    result_dir: Path,
    *,
    x_test: np.ndarray,
    y_true_test: np.ndarray,
    runs: list[dict],
    title: str,
    function_name: str,
    noise_train: float,
    noise_test: float,
) -> Path:
    """Persist per-run prediction arrays so ensemble plots can be regenerated later."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "x_test": np.asarray(x_test, dtype=np.float64).ravel(),
        "y_true_test": np.asarray(y_true_test, dtype=np.float64).ravel(),
        "title": title,
        "function_name": function_name,
        "noise_train": float(noise_train),
        "noise_test": float(noise_test),
        "num_runs": len(runs),
    }
    for i, run in enumerate(runs):
        prefix = f"run_{i + 1:03d}"
        payload[f"{prefix}_x_train"] = np.asarray(run["x_train"], dtype=np.float64).ravel()
        payload[f"{prefix}_y_train"] = np.asarray(run["y_train"], dtype=np.float64).ravel()
        if run.get("y_pred_gp") is not None:
            payload[f"{prefix}_y_pred_gp"] = np.asarray(run["y_pred_gp"], dtype=np.float64).ravel()
        if run.get("y_pred_tabpfn") is not None:
            payload[f"{prefix}_y_pred_tabpfn"] = np.asarray(run["y_pred_tabpfn"], dtype=np.float64).ravel()
        if run.get("y_std_gp") is not None:
            payload[f"{prefix}_y_std_gp"] = np.asarray(run["y_std_gp"], dtype=np.float64).ravel()
        if run.get("y_std_tabpfn") is not None:
            payload[f"{prefix}_y_std_tabpfn"] = np.asarray(run["y_std_tabpfn"], dtype=np.float64).ravel()
    out = predictions_npz_path(result_dir)
    np.savez_compressed(out, **payload)
    return out


def load_predictions_npz(result_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    path = predictions_npz_path(result_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path, allow_pickle=False)
    meta = {
        "title": str(data["title"]) if "title" in data else "",
        "function_name": str(data["function_name"]) if "function_name" in data else "",
        "noise_train": float(data["noise_train"]) if "noise_train" in data else 0.0,
        "noise_test": float(data["noise_test"]) if "noise_test" in data else 0.0,
    }
    x_test = np.asarray(data["x_test"], dtype=np.float64).ravel()
    y_true_test = np.asarray(data["y_true_test"], dtype=np.float64).ravel()
    n_runs = int(data["num_runs"]) if "num_runs" in data else 0
    runs: list[dict] = []
    for i in range(1, n_runs + 1):
        prefix = f"run_{i:03d}"
        run = {
            "x_train": np.asarray(data[f"{prefix}_x_train"], dtype=np.float64).ravel(),
            "y_train": np.asarray(data[f"{prefix}_y_train"], dtype=np.float64).ravel(),
        }
        for key in ("y_pred_gp", "y_pred_tabpfn", "y_std_gp", "y_std_tabpfn"):
            full = f"{prefix}_{key}"
            if full in data:
                run[key] = np.asarray(data[full], dtype=np.float64).ravel()
        runs.append(run)
    return x_test, y_true_test, runs, meta


def approximate_ncrps_from_scalar_metrics(metric: dict, y_test_std: float) -> bool:
    """
    Estimate Gaussian NCRPS from saved scalar metrics (RRMSE, NIS_width).

    Used only when per-point predictions were not saved. Marked with NCRPS_approx=True.
    """
    if metric.get("NCRPS") is not None:
        return False
    if y_test_std <= 0:
        return False
    nis_width = metric.get("NIS_width")
    rmse = metric.get("RMSE")
    if nis_width is None or rmse is None:
        rrmse = metric.get("RRMSE")
        if rrmse is None:
            return False
        rmse = float(rrmse) * y_test_std
    sigma = (float(nis_width) * y_test_std) / (2.0 * 1.96)
    if sigma <= 0:
        return False
    try:
        from scipy.stats import norm

        z = float(rmse) / sigma
        crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        metric["CRPS"] = float(crps)
        metric["NCRPS"] = float(crps / y_test_std)
        metric["NCRPS_approx"] = True
        return True
    except Exception:
        return False


def inject_ncrps_into_bundle(bundle: dict) -> int:
    """Add approximate NCRPS to metrics lists inside a gpVpfn JSON bundle. Returns count injected."""
    y_test_std = None
    for section in ("gp_data", "tabpfn_data"):
        info_key = "gp_model_info" if section == "gp_data" else "pfn_model_info"
        info = (bundle.get(section) or {}).get(info_key) or {}
        if y_test_std is None and info.get("y_test_std") is not None:
            y_test_std = float(info["y_test_std"])
    if y_test_std is None or y_test_std <= 0:
        return 0
    n = 0
    for section in ("gp_data", "tabpfn_data"):
        metrics_list = (bundle.get(section) or {}).get("metrics") or []
        for m in metrics_list:
            if isinstance(m, dict) and approximate_ncrps_from_scalar_metrics(m, y_test_std):
                n += 1
    return n
