"""
B3: Borehole SF. Loads data and bounds from load_experimental_data, sets up runs, calls run_BO.run_BO_gp.
"""
import json
import os
from pathlib import Path
import torch
import time

import defaults
from load_experimental_data import generate_mf_borehole_data, borehole_mixed_variables
from run_BO import run_BO_gp, run_BO_pfn, run_BO_Result, get_bounds_borehole
from gpplus.utils import set_seed
from gpplus.utils.metrics_functions import analyze_metrics


def borehole_SF_GPvsPFN_BO(
    num_runs: int = defaults.NUM_RUNS,
    num_test: int = 5000,
    start_size: int = 10,
    num_inits: int = defaults.TRAINER_NUM_INITS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float | None = defaults.TRAINER_LR,
    convergence_patience: int = defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs: int = defaults.TRAINER_MIN_EPOCHS,
    min_loss_change: float = defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    optimizer_kwargs: dict | None = defaults.TRAINER_OPTIMIZER_KWARGS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    save_path: str | None = "./results/borehole",
    title: str | None = None,
    standardize_X: bool = defaults.STANDARDIZE_X,
    standardize_y: bool = defaults.STANDARDIZE_Y,
    x_standardize_method: int = defaults.X_STANDARDIZE_METHOD,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = defaults.NOISE_TYPE,
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    pfn_dtype: torch.dtype = defaults.DTYPE_PFN,
    trainer_info: bool = defaults.TRAINER_INFO,
    run_models: str | None = defaults.RUN_MODELS,
    log_lbfgs_inner: bool = defaults.TRAINER_LOG_LBFGS_INNER,
    minimization_problem: bool = False,  # Borehole: maximize objective
    acquisition: str = defaults.BO_ACQUISITION,
    gp_optimize_af: bool = defaults.GP_OPTIMIZE_AF,
    n_AF_opt: int = defaults.BO_N_AF_OPT,
    n_AF_sample: int = defaults.BO_N_AF_SAMPLE,
    max_iter: int = defaults.BO_MAX_ITER,
    patience_no_improve: int = defaults.BO_PATIENCE_NO_IMPROVE,
    bo_full_info: bool = defaults.BO_FULL_INFO,
):
    """Borehole SF: load data from load_experimental_data, get bounds, run BO via run_BO_gp."""
    device = torch.device(gp_device)
    dtype = gp_dtype
    set_seed(seed)

    if run_models == "pfn":
        num_inits = 0

    if run_models is None:
        prefix = "gpVpfn"
    else:
        prefix = run_models

    if title is None:
        title_str = f"{prefix}_BO_borehole_SF_{start_size}Dn_{num_inits}inits_{max_iter}maxIter_{patience_no_improve}patience_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else:
        title_str = f"{prefix}_BO_borehole_SF_{title}_{start_size}Dn_{num_inits}inits_{max_iter}maxIter_{patience_no_improve}patience_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"

    dim = 8
    num_runs_gen = max(num_runs, 20)
    train_per_run = start_size * dim
    total_train = num_runs_gen * train_per_run
    train_noise_list = [noise_train] + [0.0] * 4
    test_noise_list = [noise_test] + [0.0] * 4

    X_train_all, _, X_test_all, y_test_all = generate_mf_borehole_data(
        train_samples_per_source=[total_train, 0, 0, 0, 0],
        test_samples_per_source=[num_test, 0, 0, 0, 0],
        seed=seed,
        train_noise=train_noise_list,
        test_noise=test_noise_list,
        noise_type=noise_type,
    )
    if X_train_all.shape[1] >= 9:
        X_train_all = X_train_all[:, :8]
    if X_test_all.shape[1] >= 9:
        X_test_all = X_test_all[:, :8]

    X_train_all = X_train_all.to(device=device, dtype=dtype)
    X_test_all = X_test_all.to(device=device, dtype=dtype)
    y_test_all = y_test_all.to(device=device, dtype=dtype)

    all_indices = torch.randperm(total_train, device=device)
    train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)
    X_inits = [X_train_all[train_indices_2d[i]] for i in range(num_runs)]

    bounds = get_bounds_borehole(device, dtype)

    def objective_fn_clean(x: torch.Tensor) -> torch.Tensor:
        return borehole_mixed_variables(x, source="s0")

    with torch.no_grad():
        y_test_clean_for_noise = objective_fn_clean(X_test_all)
        test_std_value = (
            float(y_test_clean_for_noise.std().item())
            if y_test_clean_for_noise.numel() > 1
            else 0.0
        )

    def objective_fn(x: torch.Tensor) -> torch.Tensor:
        y_clean = objective_fn_clean(x)
        if noise_train <= 0.0 or test_std_value <= 0.0:
            return y_clean
        scale = noise_train * test_std_value
        if noise_type == "gaussian":
            noise = torch.randn_like(y_clean) * scale
        elif noise_type == "uniform":
            noise = (torch.rand_like(y_clean) - 0.5) * 2.0 * scale * torch.sqrt(
                torch.tensor(3.0, dtype=y_clean.dtype, device=y_clean.device)
            )
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        return y_clean + noise

    # Data & model setup (print once, like B1/B4)
    print("\n" + "=" * 60)
    print("BO Borehole SF – Data & model setup")
    print("=" * 60)
    gp_model_info = {
        "title": title_str,
        "input_dim": dim,
        "start_size": start_size,
        "train_samples_per_run": train_per_run,
        "test_samples": num_test,
        "num_runs": num_runs,
        "standardize_X": standardize_X,
        "standardize_y": standardize_y,
        "x_standardize_method": x_standardize_method,
        "dtype": str(gp_dtype),
        "device": str(gp_device),
        "num_epochs": num_epochs,
        "num_inits": num_inits,
        "optimizer": optimizer_class.__name__ if optimizer_class else None,
        "convergence_patience": convergence_patience,
        "acquisition": acquisition,
        "gp_optimize_af": gp_optimize_af,
        "n_AF_opt": n_AF_opt,
        "n_AF_sample": n_AF_sample,
        "max_iter": max_iter,
        "patience_no_improve": patience_no_improve,
        "minimization_problem": minimization_problem,
        "seed": seed,
        "seed_trainer": seed_trainer,
    }
    for k, v in gp_model_info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    results: list[run_BO_Result] = []
    run_seeds: list[int] = []
    bo_metrics: list[dict] = []
    time_start = time.time()
    for run_idx in range(num_runs):
        print(f"\nrun {run_idx + 1} / {num_runs}")
        run_seed = seed_trainer if seed_trainer is not None else seed + run_idx
        run_seeds.append(run_seed)
        # Do not create per-run folders / JSON; aggregate across runs below instead.
        run_save_path = None
        if run_models in (None, "gp"):
            r = run_BO_gp(
                X_init=X_inits[run_idx],
                X_test=X_test_all,
                y_test=y_test_all,
                bounds=bounds,
                objective_fn=objective_fn,
                objective_fn_clean=objective_fn_clean,
                minimization_problem=minimization_problem,
                acquisition=acquisition,
                gp_optimize_af=gp_optimize_af,
                n_AF_opt=n_AF_opt,
                n_AF_sample=n_AF_sample,
                max_iter=max_iter,
                patience_no_improve=patience_no_improve,
                run_seed=run_seed,
                num_epochs=num_epochs,
                num_inits=num_inits,
                lr=lr,
                convergence_patience=convergence_patience,
                min_epochs=min_epochs,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                optimizer_kwargs=optimizer_kwargs if optimizer_kwargs is not None else defaults.TRAINER_OPTIMIZER_KWARGS,
                initializer_class=initializer_class,
                gp_device=gp_device,
                gp_dtype=gp_dtype,
                standardize_X=standardize_X,
                standardize_y=standardize_y,
                x_standardize_method=x_standardize_method,
                save_path=run_save_path,
                run_log_filename=None,
                kernel_module=defaults.SF_kernel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
                log_lbfgs_inner=log_lbfgs_inner,
            )
        elif run_models == "pfn":
            r = run_BO_pfn(
                X_init=X_inits[run_idx],
                X_test=X_test_all,
                y_test=y_test_all,
                bounds=bounds,
                objective_fn=objective_fn,
                objective_fn_clean=objective_fn_clean,
                minimization_problem=minimization_problem,
                acquisition=acquisition,
                n_AF_sample=n_AF_sample,
                max_iter=max_iter,
                patience_no_improve=patience_no_improve,
                run_seed=run_seed,
                gp_device=gp_device,
                gp_dtype=gp_dtype,
                save_path=run_save_path,
                run_log_filename=None,
                verbose=True,
                pfn_device=amp_device,
                pfn_dtype=pfn_dtype,
                bo_test_metrics=defaults.BO_TEST_METRICS,
                gi_pfn=defaults.BO_GI_PFN,
                standardize_X=standardize_X,
                standardize_y=standardize_y,
                x_standardize_method=x_standardize_method,
            )
        else:
            raise ValueError(f"Unsupported run_models='{run_models}'. Expected None, 'gp', or 'pfn'.")
        results.append(r)
        # Per-run BO summary metrics (final values + total and average times)
        total_train_time = float(sum(r.train_time_history)) if r.train_time_history else 0.0
        total_af_time = float(sum(r.af_time_history)) if r.af_time_history else 0.0
        total_time = total_train_time + total_af_time
        avg_train_time = (
            total_train_time / float(r.n_iterations) if r.n_iterations > 0 else 0.0
        )
        avg_af_time = (
            total_af_time / float(r.n_iterations) if r.n_iterations > 0 else 0.0
        )
        start_best_y = r.best_y_history[0] if r.best_y_history else None
        final_best_y = r.best_y_history[-1] if r.best_y_history else None
        final_best_y_clean = (
            r.best_y_clean_history[-1] if r.best_y_clean_history else None
        )
        # Starting and final y mean/std in raw space
        start_y = r.y_history[:start_size]
        final_y = r.y_history
        start_y_mean = float(start_y.mean().item()) if start_y.numel() > 0 else None
        start_y_std = float(start_y.std(unbiased=False).item()) if start_y.numel() > 0 else None
        final_y_mean = float(final_y.mean().item()) if final_y.numel() > 0 else None
        final_y_std = float(final_y.std(unbiased=False).item()) if final_y.numel() > 0 else None
        # Best x in raw space at the end of the run
        if minimization_problem:
            best_idx = int(torch.argmin(r.y_history))
        else:
            best_idx = int(torch.argmax(r.y_history))
        final_best_x = r.x_history[best_idx].tolist()

        metrics_entry: dict = {
            "run": run_idx + 1,
            "run_seed": run_seed,
            "Total_Time": total_time,
            "Train_Time": total_train_time,
            "AF_Time": total_af_time,
            "avg_train_time": avg_train_time,
            "avg_af_time": avg_af_time,
            "n_iterations": r.n_iterations,
            "start_y_mean": start_y_mean,
            "final_y_mean": final_y_mean,
            "start_y_std": start_y_std,
            "final_y_std": final_y_std,
            "start_best_y": start_best_y,
            "final_best_y": final_best_y,
            "final_best_y_clean": final_best_y_clean,
            "best_x": final_best_x,
        }
        # Only include test metrics if BO_TEST_METRICS is enabled.
        if defaults.BO_TEST_METRICS:
            final_test_metrics = r.test_metrics_history[-1] if r.test_metrics_history else {}
            final_nis = r.nis_history[-1] if r.nis_history else {}
            # Scalar test metrics from final GP evaluation (RRMSE, RMSE, MSE, MAE, etc.)
            for k in ("RRMSE", "RMSE", "MSE", "MAE"):
                if (
                    isinstance(final_test_metrics, dict)
                    and k in final_test_metrics
                    and final_test_metrics[k] is not None
                ):
                    metrics_entry[k] = float(final_test_metrics[k])
            # NIS metrics from final iteration, if available
            for k in ("NIS", "NIS_width", "NIS_outside"):
                if (
                    isinstance(final_nis, dict)
                    and k in final_nis
                    and final_nis[k] is not None
                ):
                    metrics_entry[k] = float(final_nis[k])

        bo_metrics.append(metrics_entry)

        print(f"{title_str}")
        print(f"\n--- Run {run_idx + 1}/{num_runs} (run_seed={run_seed}) ---")
        print(f"  n_iterations: {r.n_iterations}")
        if r.best_y_history:
            print(f"  best_y (final): {r.best_y_history[-1]:.6f}")
            print(f"  best_y_history (first 3): {[round(y, 6) for y in r.best_y_history[:3]]} ...")
        if r.x_chosen_history:
            print(f"  last x_chosen: {r.x_chosen_history[-1].tolist()}")
        if r.train_time_history:
            print(f"  avg_train_time: {avg_train_time:.4f}")
        if r.af_time_history:
            print(f"  avg_af_time: {avg_af_time:.4f}")
        print()

    time_end = time.time()
    print(f"Total BO time for Borehole problem: {time_end - time_start:.2f}s")
    # Aggregate BO metrics across runs into a single JSON (summary + per-run metrics + optional full histories).
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Summary over runs: mean/std/median/min/max for each numeric metric.
        summary_input = [
            {
                k: v
                for k, v in m.items()
                if k not in ("run", "run_seed") and isinstance(v, (int, float))
            }
            for m in bo_metrics
        ]
        bo_summary = analyze_metrics(summary_input, print_summary=False, label="BO", title=title_str)

        # Build JSON in the desired key order.
        out: dict = {}
        out["BO_summary"] = bo_summary
        out["BO_metrics"] = bo_metrics

        # Optional full per-iteration information for each run (controlled by BO_FULL_INFO flag),
        # restricted to BO-related quantities only.
        if bo_full_info:
            full_info_runs = []
            for idx, (r, run_seed) in enumerate(zip(results, run_seeds)):
                full_info_runs.append(
                    {
                        "run": idx + 1,
                        "run_seed": run_seed,
                        "n_iterations": r.n_iterations,
                        "x_chosen": [t.tolist() for t in r.x_chosen_history],
                        "af_values": r.af_value_history,
                        "y_pred_mean_history": r.y_pred_mean_history,
                        "train_time_s": r.train_time_history,
                        "af_time_s": r.af_time_history,
                        "best_y_history": r.best_y_history,
                        "best_y_clean_history": r.best_y_clean_history,
                    }
                )
            out["BO_full_info"] = full_info_runs

        # High-level BO / GP model configuration info, placed after full metrics.
        out["BO_model_info"] = gp_model_info

        # Append defaults.py source at end of JSON for reproducibility (like regression files).
        defaults_path = Path(__file__).resolve().parent / "defaults.py"
        try:
            out["defaults_py"] = defaults_path.read_text(encoding="utf-8")
        except Exception:
            pass

        out_file = os.path.join(save_path, f"{title_str}.json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)

    return {"runs": results}


if __name__ == "__main__":
    print("B3 Borehole BO smoke test (max_iter=2)...")
    out = borehole_SF_GPvsPFN_BO(num_runs=1, num_test=100, start_size=2, max_iter=2, patience_no_improve=2, run_models="gp")
    print("Done. Runs:", len(out["runs"]))
    if out["runs"]:
        r = out["runs"][0]
        print("  n_iterations:", r.n_iterations)
        print("  best_y_history:", r.best_y_history[:3], "...")
