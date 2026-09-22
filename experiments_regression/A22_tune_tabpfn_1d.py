"""
Tune TabPFN-2.5 hyperparameters for A22 1D regression problems.

Reuses the same data generation / train-test protocol as A22_regression_1D.py
(seed, Sobol split, train_size=20, num_runs), evaluates TabPFN only, and
selects a config that improves point metrics while tightening intervals
(lower NIS_width) without destroying coverage.

Usage (from experiments/):
  python A22_tune_tabpfn_1d.py                 # screen + full eval
  python A22_tune_tabpfn_1d.py --phase screen  # grid search only
  python A22_tune_tabpfn_1d.py --phase full --config-json path/to/best.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import prepend_cache_path

import defaults
from A22_regression_1D import REGRESSION_1D_FUNCTIONS, _X_BOUNDS_DEPENDENT_FUNCTIONS
from experimental_utils.a22_results_io import save_predictions_npz as persist_predictions_npz
from experimental_utils.plot_tabpfn1d_comparison import (
    save_1d_all_runs_gp_tabpfn_plot,
    save_1d_train_gp_tabpfn_plot,
)
from gpplus.utils import set_seed, train_eval_PFN
from gpplus.utils.metrics_functions import analyze_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from run_metadata import experiment_data_info, pfn_model_info

warnings.filterwarnings("ignore")

RESULTS_ROOT = Path("./results_1D/A22_regression_1D")
TUNED_ROOT = Path("./results_1D/A22_regression_1D_tabpfn_tuned")
SCREEN_FUNCS = ("chirp", "discontinuity", "localized_bump", "damped_sine")
CKPT = {
    "default": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "small-samples": "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
    "real": "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    "quantiles": "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
}


def _ckpt_path(name: str) -> str:
    return str(prepend_cache_path(CKPT[name]))


def _config_grid() -> list[dict]:
    """Configs aimed at better means + tighter (more confident) intervals."""
    grid: list[dict] = []
    # Baseline reference
    grid.append(
        dict(
            name="baseline_default_n8_T0.9",
            n_estimators=8,
            softmax_temperature=0.9,
            average_before_softmax=False,
            model_path="auto",
            checkpoint="auto",
        )
    )
    for ckpt in ("default", "small-samples"):
        for n_est in (8, 16, 32):
            for temp in (0.9, 0.6, 0.45, 0.3):
                if n_est == 8 and temp == 0.9 and ckpt == "default":
                    continue  # same as baseline
                grid.append(
                    dict(
                        name=f"{ckpt}_n{n_est}_T{temp}",
                        n_estimators=n_est,
                        softmax_temperature=temp,
                        average_before_softmax=False,
                        model_path=_ckpt_path(ckpt),
                        checkpoint=ckpt,
                    )
                )
    # A few extras
    for temp in (0.45, 0.3):
        grid.append(
            dict(
                name=f"default_n16_T{temp}_avgSoftmax",
                n_estimators=16,
                softmax_temperature=temp,
                average_before_softmax=True,
                model_path=_ckpt_path("default"),
                checkpoint="default",
            )
        )
    grid.append(
        dict(
            name="quantiles_n16_T0.45",
            n_estimators=16,
            softmax_temperature=0.45,
            average_before_softmax=False,
            model_path=_ckpt_path("quantiles"),
            checkpoint="quantiles",
        )
    )
    grid.append(
        dict(
            name="real_n16_T0.45",
            n_estimators=16,
            softmax_temperature=0.45,
            average_before_softmax=False,
            model_path=_ckpt_path("real"),
            checkpoint="real",
        )
    )
    return grid


def _score(med: dict) -> float:
    """Lower is better. Prefer accuracy + CRPS, reward tighter intervals, penalize bad coverage."""
    rrmse = float(med.get("RRMSE", 1.0))
    ncrps = float(med.get("NCRPS", med.get("NCRPS_exact", 1.0)))
    nis_w = float(med.get("NIS_width", 2.0))
    nis_out = float(med.get("NIS_outside", 0.0))
    # Target ~nominal 5% outside for 95% PI; penalize above that hard.
    coverage_pen = 2.5 * max(0.0, nis_out - 0.08) + 0.5 * max(0.0, nis_out - 0.05)
    return rrmse + ncrps + 0.35 * nis_w + coverage_pen


def _median_metrics(metrics: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics if m.get(k) is not None]
        out[k] = float(statistics.median(vals)) if vals else float("nan")
    return out


def _make_regressor(cfg: dict, device: str, seed: int) -> TabPFNRegressor:
    kwargs = dict(
        n_estimators=int(cfg["n_estimators"]),
        softmax_temperature=float(cfg["softmax_temperature"]),
        average_before_softmax=bool(cfg.get("average_before_softmax", False)),
        device=device,
        random_state=seed,
    )
    if cfg.get("model_path") not in (None, "auto"):
        kwargs["model_path"] = cfg["model_path"]
    return TabPFNRegressor(**kwargs)


def eval_tabpfn_on_function(
    function_name: str,
    cfg: dict,
    *,
    num_runs: int = 10,
    train_size: int = 20,
    num_test: int = 5000,
    seed: int = defaults.SEED,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    pfn_dtype=defaults.DTYPE_PFN,
    preprocess_pfn: bool = defaults.PREPROCESS_PFN,
    save_path: Path | None = None,
    plot: bool = False,
    baseline_gp_bundle: dict | None = None,
) -> dict:
    """Run TabPFN-only A22 protocol for one function + one config."""
    from A22_regression_1D import _ensure_ncrps

    fn_cfg = REGRESSION_1D_FUNCTIONS[function_name]
    generate_data_fn = fn_cfg["generate_data"]
    true_fn = fn_cfg["true_fn"]
    x_bounds = list(fn_cfg["default_x_bounds"])
    x_bounds_tuple = (float(x_bounds[0]), float(x_bounds[1]))
    test_x_bounds = list(x_bounds)

    if function_name in _X_BOUNDS_DEPENDENT_FUNCTIONS:

        def eval_true_fn(X: torch.Tensor) -> torch.Tensor:
            return true_fn(X, x_bounds=x_bounds_tuple)

    else:
        eval_true_fn = true_fn

    title = (
        f"A22_{function_name}_1Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]"
        f"_tabpfnTuned_noiseTest0.0_noiseTrain0.0_x{num_runs}"
    )

    set_seed(seed)
    num_runs_gen = max(num_runs, 20)
    train_per_run = train_size * 1
    total_train = num_runs_gen * train_per_run
    total_samples = num_test + total_train

    X_train_all, y_train_all, X_test_all, y_test_all = generate_data_fn(
        n_train=total_train,
        n_test=num_test,
        dimensions=1,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        train_noise=0.0,
        test_noise=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=seed,
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)
    qual_dict = learn_encodings(X)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)

    x_test_1d = X_test_all[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_true_1d = eval_true_fn(X_test_all.to(dtype=torch.float64)).detach().cpu().numpy().ravel()

    TabPFN_metrics: list[dict] = []
    all_runs_plot_data: list[dict] = []
    tabpfn_model_info = None

    for i in range(num_runs):
        run_seed = seed + i
        run_train_indices = train_indices_2d[i]
        X_train = X_train_all[run_train_indices].detach().clone().to(dtype=defaults.DTYPE_GP)
        y_train = y_train_all[run_train_indices].detach().clone().to(dtype=defaults.DTYPE_GP)
        X_test = X_test_all.detach().clone().to(dtype=defaults.DTYPE_GP)
        y_test = y_test_all.detach().clone().to(dtype=defaults.DTYPE_GP)

        x_train_plot = X_train[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
        y_train_plot = y_train.detach().cpu().to(dtype=torch.float64).numpy().ravel()

        # Match A22: PREPROCESS_PFN=False → raw X/y for TabPFN
        pfn_X_train, pfn_X_test = X_train, X_test
        pfn_y_train, pfn_y_test = y_train, y_test
        pfn_y_mean = pfn_y_std = None
        if preprocess_pfn:
            raise NotImplementedError("Tuning script expects PREPROCESS_PFN=False")

        regressor = _make_regressor(cfg, amp_device, seed)
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            pfn_X_train,
            pfn_X_test,
            pfn_y_train,
            pfn_y_test,
            amp_device=amp_device,
            amp_dtype=pfn_dtype,
            regressor=regressor,
            source_cols=source_cols,
            y_train_mean=pfn_y_mean,
            y_train_std=pfn_y_std,
            record_y_train_mean=None,
            record_y_train_std=None,
        )
        _ensure_ncrps(tabpfn_metric, y_test, y_pred_tabpfn, output_std_tabpfn)
        TabPFN_metrics.append(tabpfn_metric)

        # Attach GP preds from baseline bundle for comparison plots when available
        y_pred_gp = y_std_gp = None
        if baseline_gp_bundle is not None and plot:
            # GP curves not stored in JSON; leave None unless predictions.npz exists
            pass

        if plot and save_path is not None:
            out_plot_dir = Path(save_path) / "plots" / "prediction_runs"
            try:
                save_1d_train_gp_tabpfn_plot(
                    x_train_plot,
                    y_train_plot,
                    x_test_1d,
                    y_pred_gp,
                    y_pred_tabpfn,
                    y_std_gp,
                    output_std_tabpfn,
                    out_plot_dir,
                    title=title,
                    run_index=i + 1,
                    y_true_test=y_true_1d,
                    file_suffix="tuned",
                )
            except Exception as e:
                print(f"plot failed run {i+1}: {e}")

        all_runs_plot_data.append(
            {
                "x_train": x_train_plot,
                "y_train": y_train_plot,
                "y_pred_gp": y_pred_gp,
                "y_pred_tabpfn": y_pred_tabpfn,
                "y_std_gp": y_std_gp,
                "y_std_tabpfn": output_std_tabpfn,
            }
        )

        if i == 0:
            shared = experiment_data_info(
                cat_cols=cat_cols,
                cont_cols=cont_cols,
                source_cols=source_cols,
                qual_dict=qual_dict,
                input_dim=X_train.shape[1],
                train_samples=X_train.shape[0],
                test_samples=num_test,
                standardize_X=defaults.STANDARDIZE_X,
                standardize_y=defaults.STANDARDIZE_Y,
                x_standardize_method=defaults.X_STANDARDIZE_METHOD,
                X_scaling_type="None",
                y_train_mean=None,
                y_train_std=None,
                y_test_mean=float(y_test_all.mean().item()),
                y_test_std=float(y_test_all.std().item()),
                num_runs=num_runs,
                seed=seed,
                seed_trainer=None,
                noise_train=0.0,
                noise_test=0.0,
                noise_type=defaults.NOISE_TYPE,
                preprocess_pfn=preprocess_pfn,
                pfn_dtype=pfn_dtype,
                dimensions=1,
                x_bounds=x_bounds,
                function_name=function_name,
                function_description=fn_cfg["description"],
            )
            tabpfn_model_info = {
                **pfn_model_info(regressor, experiment_data=shared),
                "tuning_config": {k: v for k, v in cfg.items() if k != "model_path"},
                "model_path": cfg.get("model_path"),
                "n_estimators": cfg.get("n_estimators"),
                "softmax_temperature": cfg.get("softmax_temperature"),
                "average_before_softmax": cfg.get("average_before_softmax"),
            }

    summary = analyze_metrics(TabPFN_metrics, print_summary=False, label="TabPFN", title=title)
    keys = ["RRMSE", "NCRPS", "NIS_width", "NIS", "NIS_outside", "NLPD", "MAE"]
    med = _median_metrics(TabPFN_metrics, keys)

    if plot and save_path is not None and all_runs_plot_data:
        out_plot_dir = Path(save_path) / "plots" / "prediction_runs"
        try:
            save_1d_all_runs_gp_tabpfn_plot(
                all_runs_plot_data,
                x_test_1d,
                out_plot_dir,
                title=title,
                y_true_test=y_true_1d,
                file_suffix="tuned",
            )
        except Exception as e:
            print(f"all-runs plot failed: {e}")
        try:
            persist_predictions_npz(
                Path(save_path),
                x_test=x_test_1d,
                y_true_test=y_true_1d,
                runs=all_runs_plot_data,
                title=title,
                function_name=function_name,
                noise_train=0.0,
                noise_test=0.0,
            )
        except Exception as e:
            print(f"npz save failed: {e}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "tabpfn_data": {
                "summary": summary,
                "metrics": TabPFN_metrics,
                "pfn_model_info": tabpfn_model_info,
            },
            "tuning_config": cfg,
            "median_metrics": med,
            "score": _score(med),
        }
        if baseline_gp_bundle and "gp_data" in baseline_gp_bundle:
            payload["gp_data"] = baseline_gp_bundle["gp_data"]
        (save_path / f"pfn_tuned_{title}.json").write_text(json.dumps(payload, indent=2, default=str))

    return {"function": function_name, "config": cfg, "median": med, "score": _score(med), "metrics": TabPFN_metrics}


def load_baseline_median(function_name: str) -> dict | None:
    matches = list(RESULTS_ROOT.glob(f"{function_name}/gpVpfn_*.json"))
    if not matches:
        return None
    d = json.loads(matches[0].read_text())
    metrics = d["tabpfn_data"]["metrics"]
    keys = ["RRMSE", "NCRPS", "NIS_width", "NIS", "NIS_outside", "NLPD", "MAE"]
    return {"path": str(matches[0]), "bundle": d, "median": _median_metrics(metrics, keys), "score": _score(_median_metrics(metrics, keys))}


def run_screen(num_runs: int = 5, functions: tuple[str, ...] = SCREEN_FUNCS) -> dict:
    grid = _config_grid()
    print(f"Screening {len(grid)} configs on {functions} with {num_runs} runs each")
    # Deduplicate overly large grid: keep a leaner set for wall-clock
    lean = []
    keep_names_prefix = {
        "baseline_default_n8_T0.9",
        "default_n8_T0.45",
        "default_n8_T0.3",
        "default_n16_T0.9",
        "default_n16_T0.45",
        "default_n16_T0.3",
        "default_n32_T0.45",
        "small-samples_n16_T0.45",
        "small-samples_n16_T0.3",
        "small-samples_n32_T0.45",
        "default_n16_T0.3_avgSoftmax",
        "quantiles_n16_T0.45",
        "real_n16_T0.45",
    }
    for c in grid:
        if c["name"] in keep_names_prefix:
            lean.append(c)
    print(f"Using lean grid of {len(lean)} configs")

    rows = []
    t0 = time.time()
    for cfg in lean:
        print("\n" + "=" * 70)
        print(f"CONFIG {cfg['name']}")
        print("=" * 70)
        func_scores = []
        func_meds = []
        for fn in functions:
            print(f"\n--- {fn} ---")
            res = eval_tabpfn_on_function(fn, cfg, num_runs=num_runs, plot=False, save_path=None)
            base = load_baseline_median(fn)
            delta = None
            if base:
                delta = {
                    k: res["median"][k] - base["median"][k]
                    for k in res["median"]
                    if k in base["median"]
                }
            print(
                f"  med RRMSE={res['median']['RRMSE']:.4f} NCRPS={res['median']['NCRPS']:.4f} "
                f"NIS_w={res['median']['NIS_width']:.4f} NIS_out={res['median']['NIS_outside']:.4f} "
                f"score={res['score']:.4f}"
                + (
                    f" | dRRMSE={delta['RRMSE']:+.4f} dNIS_w={delta['NIS_width']:+.4f}"
                    if delta
                    else ""
                )
            )
            func_scores.append(res["score"])
            func_meds.append(res["median"])
        # Aggregate across screen functions
        agg = {
            k: float(statistics.mean([m[k] for m in func_meds]))
            for k in func_meds[0]
        }
        rows.append(
            {
                "config": cfg,
                "mean_score": float(statistics.mean(func_scores)),
                "agg_median_like": agg,
                "per_function_scores": dict(zip(functions, func_scores)),
            }
        )

    rows.sort(key=lambda r: r["mean_score"])
    TUNED_ROOT.mkdir(parents=True, exist_ok=True)
    out = {"screen_functions": list(functions), "num_runs": num_runs, "ranking": rows}
    out_path = TUNED_ROOT / "screen_ranking.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "#" * 70)
    print(f"Screen done in {time.time()-t0:.1f}s. Top 5:")
    for i, r in enumerate(rows[:5]):
        a = r["agg_median_like"]
        print(
            f"  {i+1}. {r['config']['name']}: score={r['mean_score']:.4f} "
            f"RRMSE={a['RRMSE']:.4f} NCRPS={a['NCRPS']:.4f} "
            f"NIS_w={a['NIS_width']:.4f} NIS_out={a['NIS_outside']:.4f}"
        )
    print(f"Wrote {out_path}")
    return out


def run_full(cfg: dict, num_runs: int = 10) -> None:
    print(f"Full eval with config: {cfg['name']}")
    summary_rows = []
    for fn in REGRESSION_1D_FUNCTIONS:
        print("\n" + "#" * 70)
        print(f"FULL {fn}")
        print("#" * 70)
        base = load_baseline_median(fn)
        save_path = TUNED_ROOT / fn
        res = eval_tabpfn_on_function(
            fn,
            cfg,
            num_runs=num_runs,
            plot=True,
            save_path=save_path,
            baseline_gp_bundle=base["bundle"] if base else None,
        )
        delta = None
        if base:
            delta = {k: res["median"][k] - base["median"][k] for k in res["median"]}
        row = {
            "function": fn,
            "tuned": res["median"],
            "baseline": base["median"] if base else None,
            "delta": delta,
            "score_tuned": res["score"],
            "score_baseline": base["score"] if base else None,
        }
        summary_rows.append(row)
        print(
            f"TUNED  RRMSE={res['median']['RRMSE']:.4f} NCRPS={res['median']['NCRPS']:.4f} "
            f"NIS_w={res['median']['NIS_width']:.4f} NIS_out={res['median']['NIS_outside']:.4f}"
        )
        if base:
            print(
                f"BASE   RRMSE={base['median']['RRMSE']:.4f} NCRPS={base['median']['NCRPS']:.4f} "
                f"NIS_w={base['median']['NIS_width']:.4f} NIS_out={base['median']['NIS_outside']:.4f}"
            )
            print(
                f"DELTA  RRMSE={delta['RRMSE']:+.4f} NCRPS={delta['NCRPS']:+.4f} "
                f"NIS_w={delta['NIS_width']:+.4f} NIS_out={delta['NIS_outside']:+.4f}".replace(
                    "DELTA", "DELTA"
                )
            )

    out = {"config": cfg, "results": summary_rows}
    (TUNED_ROOT / "full_comparison.json").write_text(json.dumps(out, indent=2, default=str))
    # Compact table
    print("\n" + "=" * 90)
    print(f"{'fn':18} {'dRRMSE':>8} {'dNCRPS':>8} {'dNIS_w':>8} {'dNIS_out':>9}  better?")
    for r in summary_rows:
        if not r["delta"]:
            continue
        d = r["delta"]
        better = (d["RRMSE"] <= 0.01 and d["NCRPS"] <= 0.01 and d["NIS_width"] < -0.05) or (
            r["score_tuned"] < r["score_baseline"]
        )
        print(
            f"{r['function']:18} {d['RRMSE']:+8.4f} {d['NCRPS']:+8.4f} "
            f"{d['NIS_width']:+8.4f} {d['NIS_outside']:+9.4f}  {better}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "full", "both"], default="both")
    parser.add_argument("--num-runs-screen", type=int, default=5)
    parser.add_argument("--num-runs-full", type=int, default=10)
    parser.add_argument("--config-json", type=str, default=None, help="Path to ranking json or config dict")
    parser.add_argument("--config-rank", type=int, default=0, help="Which ranked config to use for full")
    args = parser.parse_args()

    TUNED_ROOT.mkdir(parents=True, exist_ok=True)

    ranking = None
    if args.phase in ("screen", "both"):
        ranking = run_screen(num_runs=args.num_runs_screen)

    if args.phase in ("full", "both"):
        if args.config_json:
            payload = json.loads(Path(args.config_json).read_text())
            if "ranking" in payload:
                cfg = payload["ranking"][args.config_rank]["config"]
            else:
                cfg = payload
        elif ranking is not None:
            cfg = ranking["ranking"][args.config_rank]["config"]
        else:
            screen_path = TUNED_ROOT / "screen_ranking.json"
            payload = json.loads(screen_path.read_text())
            cfg = payload["ranking"][args.config_rank]["config"]
        run_full(cfg, num_runs=args.num_runs_full)


if __name__ == "__main__":
    main()
