"""
Regenerate A22 plots from saved results without re-running the full experiment when possible.

What is saved today under results/A22_regression_1D/<function>/:
  - gpVpfn_*.json          scalar metrics + experiment config (always)
  - plots/prediction_runs/ per-run PNGs (always)
  - predictions.npz        per-run prediction arrays (only if the experiment saved them)

NCRPS:
  - Older runs often lack NCRPS in JSON because per-point std was not written into metrics.
  - This script injects an *approximate* Gaussian NCRPS from RRMSE + NIS_width (marked NCRPS_approx)
    and replots RRMSE / NIS / NCRPS violins.

Ensemble (all-runs overlay):
  - Requires predictions.npz. Per-run PNGs alone are not enough to recover curves.
  - Use --replay to re-run inference and create predictions.npz + ensemble plot.

Examples:
  python A22_replot_from_results.py
  python A22_replot_from_results.py --function chirp
  python A22_replot_from_results.py --replay --function chirp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental_utils.a22_results_io import (
    find_gpvpfn_json,
    inject_ncrps_into_bundle,
    load_predictions_npz,
    predictions_npz_path,
)
from experimental_utils.plot_tabpfn1d_comparison import save_1d_all_runs_gp_tabpfn_plot
from gpplus.utils.metrics_functions import plot_metrics


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "A22_regression_1D"


def _replot_metrics(result_dir: Path, write_json: bool = True) -> bool:
    json_path = find_gpvpfn_json(result_dir)
    if json_path is None:
        print(f"[skip] no JSON in {result_dir}")
        return False
    bundle = json.loads(json_path.read_text(encoding="utf-8"))
    n = inject_ncrps_into_bundle(bundle)
    title = (bundle.get("gp_data") or {}).get("gp_model_info", {}).get("title")
    if not title:
        title = (bundle.get("tabpfn_data") or {}).get("pfn_model_info", {}).get("title")
    if not title:
        title = result_dir.name

    tabpfn_metrics = (bundle.get("tabpfn_data") or {}).get("metrics")
    gp_metrics = (bundle.get("gp_data") or {}).get("metrics")
    if not tabpfn_metrics or not gp_metrics:
        print(f"[skip] incomplete metrics in {result_dir}")
        return False

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(tabpfn_metrics, gp_metrics, labels=["TabPFN", "GP"], title=title, save_path=str(plot_dir))
    print(f"[metrics] {result_dir.name}: replotted (approx NCRPS injected for {n} run-metrics)")

    if write_json:
        json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return True


def _replot_ensemble(result_dir: Path) -> bool:
    if not predictions_npz_path(result_dir).is_file():
        print(f"[ensemble skip] {result_dir.name}: no predictions.npz (per-run PNGs are not enough)")
        return False
    x_test, y_true_test, runs, meta = load_predictions_npz(result_dir)
    out_dir = result_dir / "plots" / "prediction_runs"
    title = meta.get("title") or result_dir.name
    suffix = f"ntr{meta.get('noise_train', 0.0)}_nte{meta.get('noise_test', 0.0)}"
    fp = save_1d_all_runs_gp_tabpfn_plot(
        runs,
        x_test,
        out_dir,
        title=title,
        y_true_test=y_true_test,
        file_suffix=suffix,
    )
    print(f"[ensemble] {result_dir.name}: {fp}")
    return True


def _replay_inference(result_dir: Path) -> None:
    """Re-run GP+TabPFN for one function folder using saved JSON config."""
    from A22_regression_1D import regression_1D_GPvsPFN
    from experimental_utils.a22_results_io import load_gpvpfn_bundle

    bundle = load_gpvpfn_bundle(result_dir)
    gp_info = (bundle.get("gp_data") or {}).get("gp_model_info") or {}
    pfn_info = (bundle.get("tabpfn_data") or {}).get("pfn_model_info") or {}
    info = {**pfn_info, **gp_info}

    fn_name = info.get("function_name") or result_dir.name
    print(f"[replay] {fn_name}: re-running inference (this trains GP again)...")
    regression_1D_GPvsPFN(
        function_name=fn_name,
        num_runs=int(info.get("num_runs", 10)),
        num_test=int(info.get("test_samples", 5000)),
        train_size=int(info.get("train_samples", 20)),
        dimensions=int(info.get("dimensions", 1)),
        x_bounds=info.get("x_bounds"),
        noise_train=float(info.get("noise_train", 0.0)),
        noise_test=float(info.get("noise_test", 0.0)),
        noise_type=info.get("noise_type", "gaussian"),
        seed=int(info.get("seed", 42)),
        seed_trainer=info.get("seed_trainer"),
        standardize_X=bool(info.get("standardize_X", True)),
        standardize_y=bool(info.get("standardize_y", True)),
        x_standardize_method=int(info.get("x_standardize_method", 2)),
        preprocess_pfn=bool(info.get("preprocess_pfn", False)),
        num_epochs=int(info.get("num_epochs", 1)),
        num_inits=int(info.get("num_inits", 16)),
        save_path=str(result_dir),
        save_predictions_npz=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate A22 metrics / ensemble plots from saved results")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--function", type=str, default=None, help="Single function folder name, e.g. chirp")
    parser.add_argument("--metrics-only", action="store_true", help="Only replot metric violins (fast)")
    parser.add_argument("--ensemble-only", action="store_true", help="Only replot all-runs overlay (needs npz)")
    parser.add_argument("--replay", action="store_true", help="Re-run inference to create predictions.npz")
    parser.add_argument("--no-write-json", action="store_true", help="Do not write NCRPS back into JSON")
    args = parser.parse_args()

    root = args.results_root
    if not root.is_dir():
        raise SystemExit(f"Results root not found: {root}")

    if args.function:
        dirs = [root / args.function]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())

    do_metrics = not args.ensemble_only
    do_ensemble = not args.metrics_only

    for result_dir in dirs:
        if not result_dir.is_dir():
            continue
        print("\n" + "=" * 60)
        print(result_dir.name)
        print("=" * 60)

        if args.replay:
            _replay_inference(result_dir)

        if do_metrics:
            _replot_metrics(result_dir, write_json=not args.no_write_json)
        if do_ensemble:
            _replot_ensemble(result_dir)


if __name__ == "__main__":
    main()
