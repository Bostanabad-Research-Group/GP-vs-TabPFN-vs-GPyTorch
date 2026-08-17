"""
Merge per-run BO JSON files (run_0/*.json, run_1/*.json, ...) into a single
aggregated JSON per experiment, matching the B1-B5 format (BO_summary,
BO_metrics, BO_full_info, BO_model_info, defaults_py).

Usage:
  # Merge only one folder (safest):
  python merge_per_run_jsons_to_single.py --folder ./results/GP+/B6_rosenbrock
  python merge_per_run_jsons_to_single.py --folder ./results/GP+/B6_rosenbrock --dry-run

  # Merge everything under default root (experiments_BO/results):
  python merge_per_run_jsons_to_single.py
  python merge_per_run_jsons_to_single.py --results-dir ./results --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Set this to the folder you want to merge (the one that contains run_0, run_1, ...).
# Then run:  python merge_per_run_jsons_to_single.py
# DEFAULT_FOLDER: str | None = "./results/PFN_V2.0_no_GI/B6_rosenbrock"
# DEFAULT_FOLDER: str | None = "./results/PFN_V2.0_no_GI/B7_zakharov"
# DEFAULT_FOLDER: str | None = "./results/PFN_V2.0_no_GI/B8_griewank"
# DEFAULT_FOLDER: str | None = "./results/PFN_V2.0_no_GI/B9_dixon_price"


def _numeric_summary(values: list[float]) -> dict:
    """Compute mean, std, median, min, max, count for a list of numbers."""
    import statistics
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "count": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": n,
    }


def _analyze_metrics(metrics_list: list[dict], exclude_keys: tuple[str, ...] = ("run", "run_seed")) -> dict:
    """Build BO_summary from list of BO_metrics entries (numeric keys only)."""
    summary = {}
    if not metrics_list:
        return summary
    keys = set()
    for m in metrics_list:
        for k, v in m.items():
            if k not in exclude_keys and v is not None and isinstance(v, (int, float)):
                keys.add(k)
    for k in sorted(keys):
        values = [m[k] for m in metrics_list if k in m and m[k] is not None and isinstance(m[k], (int, float))]
        if values:
            summary[k] = _numeric_summary(values)
    return summary


def _flatten_x_chosen(x: list) -> list:
    """Return a single list of coords from x_chosen entry (may be [[x]] or [x])."""
    if not x:
        return x
    if isinstance(x[0], (int, float)):
        return x
    return _flatten_x_chosen(x[0])


def run_file_to_metrics(run_idx: int, data: dict) -> dict:
    """Build one BO_metrics entry from a per-run JSON (run_BO log_data)."""
    run_seed = data.get("run_seed")
    n_iter = data.get("n_iterations", 0)
    train_s = data.get("train_time_s") or []
    af_s = data.get("af_time_s") or []
    total_train = sum(train_s) if train_s else 0.0
    total_af = sum(af_s) if af_s else 0.0
    total_time = total_train + total_af
    best_y = data.get("best_y_history") or []
    start_best_y = best_y[0] if best_y else None
    final_best_y = best_y[-1] if best_y else None
    x_chosen = data.get("x_chosen") or []
    if best_y and x_chosen:
        best_idx = min(range(len(best_y)), key=lambda i: best_y[i])
        raw = x_chosen[best_idx]
        best_x = _flatten_x_chosen(raw) if raw else []
    else:
        best_x = []
    avg_train = total_train / n_iter if n_iter else 0.0
    avg_af = total_af / n_iter if n_iter else 0.0
    return {
        "run": run_idx + 1,
        "run_seed": run_seed,
        "Total_Time": total_time,
        "Train_Time": total_train,
        "AF_Time": total_af,
        "avg_train_time": avg_train,
        "avg_af_time": avg_af,
        "n_iterations": n_iter,
        "start_best_y": start_best_y,
        "final_best_y": final_best_y,
        "best_x": best_x,
    }


def run_file_to_full_info(run_idx: int, data: dict) -> dict:
    """Build one BO_full_info entry from a per-run JSON."""
    return {
        "run": run_idx + 1,
        "run_seed": data.get("run_seed"),
        "n_iterations": data.get("n_iterations"),
        "x_chosen": data.get("x_chosen"),
        "af_values": data.get("af_values"),
        "train_time_s": data.get("train_time_s"),
        "af_time_s": data.get("af_time_s"),
        "best_y_history": data.get("best_y_history"),
    }


def find_run_dirs(parent: Path) -> list[tuple[int, Path]]:
    """Return [(run_index, path), ...] for run_0, run_1, ... under parent."""
    pattern = re.compile(r"^run_(\d+)$")
    out = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def collect_basenames(run_dirs: list[tuple[int, Path]]) -> set[str]:
    """Return set of JSON basenames that appear in any of the run dirs."""
    names = set()
    for _, run_path in run_dirs:
        for f in run_path.iterdir():
            if f.suffix.lower() == ".json":
                names.add(f.name)
    return names


def merge_per_run_jsons_in_dir(parent: Path, defaults_py_path: Path | None, dry_run: bool) -> int:
    """
    If parent contains run_0, run_1, ... merge same-named JSONs into single files in parent.
    Returns number of merged files written.
    """
    run_dirs = find_run_dirs(parent)
    if not run_dirs:
        return 0
    basenames = collect_basenames(run_dirs)
    if not basenames:
        return 0
    count = 0
    for basename in sorted(basenames):
        runs_by_idx = {}
        for run_idx, run_path in run_dirs:
            f = run_path / basename
            if not f.exists():
                continue
            runs_by_idx[run_idx] = f
        if len(runs_by_idx) == 0:
            continue
        indices = sorted(runs_by_idx.keys())
        run_files = [runs_by_idx[i] for i in indices]
        all_data = []
        for f in run_files:
            with open(f, encoding="utf-8") as fp:
                all_data.append(json.load(fp))
        bo_metrics = [run_file_to_metrics(i, d) for i, d in enumerate(all_data)]
        bo_summary = _analyze_metrics(bo_metrics)
        full_info = [run_file_to_full_info(i, d) for i, d in enumerate(all_data)]
        title_str = basename.replace(".json", "")
        out = {
            "BO_summary": bo_summary,
            "BO_metrics": bo_metrics,
            "BO_full_info": full_info,
            "BO_model_info": {
                "title": title_str,
                "merged_from_per_run": True,
                "num_runs": len(all_data),
            },
        }
        if defaults_py_path and defaults_py_path.exists():
            try:
                out["defaults_py"] = defaults_py_path.read_text(encoding="utf-8")
            except Exception:
                pass
        out_path = parent / basename
        if dry_run:
            print(f"  [dry-run] would write {out_path} ({len(all_data)} runs)")
        else:
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(out, fp, indent=2)
            print(f"  wrote {out_path} ({len(all_data)} runs)")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-run BO JSONs into single files (B1-B5 style).",
        epilog=(
            "Examples:\n"
            "  # Merge only B6_rosenbrock under results/GP+:\n"
            "  python merge_per_run_jsons_to_single.py --folder ./results/GP+/B6_rosenbrock\n"
            "  # Merge everything under results (default root):\n"
            "  python merge_per_run_jsons_to_single.py\n"
            "  # Dry-run a specific folder:\n"
            "  python merge_per_run_jsons_to_single.py --folder ./results/PFN_V2.0_GI/B7_zakharov --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Single folder to merge (e.g. ./results/GP+/B6_rosenbrock). If set, only this folder is processed; no tree walk.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root results directory when not using --folder (default: experiments_BO/results). Ignored if --folder is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done.",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    defaults_py = script_dir / "defaults.py"

    if args.folder is None and DEFAULT_FOLDER:
        args.folder = (script_dir / DEFAULT_FOLDER).resolve()
    if args.folder is not None:
        # Single folder mode: only process this directory
        target = args.folder.resolve()
        if not target.exists():
            print(f"Folder does not exist: {target}", file=sys.stderr)
            sys.exit(1)
        if not target.is_dir():
            print(f"Not a directory: {target}", file=sys.stderr)
            sys.exit(1)
        run_dirs = find_run_dirs(target)
        if not run_dirs:
            print(f"No run_0, run_1, ... subdirs found in {target}", file=sys.stderr)
            sys.exit(1)
        print(f"{target}: merging {len(run_dirs)} run dirs ...")
        total = merge_per_run_jsons_in_dir(target, defaults_py, args.dry_run)
        print(f"Done. Merged {total} JSON file(s).")
        return

    # Walk from root
    if args.results_dir is None:
        args.results_dir = script_dir / "results"
    args.results_dir = args.results_dir.resolve()
    if not args.results_dir.exists():
        print(f"Results dir does not exist: {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    total = 0
    for root, dirs, _ in os.walk(args.results_dir, topdown=True):
        root_path = Path(root)
        run_dirs = find_run_dirs(root_path)
        if not run_dirs:
            continue
        rel = root_path.relative_to(args.results_dir) if root_path != args.results_dir else root_path.name
        print(f"{rel}: merging {len(run_dirs)} run dirs ...")
        n = merge_per_run_jsons_in_dir(root_path, defaults_py, args.dry_run)
        total += n
        if n:
            dirs.clear()
    print(f"Done. Merged {total} JSON file(s).")


if __name__ == "__main__":
    main()
