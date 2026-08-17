"""
Rerun trainer analysis plots for all trainer_analysis folders under a results tree.

Finds every *_GP_Trainer_Analysis.json under the given root (default: results_v2.0/NLL3),
and for each one regenerates:
  - Hyperparameter vs epochs plot (and embedding plots if present)
  - Epoch metrics plot (loss, NLL, NIS, LOO_NLL, KF, Residual_MSE vs epoch)
into the "plots" subfolder next to each JSON.

Usage:
  python rerun_trainer_analysis_plots.py
  python rerun_trainer_analysis_plots.py results_v2.0/NLL3
  python rerun_trainer_analysis_plots.py results_v2.0/NLL3/buckling
"""

from __future__ import annotations

from pathlib import Path

from plot_trainer_analysis_hyperparams import find_trainer_analysis_jsons, process_file
from plot_epoch_metrics import plot_epoch_metrics, plot_iter_metrics, plot_iter2_metrics


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "results_v3.0" / "Test1"

    import argparse
    parser = argparse.ArgumentParser(
        description="Rerun trainer hyperparameter analysis plots for JSONs under a results tree.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(default_root),
        help=f"Root directory to search for trainer_analysis/*_GP_Trainer_Analysis.json (default: {default_root})",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    jsons = find_trainer_analysis_jsons(root)
    if not jsons:
        print(f"No *_GP_Trainer_Analysis.json files found under {root}")
        return

    print(f"Found {len(jsons)} trainer analysis JSON(s) under {root}\n")
    for json_path in sorted(jsons):
        out_dir = json_path.parent / "plots"
        print(f"Processing: {json_path}")
        process_file(json_path, out_dir=out_dir)
        # try:
        #     plot_epoch_metrics(json_path, out_dir=out_dir)
        # except ValueError:
        #     pass  # no epoch_metrics in this JSON
        # except Exception as e:
            # print(f"  Epoch metrics skipped: {e}")
        try:
            plot_iter_metrics(json_path, out_dir=out_dir)
        except ValueError:
            pass  # no lbfgs_inner_metrics in this JSON
        except Exception as e:
            print(f"  Iter metrics skipped: {e}")
        try:
            plot_iter2_metrics(json_path, out_dir=out_dir)
        except ValueError:
            pass
        except Exception as e:
            print(f"  Iter2 metrics skipped: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
