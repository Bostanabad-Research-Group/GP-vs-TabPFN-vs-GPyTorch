"""Where archived runs, new runs, and comparisons are stored.

    results/
      summary.md
      regression_results/
        regression_original_results/
        regression_new_results/
        regression_results_comparison/
      bo_results/
        bo_original_results/
        bo_new_results/
        bo_results_comparison/
      classification_results/
        classification_original_results/
        classification_new_results/
        classification_results_comparison/
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def suite_dir(name: str) -> Path:
    return RESULTS / f"{name}_results"


def original_results(name: str) -> Path:
    return suite_dir(name) / f"{name}_original_results"


def new_results(name: str) -> Path:
    return suite_dir(name) / f"{name}_new_results"


def comparison_dir(name: str) -> Path:
    return suite_dir(name) / f"{name}_results_comparison"
