"""Chain LBFGS then Adam gpytorch log-scale suites (buckling + zakharov).

Each child process patches `defaults_gpytorch` independently.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = (
    "B4_run_GPvsPFN_gpytorch_LBFGS_logscale.py",
    "B4_run_GPvsPFN_gpytorch_adam_logscale.py",
)


def run_script(script_name: str) -> None:
    script_path = HERE / script_name
    print(f"\n{'=' * 80}\nRunning {script_name}\n{'=' * 80}\n", flush=True)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(HERE))
    if result.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {result.returncode}")


def main() -> None:
    for script in SCRIPTS:
        run_script(script)
    print("\n[B4 logscale run_all] LBFGS + Adam suites finished.")


if __name__ == "__main__":
    main()
