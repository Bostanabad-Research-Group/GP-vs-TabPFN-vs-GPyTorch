"""
Regenerate the combined 4-panel LML figures from saved *_surfaces.npz files.

Does NOT recompute any surfaces — it only re-reads the saved coarse/zoom npz
files and rewrites each ``*_combined.png`` with a single shared colorbar.

Usage:
  python A22_replot_combined_change_of_variables.py
  python A22_replot_combined_change_of_variables.py <results_root> [<results_root> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from experimental_utils.lml_surface_omega_outputscale import replot_combined_from_npz

DEFAULT_ROOTS = (
    Path("./results/A22_regression_1D_change_of_variables"),
    Path("./results/A22_regression_1D_change_of_variables_201"),
    Path("./results/A22_regression_1D_change_of_variables_401"),
)

DESCRIPTIONS = {
    "forrester": "Forrester benchmark",
    "damped_forrester": "5*damped sine + Forrester",
    "sin_cubic": "1/4*sin(6*pi*x) + 6*x^3 - 7*x^2 + x + 0.5",
    "smooth_multisine": "sin(2*pi*x) + 0.5*sin(6*pi*x)",
    "chirp": "Linear chirp (non-stationary frequency)",
    "discontinuity": "0.6*sin(2*pi*x) + Heaviside jump at x=0",
    "localized_bump": "Flat + narrow Gaussian spike",
    "triangle_wave": "Triangle wave (periodic kinks)",
    "gramacy_lee": "Gramacy & Lee sin(10*pi*z)/(2z) + (z-1)^4",
    "damped_sine": "exp(-6*|x|)*sin(10*pi*x)",
}


def _description_for_dir(run_dir: Path) -> str | None:
    for name, desc in DESCRIPTIONS.items():
        if name in run_dir.parts:
            return desc
    return None


def replot_root(root: Path) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for npz in sorted(root.rglob("*_lml_surfaces.npz")):
        if npz.name.endswith("_zoom_surfaces.npz"):
            continue
        run_dir = npz.parent
        panel_title = _description_for_dir(run_dir)
        out = replot_combined_from_npz(run_dir, panel_title=panel_title)
        if out is not None:
            print(f"  Replotted {out}")
            count += 1
        else:
            print(f"  Skipped (missing zoom npz): {run_dir}")
    return count


def main(argv: list[str]) -> None:
    roots = [Path(a) for a in argv[1:]] if len(argv) > 1 else list(DEFAULT_ROOTS)
    total = 0
    for root in roots:
        print(f"\nScanning {root.resolve()}")
        total += replot_root(root)
    print(f"\nDone — regenerated {total} combined figure(s).")


if __name__ == "__main__":
    main(sys.argv)
