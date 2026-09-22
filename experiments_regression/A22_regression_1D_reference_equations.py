"""
Equations for the five 1D regression examples in the paper.

  python A22_regression_1D_reference_equations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Default domain matching defaults x_bounds in A22_regression_1D.py
X_LO, X_HI = -0.5, 0.5


def _as_x(X):
    """Accept (n, 1) array/tensor or 1D array; return a 1D numpy array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.ravel()


# --------------------------------------------------------------------------- #
# Paper equations
# --------------------------------------------------------------------------- #
def eq_chirp(X):
    """Non-stationary frequency (linear chirp).

    f(x) = sin(2*pi * (2 + 14*(x - X_LO)) * x)
    """
    x = _as_x(X)
    freq = 2.0 + 14.0 * (x - X_LO)
    return np.sin(2 * np.pi * freq * x)


def eq_discontinuity(X):
    """Smooth trend with a hard jump (Heaviside) at x = 0. GP over-smooths the
    step; PFN can represent the discontinuity.

    f(x) = 0.6*sin(2*pi*x) + 1.0 * 1[x >= 0]
    """
    x = _as_x(X)
    return 0.6 * np.sin(2 * np.pi * x) + 1.0 * (x >= 0.0)


def eq_localized_bump(X):
    """Flat background with one narrow Gaussian spike. Classic length-scale
    conflict: GP either over-smooths the spike or gets wiggly everywhere.

    f(x) = exp(-(x / 0.04)^2)
    """
    x = _as_x(X)
    return np.exp(-((x / 0.04) ** 2))


def eq_triangle_wave(X):
    """Piecewise-linear triangle wave (periodic kinks). Generalizes the |x|
    toy: non-smooth everywhere, smooth GP prior is mismatched.

    period = 0.4
    """
    x = _as_x(X)
    period = 0.4
    # triangle in [-1, 1]
    frac = (x / period) % 1.0
    return 2.0 * np.abs(2.0 * frac - 1.0) - 1.0


def eq_damped_sine(X):
    """Amplitude-modulated sine.

    f(x) = exp(-6*|x|) * sin(10*pi*x)
    """
    x = _as_x(X)
    return np.exp(-6.0 * np.abs(x)) * np.sin(10 * np.pi * x)


CANDIDATES = {
    "discontinuity": (eq_discontinuity, "Discontinuous sine"),
    "triangle_wave": (eq_triangle_wave, "Triangle wave"),
    "chirp": (eq_chirp, "Non-stationary chirp"),
    "localized_bump": (eq_localized_bump, "Localized bump"),
    "damped_sine": (eq_damped_sine, "Damped sine"),
}


def plot_candidates(save_path: str | Path | None = None, n: int = 1000, n_train: int = 10):
    """Render every candidate equation in a grid and (optionally) save it."""
    import matplotlib.pyplot as plt

    x = np.linspace(X_LO, X_HI, n)
    rng = np.random.default_rng(0)
    x_train = np.sort(rng.uniform(X_LO, X_HI, size=n_train))

    ncols = 2
    nrows = int(np.ceil(len(CANDIDATES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows))
    axes = np.asarray(axes).ravel()

    for ax, (key, (fn, desc)) in zip(axes, CANDIDATES.items()):
        y = fn(x)
        ax.plot(x, y, color="C0", lw=2, label="f(x)")
        ax.scatter(x_train, fn(x_train), color="C3", s=28, zorder=5,
                   label=f"{n_train} train pts")
        ax.set_title(f"{key}\n{desc}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes[len(CANDIDATES):]:
        ax.set_visible(False)

    fig.suptitle("Candidate 1D regression equations for GP vs PFN", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved candidate equation plot to: {save_path.resolve()}")
    return fig


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results" / "A22_reference_equations" / "candidates.png"
    plot_candidates(save_path=out)
