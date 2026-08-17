"""
Candidate 1D regression equations for the GP-vs-PFN comparison.

Goal: pick a "good" 1D function to benchmark a Gaussian Process (stationary
RBF/Matern-style kernel) against TabPFN. The interesting test functions are the
ones where GP and PFN are expected to *disagree*, e.g.:

  - non-stationary frequency content (a stationary GP kernel has one length
    scale, so it cannot be sharp in one region and smooth in another),
  - discontinuities / kinks (a smooth GP prior over-smooths jumps),
  - localized features on an otherwise flat background (length-scale conflict),
  - heteroscedastic-looking structure.

All functions follow the repo convention used in load_experimental_data.py:
they take X of shape (n, 1) (or a 1D array) and return shape (n,). They are all
defined on the default domain x in [-0.5, 0.5] so they can be dropped straight
into generate_tabpfn_1d_*_data / regression_1D_GPvsPFN, except where a function
has a canonical literature domain (noted in DOMAIN below), in which case the
input is internally remapped from [-0.5, 0.5] to that domain.

Run this file directly to render every candidate in a grid and save the figure:

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


def _remap(x, lo, hi):
    """Map x from [X_LO, X_HI] onto [lo, hi] linearly."""
    t = (x - X_LO) / (X_HI - X_LO)
    return lo + t * (hi - lo)


# --------------------------------------------------------------------------- #
# Candidate equations
# --------------------------------------------------------------------------- #
def eq_smooth_multisine(X):
    """Smooth, stationary two-tone sine. Baseline: GP (RBF) should excel.

    f(x) = sin(2*pi*x) + 0.5 * sin(6*pi*x)
    """
    x = _as_x(X)
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)


def eq_chirp(X):
    """Non-stationary frequency (linear chirp). Stationary GP must compromise
    its single length scale; PFN can adapt locally.

    f(x) = sin(2*pi * (2 + 14*(x - X_LO)) * x)
    """
    x = _as_x(X)
    freq = 2.0 + 14.0 * (x - X_LO)  # frequency grows left -> right
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


def eq_forrester(X):
    """Forrester et al. (2008) benchmark on its canonical domain x in [0, 1].

    f(z) = (6z - 2)^2 * sin(12z - 4),  z in [0, 1]
    """
    x = _as_x(X)
    z = _remap(x, 0.0, 1.0)
    return (6 * z - 2) ** 2 * np.sin(12 * z - 4)


def eq_gramacy_lee(X):
    """Gramacy & Lee (2012) on its canonical domain z in [0.5, 2.5]. Increasing
    frequency toward the left + polynomial tail; a notoriously GP-unfriendly 1D
    benchmark.

    f(z) = sin(10*pi*z) / (2z) + (z - 1)^4
    """
    x = _as_x(X)
    z = _remap(x, 0.5, 2.5)
    return np.sin(10 * np.pi * z) / (2 * z) + (z - 1) ** 4


def eq_damped_sine(X):
    """Amplitude-modulated (damped) sine. Heteroscedastic-looking variance in
    the signal; tests whether the model adapts amplitude across x.

    f(x) = exp(-6*|x|) * sin(10*pi*x)
    """
    x = _as_x(X)
    return np.exp(-6.0 * np.abs(x)) * np.sin(10 * np.pi * x)


def eq_damped_forrester(X):
    """Damped sine plus Forrester benchmark. Combines amplitude modulation
    with multimodal oscillations and a steep rise near the right edge.

    f(x) = 5*exp(-6*|x|)*sin(10*pi*x) + (6z-2)^2*sin(12z-4),  z in [0,1]
    """
    x = _as_x(X)
    z = _remap(x, 0.0, 1.0)
    damped = np.exp(-6.0 * np.abs(x)) * np.sin(10 * np.pi * x)
    forrester = (6 * z - 2) ** 2 * np.sin(12 * z - 4)
    return 5.0 * damped + forrester


CANDIDATES = {
    "smooth_multisine": (eq_smooth_multisine, "Smooth two-tone sine (GP-friendly baseline)"),
    "chirp": (eq_chirp, "Linear chirp (non-stationary frequency)"),
    "discontinuity": (eq_discontinuity, "Smooth trend + Heaviside jump"),
    "localized_bump": (eq_localized_bump, "Flat + narrow Gaussian spike"),
    "triangle_wave": (eq_triangle_wave, "Triangle wave (periodic kinks)"),
    "forrester": (eq_forrester, "Forrester (z in [0,1])"),
    "gramacy_lee": (eq_gramacy_lee, "Gramacy & Lee (z in [0.5,2.5])"),
    "damped_sine": (eq_damped_sine, "Damped sine (amplitude modulated)"),
    "damped_forrester": (eq_damped_forrester, "Damped sine + Forrester combined"),
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
