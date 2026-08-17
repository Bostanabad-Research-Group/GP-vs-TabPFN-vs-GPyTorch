"""
1D LML surface in transformed GP hyperparameters: omega vs log10(outputscale).

Ported from gp-private/experiments_opt_study/lml_surface_change_of_variables.py
for A22 change-of-variables studies (transformed panels only).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

OMEGA_BOUNDS_DEFAULT = (-6.0, 3.0)
S_BOUNDS_DEFAULT = (-5.0, 4.0)
ARD_PANEL_INCHES = 4.25
CHOLESKY_FALLBACK_JITTERS = (1e-12, 1e-10, 1e-8)
MIN_LENGTHSCALE_DEFAULT = 1e-6
MAX_LENGTHSCALE_DEFAULT = 1000.0
LENGTHSCALE_ZOOM_HALF_WIDTH_DEFAULT = 0.15


def _lml_from_kernel(
    y: torch.Tensor,
    kernel: torch.Tensor,
    nugget: float,
    *,
    cholesky_jitter: float = 0.0,
    profile_constant_mean: bool = False,
    normalize_by_n: bool = True,
    fallback_jitters: tuple[float, ...] = CHOLESKY_FALLBACK_JITTERS,
) -> float:
    """Log marginal likelihood via Cholesky (profiled mean optional)."""
    n = int(y.shape[0])
    k_base = kernel.to(dtype=torch.float64, device=y.device)
    eye = torch.eye(n, dtype=torch.float64, device=y.device)

    jitter_attempts = [float(cholesky_jitter)]
    for j in fallback_jitters:
        if j not in jitter_attempts:
            jitter_attempts.append(float(j))

    lfac = None
    for jitter in jitter_attempts:
        k = k_base + (float(nugget) + jitter) * eye
        try:
            lfac = torch.linalg.cholesky(k)
            break
        except RuntimeError:
            continue
    if lfac is None:
        return float("nan")

    if profile_constant_mean:
        ones = torch.ones(n, 1, dtype=torch.float64, device=y.device)
        a = torch.cholesky_solve(ones, lfac)
        num = float(ones.T @ torch.cholesky_solve(y.unsqueeze(1), lfac))
        den = float(ones.T @ a)
        if abs(den) < 1e-18:
            return float("nan")
        beta = num / den
        resid = y - beta
    else:
        resid = y - y.mean()

    alpha = torch.cholesky_solve(resid.unsqueeze(1), lfac)
    data_fit = -0.5 * float(resid @ alpha)
    log_det = -float(torch.sum(torch.log(torch.diag(lfac))))
    const = -0.5 * n * math.log(2.0 * math.pi)
    lml = data_fit + log_det + const
    if normalize_by_n and n > 0:
        lml /= n
    return float(lml)


def _sort_interval(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _clip_interval(bounds: tuple[float, float], outer: tuple[float, float]) -> tuple[float, float]:
    lo, hi = _sort_interval(bounds)
    ox_lo, ox_hi = _sort_interval(outer)
    lo = max(ox_lo, lo)
    hi = min(ox_hi, hi)
    if lo >= hi:
        return ox_lo, ox_hi
    return lo, hi


def _square_interval_pair(
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    *,
    outer_x: tuple[float, float] | None = None,
    outer_y: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_lo, x_hi = _sort_interval(bounds_x)
    y_lo, y_hi = _sort_interval(bounds_y)
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo, 1e-18)
    x_lo, x_hi = cx - half, cx + half
    y_lo, y_hi = cy - half, cy + half
    if outer_x is not None:
        x_lo, x_hi = _clip_interval((x_lo, x_hi), outer_x)
    if outer_y is not None:
        y_lo, y_hi = _clip_interval((y_lo, y_hi), outer_y)
    span_x = x_hi - x_lo
    span_y = y_hi - y_lo
    target = min(span_x, span_y)
    if target > 1e-18 and (span_x > target + 1e-12 or span_y > target + 1e-12):
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        x_lo, x_hi = cx - 0.5 * target, cx + 0.5 * target
        y_lo, y_hi = cy - 0.5 * target, cy + 0.5 * target
        if outer_x is not None:
            x_lo, x_hi = _clip_interval((x_lo, x_hi), outer_x)
        if outer_y is not None:
            y_lo, y_hi = _clip_interval((y_lo, y_hi), outer_y)
    return (x_lo, x_hi), (y_lo, y_hi)


def _l_interval_from_omega_interval(omega_bounds: tuple[float, float]) -> tuple[float, float]:
    """l = 10^(-omega/2): ascending l interval for ascending omega interval."""
    o_lo, o_hi = _sort_interval(omega_bounds)
    l_lo = 10.0 ** (-o_hi / 2.0)
    l_hi = 10.0 ** (-o_lo / 2.0)
    return l_lo, l_hi


def _trainer_l_bounds(omega_bounds: tuple[float, float]) -> tuple[float, float]:
    l_lo, l_hi = _l_interval_from_omega_interval(omega_bounds)
    return max(l_lo, 1e-12), l_hi


def _ascending_l_for_omega_s_surface(
    l_vals: np.ndarray,
    z_o: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip descending l on axis 0 for isotropic (l, sigma^2) LML grids."""
    l_vals = np.asarray(l_vals, dtype=float)
    z = np.asarray(z_o, dtype=float)
    if l_vals.size and l_vals[0] > l_vals[-1]:
        l_vals = l_vals[::-1]
        z = z[::-1, :]
    return l_vals, z


def _finite_argmax_2d(z: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    z_arr = np.asarray(z, dtype=float)
    m = np.isfinite(z_arr)
    if not np.any(m):
        return None, float("nan")
    z_safe = np.where(m, z_arr, -np.inf)
    idx = np.unravel_index(int(np.argmax(z_safe)), z_safe.shape)
    return (int(idx[0]), int(idx[1])), float(z_arr[idx])


def _levels_for_figure(
    surfaces: list[np.ndarray],
    *,
    n: int = 24,
    span_below_peak: float | None = None,
    min_span: float = 0.25,
) -> np.ndarray:
    mins, maxs = [], []
    for z in surfaces:
        f = np.asarray(z, dtype=float)
        f = f[np.isfinite(f)]
        if f.size:
            mins.append(float(f.min()))
            maxs.append(float(f.max()))
    if not mins:
        raise RuntimeError("no finite LML values")
    vmin = max(mins)
    vmax = max(maxs)
    if span_below_peak is not None:
        vmin = max(vmin, vmax - float(span_below_peak))
    if vmin >= vmax:
        vmin = vmax - float(min_span)
    return np.linspace(vmin, vmax, n)


def _window_bounds(
    center: float,
    half_width: float,
    outer: tuple[float, float],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, float]:
    lo_b, hi_b = _sort_interval(outer)
    if min_value is not None:
        lo_b = max(lo_b, float(min_value))
    if max_value is not None:
        hi_b = min(hi_b, float(max_value))
    width = 2.0 * float(half_width)
    span = hi_b - lo_b
    if width >= span or span <= 0:
        return lo_b, hi_b
    c = float(center)
    lo, hi = c - float(half_width), c + float(half_width)
    if lo < lo_b:
        lo, hi = lo_b, lo_b + width
    elif hi > hi_b:
        lo, hi = hi_b - width, hi_b
    return lo, hi


def _zoom_pair_limits(
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
    outer_x: tuple[float, float],
    outer_y: tuple[float, float],
    *,
    square: bool,
    min_x: float | None = None,
    max_x: float | None = None,
    min_y: float | None = None,
    max_y: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    bx_kw: dict = {}
    if min_x is not None:
        bx_kw["min_value"] = float(min_x)
    if max_x is not None:
        bx_kw["max_value"] = float(max_x)
    by_kw: dict = {}
    if min_y is not None:
        by_kw["min_value"] = float(min_y)
    if max_y is not None:
        by_kw["max_value"] = float(max_y)
    bx = _window_bounds(center_x, half_x, outer_x, **bx_kw)
    by = _window_bounds(center_y, half_y, outer_y, **by_kw)
    if square:
        half = 0.5 * max(bx[1] - bx[0], by[1] - by[0])
        bx = _window_bounds(center_x, half, outer_x, **bx_kw)
        by = _window_bounds(center_y, half, outer_y, **by_kw)
    return bx, by


def _resolve_omega_zoom_half_width(
    *,
    omega_zoom_half_width: float | None,
    omega_zoom_fraction: float,
    outer_omega: tuple[float, float],
) -> float:
    if omega_zoom_half_width is not None:
        return float(omega_zoom_half_width)
    o_lo, o_hi = _sort_interval(outer_omega)
    return float(omega_zoom_fraction) * max(o_hi - o_lo, 1e-18)


def _resolve_s_zoom_half_width(
    *,
    s_zoom_half_width: float | None,
    s_zoom_fraction: float,
    outer_s: tuple[float, float],
) -> float:
    if s_zoom_half_width is not None:
        return float(s_zoom_half_width)
    lo, hi = _sort_interval(outer_s)
    return float(s_zoom_fraction) * max(hi - lo, 1e-18)


def _build_omega_s_grids(
    omega_bounds: tuple[float, float],
    s_bounds: tuple[float, float],
    *,
    n_grid: int,
    outer_omega: tuple[float, float],
    outer_s: tuple[float, float],
    square: bool,
) -> dict:
    ob_o = _clip_interval(omega_bounds, outer_omega)
    ob_s = _clip_interval(s_bounds, outer_s)
    if square:
        ob_o, ob_s = _square_interval_pair(ob_o, ob_s, outer_x=outer_omega, outer_y=outer_s)
    omega_vals = np.linspace(ob_o[0], ob_o[1], int(n_grid))
    s_vals = np.linspace(ob_s[0], ob_s[1], int(n_grid))
    sigma2_vals = np.power(10.0, s_vals)
    l_vals = np.power(10.0, -omega_vals / 2.0)
    return {
        "omega_vals": omega_vals,
        "s_vals": s_vals,
        "sigma2_vals": sigma2_vals,
        "l_vals": l_vals,
        "omega_xlim": (float(omega_vals[0]), float(omega_vals[-1])),
        "s_ylim": (float(s_vals[0]), float(s_vals[-1])),
        "sigma2_xlim": (float(sigma2_vals[0]), float(sigma2_vals[-1])),
        "l_ylim": (float(l_vals[0]), float(l_vals[-1])),
    }


def _compute_transformed_omega_s_surface(
    *,
    y_used: torch.Tensor,
    pairwise_sqdist: torch.Tensor,
    nugget: float,
    omega_vals: np.ndarray,
    s_vals: np.ndarray,
    profile_constant_mean: bool,
    normalize_lml_by_n: bool,
    cholesky_jitter: float,
) -> np.ndarray:
    z_t = np.full((len(omega_vals), len(s_vals)), np.nan, dtype=np.float64)
    for i_w, omega in enumerate(omega_vals):
        alpha = 10.0 ** float(omega)
        exp_term = torch.exp(-alpha * pairwise_sqdist)
        for i_s, s in enumerate(s_vals):
            kernel = (10.0 ** float(s)) * exp_term
            z_t[i_w, i_s] = _lml_from_kernel(
                y_used,
                kernel,
                nugget,
                cholesky_jitter=cholesky_jitter,
                profile_constant_mean=profile_constant_mean,
                normalize_by_n=normalize_lml_by_n,
            )
    return z_t


def _build_l_sigma2_grids(
    l_bounds: tuple[float, float],
    sigma2_bounds: tuple[float, float],
    *,
    n_grid: int,
    outer_l: tuple[float, float],
    outer_sigma2: tuple[float, float],
) -> dict:
    l_lo, l_hi = _clip_interval(l_bounds, outer_l)
    sig_lo, sig_hi = _clip_interval(sigma2_bounds, outer_sigma2)
    l_vals = np.linspace(l_lo, l_hi, int(n_grid))
    sigma2_vals = np.linspace(sig_lo, sig_hi, int(n_grid))
    return {
        "l_vals": l_vals,
        "sigma2_vals": sigma2_vals,
        "l_ylim": (float(l_vals[0]), float(l_vals[-1])),
        "sigma2_xlim": (float(sigma2_vals[0]), float(sigma2_vals[-1])),
    }


def _compute_original_l_sigma2_surface(
    *,
    y_used: torch.Tensor,
    pairwise_sqdist: torch.Tensor,
    nugget: float,
    l_vals: np.ndarray,
    sigma2_vals: np.ndarray,
    profile_constant_mean: bool,
    normalize_lml_by_n: bool,
    cholesky_jitter: float,
) -> np.ndarray:
    z_o = np.full((len(l_vals), len(sigma2_vals)), np.nan, dtype=np.float64)
    for i_l, l_val in enumerate(l_vals):
        inv_l2 = 1.0 / max(float(l_val) ** 2, 1e-18)
        exp_term = torch.exp(-pairwise_sqdist * inv_l2)
        for i_sig, sigma2 in enumerate(sigma2_vals):
            kernel = float(sigma2) * exp_term
            z_o[i_l, i_sig] = _lml_from_kernel(
                y_used,
                kernel,
                nugget,
                cholesky_jitter=cholesky_jitter,
                profile_constant_mean=profile_constant_mean,
                normalize_by_n=normalize_lml_by_n,
            )
    return z_o


def _coarse_best_on_omega_s_panel(panel: dict) -> tuple[float, float]:
    z_t = np.asarray(panel["transformed_surface"], dtype=float)
    omega_vals = panel["omega_vals"]
    s_vals = panel["s_vals"]
    idx, _ = _finite_argmax_2d(z_t)
    if idx is None:
        mid = len(omega_vals) // 2
        return float(omega_vals[mid]), float(s_vals[mid])
    return float(omega_vals[idx[0]]), float(s_vals[idx[1]])


def _coarse_best_l_sigma2(z_o: np.ndarray, l_vals: np.ndarray, sigma2_vals: np.ndarray) -> tuple[float, float]:
    idx, _ = _finite_argmax_2d(z_o)
    if idx is None:
        return float(l_vals[len(l_vals) // 2]), float(sigma2_vals[len(sigma2_vals) // 2])
    return float(l_vals[idx[0]]), float(sigma2_vals[idx[1]])


def _draw_transformed_on_ax(
    ax,
    panel: dict,
    *,
    levels: np.ndarray,
    cmap,
    subtitle: str,
    equal_aspect: bool = True,
) -> object:
    s_vals = panel["s_vals"]
    omega_vals = panel["omega_vals"]
    z_t = np.asarray(panel["transformed_surface"], dtype=float)
    xv_t, yv_t = np.meshgrid(s_vals, omega_vals, indexing="xy")
    xlim = panel.get("plot_s_xlim", panel["s_ylim"])
    ylim = panel.get("plot_omega_ylim", panel["omega_xlim"])

    zm = np.ma.masked_invalid(z_t)
    mappable = ax.contourf(xv_t, yv_t, zm, levels=levels, cmap=cmap, extend="both")
    ax.contour(xv_t, yv_t, zm, levels=8, colors="white", linewidths=0.9, alpha=0.85)

    idx, _ = _finite_argmax_2d(z_t)
    if idx is not None:
        ax.scatter(
            [float(xv_t[idx])],
            [float(yv_t[idx])],
            s=40,
            c="red",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("outputscale (log10 σ²)")
    ax.set_ylabel("omega")
    ax.set_title(subtitle)
    ax.grid(True, alpha=0.25)
    return mappable


def _draw_original_on_ax(
    ax,
    *,
    l_vals: np.ndarray,
    sigma2_vals: np.ndarray,
    z_o: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    levels: np.ndarray,
    cmap,
    subtitle: str,
) -> object:
    l_plot, z_plot = _ascending_l_for_omega_s_surface(l_vals, z_o)
    xv, yv = np.meshgrid(np.asarray(sigma2_vals, dtype=float), l_plot, indexing="xy")
    zm = np.ma.masked_invalid(z_plot)
    mappable = ax.contourf(xv, yv, zm, levels=levels, cmap=cmap, extend="both")
    ax.contour(xv, yv, zm, levels=8, colors="white", linewidths=0.9, alpha=0.85)

    idx, _ = _finite_argmax_2d(z_plot)
    if idx is not None:
        ax.scatter(
            [float(xv[idx])],
            [float(yv[idx])],
            s=40,
            c="red",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("sigma^2 (outputscale)")
    ax.set_ylabel("lengthscale l")
    ax.set_title(subtitle)
    ax.grid(True, alpha=0.25)
    return mappable


def _plot_combined_4panel_figure(
    *,
    out_path: Path,
    figure_title: str,
    transformed_coarse_panel: dict,
    transformed_zoom_panel: dict,
    original_coarse: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]],
    original_zoom: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]],
    levels: np.ndarray,
    span_below_peak: float | None,
) -> None:
    """2×2: transformed coarse | zoom (top), original coarse | zoom (bottom).

    Uses a single shared color scale (``levels``) for all four panels and one
    shared colorbar placed in its own reserved axis on the right (never overlaps).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.3")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(ARD_PANEL_INCHES * 2.25, ARD_PANEL_INCHES * 2.1),
        dpi=150,
        squeeze=False,
    )

    mappable = _draw_transformed_on_ax(
        axes[0, 0],
        transformed_coarse_panel,
        levels=levels,
        cmap=cmap,
        subtitle="transformed — coarse",
    )
    _draw_transformed_on_ax(
        axes[0, 1],
        transformed_zoom_panel,
        levels=levels,
        cmap=cmap,
        subtitle="transformed — zoom (hw=0.15)",
    )

    l_c, sig_c, z_o_c, xlim_c, ylim_c = original_coarse
    l_z, sig_z, z_o_z, xlim_z, ylim_z = original_zoom
    _draw_original_on_ax(
        axes[1, 0],
        l_vals=l_c,
        sigma2_vals=sig_c,
        z_o=z_o_c,
        xlim=xlim_c,
        ylim=ylim_c,
        levels=levels,
        cmap=cmap,
        subtitle="original — coarse",
    )
    _draw_original_on_ax(
        axes[1, 1],
        l_vals=l_z,
        sigma2_vals=sig_z,
        z_o=z_o_z,
        xlim=xlim_z,
        ylim=ylim_z,
        levels=levels,
        cmap=cmap,
        subtitle="original — zoom (hw=0.15)",
    )

    clip_note = f" (clipped at -{span_below_peak:g})" if span_below_peak is not None else ""
    fig.suptitle(f"{figure_title}{clip_note}", fontsize=11)

    # Reserve the right margin for one shared colorbar so it never overlaps panels.
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.96))
    cbar_ax = fig.add_axes([0.90, 0.12, 0.025, 0.76])
    fig.colorbar(mappable, cax=cbar_ax, label="LML / n")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_original_l_sigma2_panel(
    *,
    l_vals: np.ndarray,
    sigma2_vals: np.ndarray,
    z_o: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    out_path: Path,
    levels: np.ndarray,
    panel_title: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    l_plot, z_plot = _ascending_l_for_omega_s_surface(l_vals, z_o)
    fig, ax = plt.subplots(figsize=(ARD_PANEL_INCHES, ARD_PANEL_INCHES), dpi=150)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.3")

    xv, yv = np.meshgrid(np.asarray(sigma2_vals, dtype=float), l_plot, indexing="xy")
    zm = np.ma.masked_invalid(z_plot)
    mappable = ax.contourf(xv, yv, zm, levels=levels, cmap=cmap, extend="both")
    ax.contour(xv, yv, zm, levels=8, colors="white", linewidths=0.9, alpha=0.85)

    idx, _best = _finite_argmax_2d(z_plot)
    if idx is not None:
        xb, yb = float(xv[idx]), float(yv[idx])
        ax.scatter([xb], [yb], s=40, c="red", edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("sigma^2 (outputscale)")
    ax.set_ylabel("lengthscale l")
    ax.set_title(panel_title or "")
    ax.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label="LML / n")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_transformed_omega_outputscale_panel(
    *,
    panel: dict,
    title: str,
    out_path: Path,
    levels: np.ndarray,
    panel_title: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(ARD_PANEL_INCHES, ARD_PANEL_INCHES), dpi=150)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.3")

    s_vals = panel["s_vals"]
    omega_vals = panel["omega_vals"]
    z_t = np.asarray(panel["transformed_surface"], dtype=float)
    xv_t, yv_t = np.meshgrid(s_vals, omega_vals, indexing="xy")
    xlim = panel.get("plot_s_xlim", panel["s_ylim"])
    ylim = panel.get("plot_omega_ylim", panel["omega_xlim"])

    zm = np.ma.masked_invalid(z_t)
    mappable = ax.contourf(xv_t, yv_t, zm, levels=levels, cmap=cmap, extend="both")
    ax.contour(xv_t, yv_t, zm, levels=8, colors="white", linewidths=0.9, alpha=0.85)

    idx, _best = _finite_argmax_2d(z_t)
    if idx is not None:
        xb, yb = float(xv_t[idx]), float(yv_t[idx])
        ax.scatter([xb], [yb], s=40, c="red", edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("outputscale (log10 σ²)")
    ax.set_ylabel("omega")
    ax.set_title(panel_title or "")
    ax.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label="LML / n")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_lml_1d_omega_outputscale_study(
    *,
    save_path: str | Path,
    problem_name: str,
    train_x: torch.Tensor,
    y_used: torch.Tensor,
    train_n: int,
    nugget: float = 0.0,
    n_grid: int = 201,
    s_bounds: tuple[float, float] = S_BOUNDS_DEFAULT,
    omega_bounds: tuple[float, float] = OMEGA_BOUNDS_DEFAULT,
    square_subplot_limits: bool = True,
    profile_constant_mean: bool = False,
    normalize_lml_by_n: bool = True,
    cholesky_jitter: float = 0.0,
    lml_clip_below_best: float | None = 10.0,
    lml_color_min_span: float = 0.25,
    zoom_lml_around_coarse_best: bool = True,
    omega_zoom_half_width: float | None = 0.15,
    s_zoom_half_width: float | None = 0.15,
    omega_zoom_fraction: float = 0.2,
    s_zoom_fraction: float = 0.2,
    figure_title: str | None = None,
    panel_title: str | None = None,
    include_original_space: bool = True,
    min_lengthscale: float = MIN_LENGTHSCALE_DEFAULT,
    max_lengthscale: float = MAX_LENGTHSCALE_DEFAULT,
    lengthscale_zoom_half_width: float = LENGTHSCALE_ZOOM_HALF_WIDTH_DEFAULT,
) -> dict:
    """Compute LML surfaces and save one 2×2 combined figure (coarse + zoom)."""
    t0 = time.perf_counter()
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = train_x.to(torch.float64)
    y_train = y_used.to(torch.float64)
    nugget = float(nugget)

    outer_omega = _sort_interval(omega_bounds)
    outer_s = _sort_interval(s_bounds)
    l_trainer = _trainer_l_bounds(outer_omega)
    outer_l = (
        max(float(min_lengthscale), float(l_trainer[0])),
        min(float(max_lengthscale), float(l_trainer[1])),
    )
    outer_sigma2 = (float(10.0 ** outer_s[0]), float(10.0 ** outer_s[1]))
    pairwise_sqdist = torch.cdist(x_train, x_train, p=2).pow(2)
    coarse_grid = _build_omega_s_grids(
        outer_omega,
        outer_s,
        n_grid=n_grid,
        outer_omega=outer_omega,
        outer_s=outer_s,
        square=bool(square_subplot_limits),
    )

    z_t = _compute_transformed_omega_s_surface(
        y_used=y_train,
        pairwise_sqdist=pairwise_sqdist,
        nugget=nugget,
        omega_vals=coarse_grid["omega_vals"],
        s_vals=coarse_grid["s_vals"],
        profile_constant_mean=profile_constant_mean,
        normalize_lml_by_n=normalize_lml_by_n,
        cholesky_jitter=cholesky_jitter,
    )
    panel = {"nugget": nugget, **coarse_grid, "transformed_surface": z_t}

    z_o_coarse = None
    if include_original_space:
        z_o_coarse = _compute_original_l_sigma2_surface(
            y_used=y_train,
            pairwise_sqdist=pairwise_sqdist,
            nugget=nugget,
            l_vals=coarse_grid["l_vals"],
            sigma2_vals=coarse_grid["sigma2_vals"],
            profile_constant_mean=profile_constant_mean,
            normalize_lml_by_n=normalize_lml_by_n,
            cholesky_jitter=cholesky_jitter,
        )

    run_tag = f"{problem_name}_n{train_n}_lml"
    span_below_peak = float(lml_clip_below_best) if lml_clip_below_best is not None else None

    if figure_title is None:
        pretty = problem_name.replace("_", " ")
        figure_title = f"{pretty} — LML change of variables (n={train_n} Sobol)"
    if panel_title:
        figure_title = f"{figure_title}\n{panel_title}"

    files: list[str] = []

    idx_t, best_t = _finite_argmax_2d(z_t)
    grid_optimum: dict[str, float] | None = None
    if idx_t is not None:
        grid_optimum = {
            "omega": float(coarse_grid["omega_vals"][idx_t[0]]),
            "outputscale": float(coarse_grid["s_vals"][idx_t[1]]),
            "lml": float(best_t),
        }

    np.savez_compressed(
        out_dir / f"{run_tag}_surfaces.npz",
        s_vals=coarse_grid["s_vals"],
        omega_vals=coarse_grid["omega_vals"],
        nugget=float(nugget),
        transformed_surface=z_t,
        original_surface=z_o_coarse if z_o_coarse is not None else np.array([]),
        train_n=int(train_n),
    )
    files.append(str(out_dir / f"{run_tag}_surfaces.npz"))

    rows = []
    for i_omega, omega in enumerate(coarse_grid["omega_vals"]):
        for i_s, s in enumerate(coarse_grid["s_vals"]):
            rows.append([s, omega, z_t[i_omega, i_s]])
    csv_path = out_dir / f"{run_tag}_transformed_long.csv"
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="outputscale,omega,lml",
        comments="",
    )
    files.append(str(csv_path))

    if zoom_lml_around_coarse_best and z_o_coarse is not None:
        hw_o = _resolve_omega_zoom_half_width(
            omega_zoom_half_width=omega_zoom_half_width,
            omega_zoom_fraction=float(omega_zoom_fraction),
            outer_omega=outer_omega,
        )
        hw_s = _resolve_s_zoom_half_width(
            s_zoom_half_width=s_zoom_half_width,
            s_zoom_fraction=float(s_zoom_fraction),
            outer_s=outer_s,
        )
        hw_l = float(lengthscale_zoom_half_width)

        cen_o, cen_s = _coarse_best_on_omega_s_panel(panel)
        plot_omega, plot_s = _zoom_pair_limits(
            cen_o, cen_s, hw_o, hw_s, outer_omega, outer_s, square=False,
        )
        ogrid = _build_omega_s_grids(
            plot_omega, plot_s, n_grid=n_grid,
            outer_omega=outer_omega, outer_s=outer_s, square=False,
        )
        z_zoom = _compute_transformed_omega_s_surface(
            y_used=y_train,
            pairwise_sqdist=pairwise_sqdist,
            nugget=nugget,
            omega_vals=ogrid["omega_vals"],
            s_vals=ogrid["s_vals"],
            profile_constant_mean=profile_constant_mean,
            normalize_lml_by_n=normalize_lml_by_n,
            cholesky_jitter=cholesky_jitter,
        )
        zoom_panel = {
            "nugget": nugget,
            **ogrid,
            "plot_s_xlim": plot_s,
            "plot_omega_ylim": plot_omega,
            "transformed_surface": z_zoom,
        }

        cen_l, cen_sig = _coarse_best_l_sigma2(
            z_o_coarse, coarse_grid["l_vals"], coarse_grid["sigma2_vals"]
        )
        plot_l, plot_sig = _zoom_pair_limits(
            cen_l, cen_sig, hw_l, hw_l, outer_l, outer_sigma2,
            square=True,
            min_x=float(min_lengthscale),
            max_x=float(max_lengthscale),
            min_y=0.0,
        )
        lgrid = _build_l_sigma2_grids(
            plot_l, plot_sig, n_grid=n_grid,
            outer_l=outer_l, outer_sigma2=outer_sigma2,
        )
        z_o_zoom = _compute_original_l_sigma2_surface(
            y_used=y_train,
            pairwise_sqdist=pairwise_sqdist,
            nugget=nugget,
            l_vals=lgrid["l_vals"],
            sigma2_vals=lgrid["sigma2_vals"],
            profile_constant_mean=profile_constant_mean,
            normalize_lml_by_n=normalize_lml_by_n,
            cholesky_jitter=cholesky_jitter,
        )
        original_zoom_data = (
            lgrid["l_vals"],
            lgrid["sigma2_vals"],
            z_o_zoom,
            _sort_interval(plot_sig),
            _sort_interval(plot_l),
        )

        shared_levels = _levels_for_figure(
            [z_t, z_zoom, z_o_coarse, z_o_zoom],
            span_below_peak=span_below_peak,
            min_span=float(lml_color_min_span),
        )
        levels_transformed = shared_levels
        levels_original = shared_levels
        sig_hi = float(np.max(coarse_grid["sigma2_vals"]))
        l_hi = float(np.max(coarse_grid["l_vals"]))

        transformed_coarse_png = out_dir / f"{run_tag}_transformed.png"
        _plot_transformed_omega_outputscale_panel(
            panel=panel,
            title=figure_title,
            out_path=transformed_coarse_png,
            levels=levels_transformed,
            panel_title="transformed — coarse",
        )
        files.append(str(transformed_coarse_png))

        transformed_zoom_png = out_dir / f"{run_tag}_zoom_transformed.png"
        _plot_transformed_omega_outputscale_panel(
            panel=zoom_panel,
            title=figure_title,
            out_path=transformed_zoom_png,
            levels=levels_transformed,
            panel_title="transformed — zoom (hw=0.15)",
        )
        files.append(str(transformed_zoom_png))

        original_coarse_png = out_dir / f"{run_tag}_original.png"
        _plot_original_l_sigma2_panel(
            l_vals=coarse_grid["l_vals"],
            sigma2_vals=coarse_grid["sigma2_vals"],
            z_o=z_o_coarse,
            xlim=(0.0, sig_hi),
            ylim=(0.0, l_hi),
            title=figure_title,
            out_path=original_coarse_png,
            levels=levels_original,
            panel_title="original — coarse",
        )
        files.append(str(original_coarse_png))

        original_zoom_png = out_dir / f"{run_tag}_zoom_original.png"
        _plot_original_l_sigma2_panel(
            l_vals=original_zoom_data[0],
            sigma2_vals=original_zoom_data[1],
            z_o=original_zoom_data[2],
            xlim=original_zoom_data[3],
            ylim=original_zoom_data[4],
            title=figure_title,
            out_path=original_zoom_png,
            levels=levels_original,
            panel_title="original — zoom (hw=0.15)",
        )
        files.append(str(original_zoom_png))

        combined_png = out_dir / f"{run_tag}_combined.png"
        _plot_combined_4panel_figure(
            out_path=combined_png,
            figure_title=figure_title,
            transformed_coarse_panel=panel,
            transformed_zoom_panel=zoom_panel,
            original_coarse=(
                coarse_grid["l_vals"],
                coarse_grid["sigma2_vals"],
                z_o_coarse,
                (0.0, sig_hi),
                (0.0, l_hi),
            ),
            original_zoom=original_zoom_data,
            levels=shared_levels,
            span_below_peak=span_below_peak,
        )
        files.append(str(combined_png))

        np.savez_compressed(
            out_dir / f"{run_tag}_zoom_surfaces.npz",
            s_vals=zoom_panel["s_vals"],
            omega_vals=zoom_panel["omega_vals"],
            transformed_surface=zoom_panel["transformed_surface"],
            l_vals=original_zoom_data[0],
            sigma2_vals=original_zoom_data[1],
            original_surface=original_zoom_data[2],
            hw_omega=float(hw_o),
            hw_s=float(hw_s),
            hw_l=float(hw_l),
        )
        files.append(str(out_dir / f"{run_tag}_zoom_surfaces.npz"))

    if grid_optimum is not None:
        best_csv = out_dir / f"{run_tag}_grid_maximum.csv"
        np.savetxt(
            best_csv,
            np.asarray(
                [[grid_optimum["omega"], grid_optimum["outputscale"], grid_optimum["lml"]]],
                dtype=np.float64,
            ),
            delimiter=",",
            header="omega,outputscale,lml",
            comments="",
        )
        files.append(str(best_csv))

    return {
        "save_dir": str(out_dir),
        "run_tag": run_tag,
        "problem_name": problem_name,
        "train_n": int(train_n),
        "files": files,
        "grid_optimum": grid_optimum,
        "timing_seconds": time.perf_counter() - t0,
    }


def replot_combined_from_npz(
    run_dir: str | Path,
    *,
    lml_clip_below_best: float | None = 10.0,
    lml_color_min_span: float = 0.25,
    figure_title: str | None = None,
    panel_title: str | None = None,
) -> str | None:
    """Regenerate the combined 4-panel figure from saved ``*_surfaces.npz`` files.

    Reads ``{run_tag}_surfaces.npz`` (coarse) and ``{run_tag}_zoom_surfaces.npz``
    (zoom) in ``run_dir`` and rewrites ``{run_tag}_combined.png`` with a single
    shared colorbar. Surfaces are not recomputed.
    """
    run_dir = Path(run_dir)
    coarse_files = [
        p for p in run_dir.glob("*_lml_surfaces.npz") if not p.name.endswith("_zoom_surfaces.npz")
    ]
    if not coarse_files:
        return None
    coarse_path = coarse_files[0]
    run_tag = coarse_path.name[: -len("_surfaces.npz")]
    zoom_path = run_dir / f"{run_tag}_zoom_surfaces.npz"
    if not zoom_path.is_file():
        return None

    coarse = np.load(coarse_path)
    zoom = np.load(zoom_path)

    span_below_peak = float(lml_clip_below_best) if lml_clip_below_best is not None else None

    c_s = np.asarray(coarse["s_vals"], dtype=float)
    c_omega = np.asarray(coarse["omega_vals"], dtype=float)
    z_t_coarse = np.asarray(coarse["transformed_surface"], dtype=float)
    z_o_coarse = np.asarray(coarse["original_surface"], dtype=float)
    c_sigma2 = np.power(10.0, c_s)
    c_l = np.power(10.0, -c_omega / 2.0)

    coarse_panel = {
        "s_vals": c_s,
        "omega_vals": c_omega,
        "transformed_surface": z_t_coarse,
        "s_ylim": (float(c_s.min()), float(c_s.max())),
        "omega_xlim": (float(c_omega.min()), float(c_omega.max())),
    }

    z_s = np.asarray(zoom["s_vals"], dtype=float)
    z_omega = np.asarray(zoom["omega_vals"], dtype=float)
    z_t_zoom = np.asarray(zoom["transformed_surface"], dtype=float)
    z_l = np.asarray(zoom["l_vals"], dtype=float)
    z_sigma2 = np.asarray(zoom["sigma2_vals"], dtype=float)
    z_o_zoom = np.asarray(zoom["original_surface"], dtype=float)

    zoom_panel = {
        "s_vals": z_s,
        "omega_vals": z_omega,
        "transformed_surface": z_t_zoom,
        "s_ylim": (float(z_s.min()), float(z_s.max())),
        "omega_xlim": (float(z_omega.min()), float(z_omega.max())),
        "plot_s_xlim": (float(z_s.min()), float(z_s.max())),
        "plot_omega_ylim": (float(z_omega.min()), float(z_omega.max())),
    }

    shared_levels = _levels_for_figure(
        [z_t_coarse, z_t_zoom, z_o_coarse, z_o_zoom],
        span_below_peak=span_below_peak,
        min_span=float(lml_color_min_span),
    )

    if figure_title is None:
        problem = run_tag
        n_label = ""
        parts = run_tag.split("_")
        if "lml" in parts:
            parts = parts[: parts.index("lml")]
        for i, tok in enumerate(parts):
            if tok.startswith("n") and tok[1:].isdigit():
                n_label = tok[1:]
                problem = "_".join(parts[:i])
                break
        else:
            problem = "_".join(parts)
        pretty = problem.replace("_", " ")
        figure_title = f"{pretty} — LML change of variables (n={n_label} Sobol)"
    if panel_title:
        figure_title = f"{figure_title}\n{panel_title}"

    sig_hi = float(np.max(c_sigma2))
    l_hi = float(np.max(c_l))
    combined_png = run_dir / f"{run_tag}_combined.png"
    _plot_combined_4panel_figure(
        out_path=combined_png,
        figure_title=figure_title,
        transformed_coarse_panel=coarse_panel,
        transformed_zoom_panel=zoom_panel,
        original_coarse=(c_l, c_sigma2, z_o_coarse, (0.0, sig_hi), (0.0, l_hi)),
        original_zoom=(
            z_l,
            z_sigma2,
            z_o_zoom,
            (float(z_sigma2.min()), float(z_sigma2.max())),
            (float(z_l.min()), float(z_l.max())),
        ),
        levels=shared_levels,
        span_below_peak=span_below_peak,
    )
    return str(combined_png)
