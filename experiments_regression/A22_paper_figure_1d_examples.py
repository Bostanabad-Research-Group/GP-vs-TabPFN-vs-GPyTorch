"""
Build a paper-style 1D GP vs TabPFN figure from saved A22 results (no re-training).

Uses run 3 prediction curves from ``predictions.npz`` under each function folder,
optionally overlaying TabPFN v2.5 (tuned) from ``A22_regression_1D_tabpfn_tuned``.
Writes square, label-free panel PDFs + a shared legend + a LaTeX figure snippet into:

  results_1D/A22_regression_1D/1D_regression_figure/              (--no-tuned)
  results_1D/A22_regression_1D_tabpfn_tuned/1D_regression_figure_tuned/

Does not modify existing per-run PNGs/PDFs under each function's ``plots/`` folder.

Usage:
  python A22_paper_figure_1d_examples.py --math-labels
  python A22_paper_figure_1d_examples.py --math-labels --no-tuned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experimental_utils.a22_results_io import load_predictions_npz

RESULTS_ROOT = Path(__file__).resolve().parent / "results_1D" / "A22_regression_1D"
TUNED_ROOT = Path(__file__).resolve().parent / "results_1D" / "A22_regression_1D_tabpfn_tuned"
DEFAULT_OUT = RESULTS_ROOT / "1D_regression_figure"
DEFAULT_OUT_TUNED = TUNED_ROOT / "1D_regression_figure_tuned"

# Colors match experimental_utils/plot_tabpfn1d_comparison.py (+ tuned accent)
COLOR_TRAIN = "0.15"
COLOR_TRUE = "#D95F02"
COLOR_GP = "#1B9E77"
COLOR_PFN = "#7570B3"
COLOR_PFN_TUNED = "#E7298A"

# Order matches the intended left-to-right paper figure.
PANELS: list[dict] = [
    {
        "folder": "discontinuity",
        "file_stem": "a_discontinuity",
        "latex_label": r"Discontinuous Sine",
        "caption_short": "Discontinuous Sine",
    },
    {
        "folder": "triangle_wave",
        "file_stem": "b_triangle_wave",
        "latex_label": r"Triangle Wave",
        "caption_short": "Triangle Wave",
    },
    {
        "folder": "chirp",
        "file_stem": "c_chirp",
        "latex_label": r"Non-Stationary Chirp",
        "caption_short": "Non-Stationary Chirp",
    },
    {
        "folder": "localized_bump",
        "file_stem": "d_localized_bump",
        "latex_label": r"Localized Bump",
        "caption_short": "Localized Bump",
        "ylim_lo_extra": 0.2,
    },
    {
        "folder": "damped_sine",
        "file_stem": "e_damped_sine",
        "latex_label": r"Damped Sine",
        "caption_short": "Damped Sine",
    },
]

# Labels for \subcaption (letter numbering is added by subcaption, like the other paper tex).
LATEX_LABELS_MATH = {
    "discontinuity": r"Discontinuous Sine",
    "triangle_wave": r"Triangle Wave",
    "chirp": r"Non-Stationary Chirp",
    "localized_bump": r"Localized Bump",
    "damped_sine": r"Damped Sine",
}


def _style_square_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        bottom=True,
        left=True,
        top=False,
        right=False,
        length=3.0,
        width=0.6,
        direction="out",
    )
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.28, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")


def _y_limits(
    y_train: np.ndarray,
    y_true: np.ndarray,
    pad_frac: float = 0.12,
    lo_extra: float = 0.0,
) -> tuple[float, float]:
    ref = np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_true).ravel()])
    ref = ref[np.isfinite(ref)]
    lo, hi = float(np.min(ref)), float(np.max(ref))
    span = hi - lo
    pad = pad_frac * span if span > 0 else (abs(hi) * pad_frac or 1.0)
    return lo - pad - lo_extra, hi + pad


def plot_panel(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred_gp: np.ndarray | None,
    y_pred_tabpfn: np.ndarray | None,
    out_stem: Path,
    y_std_gp: np.ndarray | None = None,
    y_std_tabpfn: np.ndarray | None = None,
    y_pred_tabpfn_tuned: np.ndarray | None = None,
    y_std_tabpfn_tuned: np.ndarray | None = None,
    interval_z: float = 1.96,
    figsize: tuple[float, float] = (2.35, 2.35),
    dpi: int = 200,
    ylim_lo_extra: float = 0.0,
) -> list[Path]:
    """Save one square panel as PDF + PNG (no legend, no axis text)."""
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    order = np.argsort(x_test)
    xs = x_test[order]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(
        x_train,
        y_train,
        s=18,
        c=COLOR_TRAIN,
        alpha=0.85,
        zorder=5,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.plot(xs, y_true[order], color=COLOR_TRUE, linewidth=1.6, zorder=2)

    if y_pred_gp is not None:
        yg = np.asarray(y_pred_gp, dtype=np.float64).ravel()
        if y_std_gp is not None:
            sg = np.asarray(y_std_gp, dtype=np.float64).ravel()
            ax.fill_between(
                xs,
                (yg - interval_z * sg)[order],
                (yg + interval_z * sg)[order],
                color=COLOR_GP,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            xs,
            yg[order],
            color=COLOR_GP,
            linewidth=1.4,
            linestyle="--",
            zorder=3,
            alpha=0.95,
        )

    if y_pred_tabpfn is not None:
        yp = np.asarray(y_pred_tabpfn, dtype=np.float64).ravel()
        if y_std_tabpfn is not None:
            sp = np.asarray(y_std_tabpfn, dtype=np.float64).ravel()
            ax.fill_between(
                xs,
                (yp - interval_z * sp)[order],
                (yp + interval_z * sp)[order],
                color=COLOR_PFN,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(xs, yp[order], color=COLOR_PFN, linewidth=1.4, zorder=4, alpha=0.95)

    if y_pred_tabpfn_tuned is not None:
        yt = np.asarray(y_pred_tabpfn_tuned, dtype=np.float64).ravel()
        if y_std_tabpfn_tuned is not None:
            st = np.asarray(y_std_tabpfn_tuned, dtype=np.float64).ravel()
            ax.fill_between(
                xs,
                (yt - interval_z * st)[order],
                (yt + interval_z * st)[order],
                color=COLOR_PFN_TUNED,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(xs, yt[order], color=COLOR_PFN_TUNED, linewidth=1.4, zorder=4, alpha=0.95)

    ax.set_ylim(*_y_limits(y_train, y_true, lo_extra=ylim_lo_extra))
    # Keep x span tight to the test grid (usually [-0.5, 0.5]).
    ax.set_xlim(float(np.min(xs)), float(np.max(xs)))
    _style_square_axes(ax)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

    written: list[Path] = []
    for ext in (".pdf", ".png"):
        fp = out_stem.with_suffix(ext)
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        written.append(fp)
    plt.close(fig)
    return written


def save_shared_legend(
    out_stem: Path,
    *,
    dpi: int = 200,
    include_pi: bool = True,
    include_tuned: bool = False,
    separate_pi: bool = False,
) -> list[Path]:
    """Horizontal legend-only figure (Train / True / GP / TabPFN [+ tuned]).

    ``separate_pi=False`` (default): one combined handle per model
    ``Mean line / PI shade`` labeled ``Mean/95% PI`` (single row).

    ``separate_pi=True``: Mean line and 95% PI patch are separate entries
    (two-row layout).
    """
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch, Rectangle

    class _MeanPI:
        def __init__(self, color: str, linestyle: str = "-"):
            self.color = color
            self.linestyle = linestyle

    class _HandlerMeanPI(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            from matplotlib.text import Text

            # Matplotlib handle box starts at (-xdescent, -ydescent).
            # Layout left → right: [Mean line] [/] [PI shade].
            x0 = -xdescent
            y0 = -ydescent
            y = y0 + height / 2.0
            line_w = width * 0.38
            slash_x = x0 + width * 0.48
            patch_x = x0 + width * 0.58
            patch_w = width * 0.38
            patch_h = height * 0.85

            line = _Line2D(
                [x0, x0 + line_w],
                [y, y],
                color=orig_handle.color,
                linewidth=2.4,
                linestyle=orig_handle.linestyle,
                solid_capstyle="butt",
                dash_capstyle="butt",
            )
            line.set_transform(trans)

            slash = Text(
                slash_x,
                y,
                "/",
                fontsize=fontsize,
                color="0.15",
                ha="center",
                va="center",
                fontweight="normal",
            )
            slash.set_transform(trans)

            patch = Rectangle(
                (patch_x, y0 + (height - patch_h) / 2.0),
                patch_w,
                patch_h,
                facecolor=orig_handle.color,
                edgecolor="none",
                alpha=0.35,
            )
            patch.set_transform(trans)
            return [line, slash, patch]

    train_h = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=COLOR_TRAIN,
        markeredgecolor="white",
        markeredgewidth=0.3,
        markersize=7.7,
        label="Train",
    )
    true_h = Line2D([0], [0], color=COLOR_TRUE, linewidth=2.2, label=r"True $f(x)$")

    if include_pi and separate_pi:
        pfn_mean = "TabPFN v2.5 (default) Mean" if include_tuned else "TabPFN v2.5 Mean"
        pfn_pi = "TabPFN v2.5 (default) 95% PI" if include_tuned else "TabPFN v2.5 95% PI"
        # Matplotlib packs legend entries column-major (top→bottom, then next col).
        # Desired grid:
        #   True f(x) | GP Mean | TabPFN (default) Mean | TabPFN (tuned) Mean
        #   Train     | GP PI   | TabPFN (default) PI   | TabPFN (tuned) PI
        if include_tuned:
            handles = [
                true_h,
                train_h,
                Line2D([0], [0], color=COLOR_GP, linewidth=2.2, linestyle="--", label="GP+ Mean"),
                Patch(facecolor=COLOR_GP, edgecolor="none", alpha=0.35, label="GP+ 95% PI"),
                Line2D([0], [0], color=COLOR_PFN, linewidth=2.2, label=pfn_mean),
                Patch(facecolor=COLOR_PFN, edgecolor="none", alpha=0.35, label=pfn_pi),
                Line2D([0], [0], color=COLOR_PFN_TUNED, linewidth=2.2, label="TabPFN v2.5 (tuned) Mean"),
                Patch(
                    facecolor=COLOR_PFN_TUNED,
                    edgecolor="none",
                    alpha=0.35,
                    label="TabPFN v2.5 (tuned) 95% PI",
                ),
            ]
            ncol = 4
            fig_w = 12.5
            fontsize = 9
        else:
            handles = [
                true_h,
                train_h,
                Line2D([0], [0], color=COLOR_GP, linewidth=2.2, linestyle="--", label="GP+ Mean"),
                Patch(facecolor=COLOR_GP, edgecolor="none", alpha=0.35, label="GP+ 95% PI"),
                Line2D([0], [0], color=COLOR_PFN, linewidth=2.2, label=pfn_mean),
                Patch(facecolor=COLOR_PFN, edgecolor="none", alpha=0.35, label=pfn_pi),
            ]
            ncol = 3
            fig_w = 10.5
            fontsize = 10
        fig_h = 0.85
        labels = None
        handler_map = None
    elif include_pi:
        gp_h = _MeanPI(COLOR_GP, linestyle="--")
        pfn_h = _MeanPI(COLOR_PFN)
        pfn_label = "TabPFN v2.5 (default) Mean/95% PI" if include_tuned else "TabPFN v2.5 Mean/95% PI"
        handles = [train_h, true_h, gp_h, pfn_h]
        labels = ["Train", r"True $f(x)$", "GP+ Mean/95% PI", pfn_label]
        handler_map = {_MeanPI: _HandlerMeanPI()}
        if include_tuned:
            handles.append(_MeanPI(COLOR_PFN_TUNED))
            labels.append("TabPFN v2.5 (tuned) Mean/95% PI")
            ncol = 5
            fig_w = 15.0
            fontsize = 9
        else:
            ncol = 4
            fig_w = 11.5
            fontsize = 10
        fig_h = 0.45
    else:
        pfn_label = "TabPFN v2.5 (default) Mean" if include_tuned else "TabPFN v2.5 Mean"
        handles = [
            train_h,
            true_h,
            Line2D([0], [0], color=COLOR_GP, linewidth=2.2, linestyle="--", label="GP+ Mean"),
            Line2D([0], [0], color=COLOR_PFN, linewidth=2.2, label=pfn_label),
        ]
        labels = None
        handler_map = None
        if include_tuned:
            handles.append(
                Line2D([0], [0], color=COLOR_PFN_TUNED, linewidth=2.2, label="TabPFN v2.5 (tuned) Mean")
            )
        ncol = 4 if not include_tuned else 5
        fig_w = 7.9 if not include_tuned else 12.5
        fig_h = 0.45
        fontsize = 9.5 if include_tuned else 10

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    legend_kwargs = dict(
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.6 if separate_pi else 3.4,
        columnspacing=1.15,
        handletextpad=0.45,
        labelspacing=0.55 if separate_pi else 0.5,
    )
    if handler_map is not None:
        legend_kwargs["handler_map"] = handler_map
        fig.legend(handles, labels, **legend_kwargs)
    else:
        fig.legend(handles=handles, **legend_kwargs)

    written: list[Path] = []
    for ext in (".pdf", ".png"):
        fp = out_stem.with_suffix(ext)
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        written.append(fp)
    plt.close(fig)
    return written


def _panel_latex_label(panel: dict, *, math_labels: bool) -> str:
    if math_labels:
        return LATEX_LABELS_MATH.get(panel["folder"], panel["latex_label"])
    return panel["latex_label"]


def write_latex(
    out_dir: Path,
    panels: list[dict],
    *,
    math_labels: bool = False,
    include_tuned: bool = False,
    legend_stem: str = "legend_1d_examples",
    tex_name: str = "fig_1d_examples.tex",
    fig_label: str = "fig:1d_examples",
) -> Path:
    """Write one figure snippet matching the paper's minipage + \\subcaption style."""
    tex_path = out_dir / tex_name
    labels = [_panel_latex_label(p, math_labels=math_labels) for p in panels]
    # Prefix like logscale_comparison_regression.tex: folder/file.pdf relative to project root.
    fig_dir = out_dir.name
    legend_w = "0.95" if include_tuned else "0.72"
    if include_tuned:
        caption_body = (
            r"One-dimensional regression examples with $n{=}20$ training points: true $f(x)$, "
            r"GP+, TabPFN v2.5 (default), and TabPFN v2.5 (tuned) mean predictions with "
            r"95\% prediction intervals are shown. Axes span "
            r"$x\in[-0.5,0.5]$, with tick labels omitted for compactness. The defining equations for each "
            r"example are given in the project repository~\cite{GithubRepo}."
        )
    else:
        caption_body = (
            r"One-dimensional regression examples with $n{=}20$ training points: true $f(x)$, "
            r"GP+, and TabPFN v2.5 mean predictions with 95\% prediction intervals are shown. Axes span "
            r"$x\in[-0.5,0.5]$, with tick labels omitted for compactness. The defining equations for each "
            r"example are given in the project repository~\cite{GithubRepo}."
        )
    lines = [
        r"% Auto-generated by A22_paper_figure_1d_examples.py",
        r"% Same pattern as logscale_comparison_regression.tex: minipage + \subcaption",
        r"% Requires: \usepackage{graphicx}, \usepackage{subcaption}",
        rf"% Paths assume project root contains folder `{fig_dir}/` (same as other paper tex files).",
        r"",
        r"\begin{figure*}[hptb]",
        r"    \centering",
        r"    \captionsetup{font=footnotesize}",
        rf"    \includegraphics[width={legend_w}\textwidth]{{{fig_dir}/{legend_stem}.pdf}}\\[0.35em]",
    ]
    for i, panel in enumerate(panels):
        sep = "\n    \\hfill" if i < len(panels) - 1 else ""
        sub_label = f"{fig_label}_{panel['folder']}"
        img = f"{fig_dir}/{panel['file_stem']}.pdf"
        lines.extend(
            [
                r"    \begin{minipage}[b]{0.185\textwidth}",
                r"        \centering",
                rf"        \includegraphics[width=\textwidth]{{{img}}}",
                f"        \\subcaption{{{labels[i]}}}",
                f"        \\label{{{sub_label}}}",
                f"    \\end{{minipage}}{sep}",
            ]
        )
    lines.extend(
        [
            rf"    \caption{{{caption_body}}}",
            rf"    \label{{{fig_label}}}",
            r"\end{figure*}",
            r"",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def build_combined_preview(panel_pngs: list[Path], legend_png: Path, out_stem: Path) -> list[Path]:
    """Optional single-image preview (legend + 5 panels) for quick inspection."""
    from PIL import Image

    panels = [Image.open(p).convert("RGB") for p in panel_pngs]
    legend = Image.open(legend_png).convert("RGB")
    target_h = min(im.height for im in panels)
    resized = []
    for im in panels:
        w = int(round(im.width * (target_h / im.height)))
        resized.append(im.resize((w, target_h), Image.Resampling.LANCZOS))
    gap = 12
    row_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    legend_w = int(round(row_w * (0.95 if legend.width > row_w * 0.8 else 0.72)))
    legend_h = int(round(legend.height * (legend_w / legend.width)))
    legend_r = legend.resize((legend_w, legend_h), Image.Resampling.LANCZOS)

    canvas_h = legend_h + 16 + target_h + 8
    canvas = Image.new("RGB", (row_w, canvas_h), (255, 255, 255))
    canvas.paste(legend_r, ((row_w - legend_w) // 2, 0))
    x = 0
    y = legend_h + 16
    for im in resized:
        canvas.paste(im, (x, y))
        x += im.width + gap

    written: list[Path] = []
    png = out_stem.with_suffix(".png")
    canvas.save(png)
    written.append(png)
    pdf = out_stem.with_suffix(".pdf")
    canvas.save(pdf, "PDF", resolution=200.0)
    written.append(pdf)
    for im in panels:
        im.close()
    legend.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build A22 paper 1D example figure from saved results")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--tuned-root", type=Path, default=TUNED_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output folder for panel PDFs/PNGs + legend + LaTeX. "
            f"Default: {DEFAULT_OUT_TUNED.name}/ under --tuned-root when including tuned, "
            f"else {DEFAULT_OUT}."
        ),
    )
    parser.add_argument("--run", type=int, default=3, help="1-based run index (default: 3)")
    parser.add_argument("--no-preview", action="store_true", help="Skip combined preview image")
    parser.add_argument(
        "--math-labels",
        action="store_true",
        help="Use equation-style / display subcaption labels in the LaTeX snippets",
    )
    parser.add_argument(
        "--no-tuned",
        action="store_true",
        help="Omit TabPFN v2.5 (tuned) overlay (default: include when tuned npz exists)",
    )
    args = parser.parse_args()

    if args.run < 1:
        raise SystemExit("--run must be >= 1")

    include_tuned = not args.no_tuned
    if args.out_dir is None:
        out_dir = DEFAULT_OUT_TUNED if include_tuned else DEFAULT_OUT
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_idx = args.run - 1  # 0-based into runs list

    print(f"Writing paper figure assets to {out_dir.resolve()}")
    print(f"Using run {args.run} (index {run_idx})")
    if include_tuned:
        print(f"Including tuned TabPFN from {args.tuned_root}")

    panel_pngs: list[Path] = []
    any_pi = False
    any_tuned = False
    for panel in PANELS:
        result_dir = args.results_root / panel["folder"]
        x_test, y_true, runs, meta = load_predictions_npz(result_dir)
        if run_idx >= len(runs):
            raise SystemExit(f"{panel['folder']}: only {len(runs)} runs, cannot use run {args.run}")
        run = runs[run_idx]

        y_pred_tuned = None
        y_std_tuned = None
        if include_tuned:
            tuned_dir = args.tuned_root / panel["folder"]
            if not (tuned_dir / "predictions.npz").is_file():
                raise SystemExit(f"Missing tuned predictions.npz for {panel['folder']}: {tuned_dir}")
            _, _, tuned_runs, _ = load_predictions_npz(tuned_dir)
            if run_idx >= len(tuned_runs):
                raise SystemExit(f"{panel['folder']} tuned: only {len(tuned_runs)} runs")
            tuned_run = tuned_runs[run_idx]
            y_pred_tuned = tuned_run.get("y_pred_tabpfn")
            y_std_tuned = tuned_run.get("y_std_tabpfn")
            any_tuned = y_pred_tuned is not None

        has_pi = (
            run.get("y_std_gp") is not None
            or run.get("y_std_tabpfn") is not None
            or y_std_tuned is not None
        )
        any_pi = any_pi or has_pi
        stem = out_dir / panel["file_stem"]
        written = plot_panel(
            x_train=run["x_train"],
            y_train=run["y_train"],
            x_test=x_test,
            y_true=y_true,
            y_pred_gp=run.get("y_pred_gp"),
            y_pred_tabpfn=run.get("y_pred_tabpfn"),
            y_std_gp=run.get("y_std_gp"),
            y_std_tabpfn=run.get("y_std_tabpfn"),
            y_pred_tabpfn_tuned=y_pred_tuned,
            y_std_tabpfn_tuned=y_std_tuned,
            out_stem=stem,
            ylim_lo_extra=float(panel.get("ylim_lo_extra", 0.0)),
        )
        panel_pngs.append(next(p for p in written if p.suffix == ".png"))
        pi_note = " [+PI]" if has_pi else " [no std in npz]"
        tuned_note = " [+tuned]" if y_pred_tuned is not None else ""
        print(f"  [{panel['caption_short']}]{pi_note}{tuned_note} {meta.get('title', '')}")
        for p in written:
            print(f"    -> {p.name}")

    legend_paths = save_shared_legend(
        out_dir / "legend_1d_examples",
        include_pi=any_pi,
        include_tuned=any_tuned,
    )
    legend_png = next(p for p in legend_paths if p.suffix == ".png")
    for p in legend_paths:
        print(f"  [legend] -> {p.name}")

    legend_v2_paths = save_shared_legend(
        out_dir / "legend_1d_examples_v2",
        include_pi=any_pi,
        include_tuned=any_tuned,
        separate_pi=True,
    )
    legend_v2_png = next(p for p in legend_v2_paths if p.suffix == ".png")
    for p in legend_v2_paths:
        print(f"  [legend v2] -> {p.name}")

    tex1 = write_latex(
        out_dir,
        PANELS,
        math_labels=args.math_labels,
        include_tuned=any_tuned,
    )
    print(f"  [latex] -> {tex1.name}")

    tex2 = write_latex(
        out_dir,
        PANELS,
        math_labels=args.math_labels,
        include_tuned=any_tuned,
        legend_stem="legend_1d_examples_v2",
        tex_name="fig_1d_examples_v2.tex",
        fig_label="fig:1d_examples_v2",
    )
    print(f"  [latex v2] -> {tex2.name}")

    if not args.no_preview:
        preview = build_combined_preview(panel_pngs, legend_png, out_dir / "fig_1d_examples_preview")
        for p in preview:
            print(f"  [preview] -> {p.name}")
        preview_v2 = build_combined_preview(
            panel_pngs, legend_v2_png, out_dir / "fig_1d_examples_v2_preview"
        )
        for p in preview_v2:
            print(f"  [preview v2] -> {p.name}")

    print("Done.")


if __name__ == "__main__":
    main()
