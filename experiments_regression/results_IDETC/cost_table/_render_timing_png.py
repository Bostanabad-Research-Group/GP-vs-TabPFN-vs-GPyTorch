"""One-off: render timing_table_LOO.tex to PNG for visual confirmation."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table

HERE = Path(__file__).resolve().parent
TEX_PATH = HERE / "timing_table_LOO.tex"
OUT_PATH = HERE / "timing_table_LOO.png"


def clean(s: str) -> str:
    s = s.strip()
    s = s.replace(r"$\pm$", "±")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace(r"D$_x$", "Dx")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return s.strip()


def main() -> None:
    text = TEX_PATH.read_text(encoding="utf-8")
    data_rows: list[list[str]] = []
    for line in text.splitlines():
        raw = line.strip()
        if r"\\[2pt]" not in raw:
            continue
        raw = raw.replace(r"\\[2pt]", "")
        cells = [clean(c) for c in raw.split("&")]
        if len(cells) == 16:
            data_rows.append(cells)

    col_top = [
        "",
        "",
        "",
        "",
        "GP+",
        "",
        "GP+ (PE)",
        "",
        "GP+ (LOO)",
        "",
        "PFN 2.5",
        "",
        "PFN 2.0",
        "",
        "GPyTorch",
        "",
    ]
    col_lab = [
        "Problem",
        "N",
        "Dx",
        "Noise",
        "Train",
        "Pred",
        "Train",
        "Pred",
        "Train",
        "Pred",
        "Fit",
        "Inf",
        "Fit",
        "Inf",
        "Train",
        "Pred",
    ]

    n_rows = len(data_rows) + 2
    fig, ax = plt.subplots(figsize=(20, 0.28 * n_rows + 1.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table = Table(ax, bbox=[0.005, 0.02, 0.99, 0.88])
    widths = [0.09, 0.05, 0.035, 0.045] + [0.065] * 12
    s = sum(widths)
    widths = [w / s for w in widths]

    for j, (lab, w) in enumerate(zip(col_top, widths)):
        cell = table.add_cell(
            0, j, width=w, height=0.04, text=lab, loc="center",
            facecolor="#2f3e46", edgecolor="#1a1a1a",
        )
        cell.get_text().set_color("white")
        cell.get_text().set_fontsize(7)
        cell.get_text().set_fontweight("bold")

    for j, (lab, w) in enumerate(zip(col_lab, widths)):
        cell = table.add_cell(
            1, j, width=w, height=0.035, text=lab, loc="center",
            facecolor="#52796f", edgecolor="#1a1a1a",
        )
        cell.get_text().set_color("white")
        cell.get_text().set_fontsize(6.5)
        cell.get_text().set_fontweight("bold")

    for i, row in enumerate(data_rows):
        bg = "#f0f4f3" if (i // 4) % 2 else "#ffffff"
        for j, (val, w) in enumerate(zip(row, widths)):
            cell = table.add_cell(
                i + 2, j, width=w, height=0.028, text=val,
                loc="left" if j == 0 else "center",
                facecolor=bg, edgecolor="#b0b0b0",
            )
            cell.get_text().set_fontsize(5.5)
            if j == 0 and val:
                cell.get_text().set_fontweight("bold")

    ax.add_table(table)
    fig.suptitle(
        "timing_table_LOO.tex (restored) — problem name repeats on each 5Dx / 20Dx block",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    for r in data_rows[:6]:
        print(repr(r[:4]))


if __name__ == "__main__":
    main()
