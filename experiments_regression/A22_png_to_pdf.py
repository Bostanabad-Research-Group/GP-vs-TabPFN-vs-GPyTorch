"""
Convert existing A22 PNG plots to sibling PDF files.

Does not delete or overwrite PNGs. Skips PDFs that already exist.
Does not re-run experiments — only wraps raster PNGs into PDF containers.

Usage:
  python A22_png_to_pdf.py
  python A22_png_to_pdf.py path/to/results/root
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parent / "results_1D" / "A22_regression_1D"


def png_to_pdf(png: Path, pdf: Path) -> None:
    with Image.open(png) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            rgb = background
        else:
            rgb = im.convert("RGB")
        rgb.save(pdf, "PDF", resolution=300.0)


def convert_root(root: Path) -> tuple[int, int, list[tuple[str, str]]]:
    pngs = sorted(root.rglob("*.png"))
    created = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    for png in pngs:
        pdf = png.with_suffix(".pdf")
        if pdf.exists():
            skipped += 1
            continue
        try:
            png_to_pdf(png, pdf)
            created += 1
            print(f"  + {pdf.relative_to(root)}")
        except Exception as e:
            errors.append((str(png), str(e)))
            print(f"  ! {png.relative_to(root)}: {e}")
    return created, skipped, errors


def main(argv: list[str]) -> None:
    roots = [Path(a) for a in argv[1:]] if len(argv) > 1 else [DEFAULT_ROOT]
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"Results root not found: {root}")
        print(f"Converting PNGs under {root.resolve()}")
        created, skipped, errors = convert_root(root)
        n_png = len(list(root.rglob("*.png")))
        n_pdf = len(list(root.rglob("*.pdf")))
        print(
            f"Done — created={created} skipped_existing={skipped} "
            f"errors={len(errors)} pngs={n_png} pdfs={n_pdf}"
        )


if __name__ == "__main__":
    main(sys.argv)
