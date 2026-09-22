"""Bayesian optimization summary figure and tables.

    python plot_summary.py
    python plot_summary.py --source results --per-problem-plots
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_BO import collect_runs, plot_problem_comparison
from plot_BO_IDETC import (
    IDETC_PROBLEMS,
    MAXIMIZATION_PROBLEMS,
    NOISE_HIGH,
    NOISE_LOW,
    plot_idetc_figures,
)

HERE = Path(__file__).resolve().parent
MODEL_FOLDERS = [
    ("GP+", "GP+"),
    ("PFN_V2.0", "PFN 2.0"),
    ("PFN_V2.5", "PFN 2.5"),
]


def _root(source: str) -> Path:
    return HERE / ("results_paper" if source == "paper" else "results")


def _models(root: Path) -> list[tuple[Path, str]]:
    found = []
    for folder, label in MODEL_FOLDERS:
        path = root / folder
        if path.is_dir():
            found.append((path, label))
    return found


def _flip(runs, do_flip: bool):
    if not do_flip:
        return runs
    return [(run_id, [-v for v in hist], t) for run_id, hist, t in runs]


def write_summary(source: str = "paper", per_problem: bool = False) -> Path:
    root = _root(source)
    models = _models(root)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    if not models:
        print(f"No BO model folders under {root}")
        return summary

    plot_idetc_figures(models, out_dir=summary, use_clean_y=False)
    final_src = summary / "BO_IDETC_noise0.08.png"
    if not final_src.exists():
        final_src = summary / "BO_IDETC_noise0.002.png"
    if final_src.exists():
        dest = summary / "BO_final.png"
        shutil.copyfile(final_src, dest)
        print(f"Saved {dest}")

    frames = []
    loaded = []
    for path, label in models:
        data, _stems = collect_runs(path, use_clean_y=False)
        loaded.append((label, data))

    for folder, dim_key, title in IDETC_PROBLEMS:
        is_max = (folder, dim_key) in MAXIMIZATION_PROBLEMS
        for noise_key, noise_label in ((NOISE_LOW, "0.002"), (NOISE_HIGH, "0.08")):
            key = (folder, dim_key, noise_key)
            per_models = []
            for label, data in loaded:
                runs = data.get(key, [])
                if not runs:
                    continue
                finals = [float(hist[-1]) for _rid, hist, _t in runs if hist]
                if not finals:
                    continue
                arr = np.asarray(finals, dtype=float)
                frames.append(
                    {
                        "problem": title,
                        "folder": folder,
                        "dim": dim_key or "fixed",
                        "noise": noise_label,
                        "goal": "maximize" if is_max else "minimize",
                        "model": label,
                        "n_runs": int(arr.size),
                        "final_best_median": float(np.median(arr)),
                        "final_best_std": float(np.std(arr)),
                        "final_best_mean": float(np.mean(arr)),
                    }
                )
                per_models.append((label, _flip(runs, is_max)))
            if per_problem and per_models:
                out = summary / "per_problem" / f"{title.replace(' ', '_')}_noise{noise_label}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                plot_problem_comparison(per_models, f"{title} (noise {noise_label})", out)

    if frames:
        frame = pd.DataFrame(frames)
        summary.mkdir(parents=True, exist_ok=True)
        frame.to_csv(summary / "bo_summary.csv", index=False)
        _write_markdown(frame, summary / "bo_summary.md")
        if per_problem:
            table_dir = summary / "tables"
            table_dir.mkdir(parents=True, exist_ok=True)
            for problem, part in frame.groupby("problem"):
                slug = str(problem).replace(" ", "_")
                _write_markdown(part, table_dir / f"{slug}.md")
        print(f"Saved {summary / 'bo_summary.md'}")
    else:
        print("No BO trajectories found for the summary table.")
    plt.close("all")
    return summary


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "Final best observed value. Median ± std across runs. "
        "Maximize problems are Buckling and Borehole. The figure plots the negative of those values.",
        "",
        "| Problem | Noise | Goal | Model | Runs | Final best |",
        "|---|---:|---|---|---:|---:|",
    ]
    ordered = frame.sort_values(["problem", "noise", "model"])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.problem} | {row.noise} | {row.goal} | {row.model} | {row.n_runs} | "
            f"{row.final_best_median:.4g} ± {row.final_best_std:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="BO summary figure and tables.")
    parser.add_argument("--source", choices=("paper", "results"), default="paper")
    parser.add_argument("--per-problem-plots", action="store_true")
    args = parser.parse_args()
    write_summary(args.source, args.per_problem_plots)


if __name__ == "__main__":
    main()
