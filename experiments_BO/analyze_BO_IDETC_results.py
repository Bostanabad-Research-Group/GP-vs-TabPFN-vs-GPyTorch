"""
Analyze BO results for the same problems used in `plot_BO_IDETC_overleaf.py`.

Computes mean / median / std over the *final* best-y values across runs, for:
  - noisy metric: final(best_y_history)
  - clean metric: final(best_y_clean_history) (falls back to best_y_history if absent)

For maximization problems (Buckling, Borehole), the Overleaf plotting code flips to `-y`
to keep the visual trend consistent. This script matches that convention so that
"lower is better" for all problems in the reported tables.

Run:
  python experiments_BO/analyze_BO_IDETC_results.py

Output:
  - prints markdown tables to stdout (no plotting)
  - writes markdown to: experiments_BO/results_BO/gpVpfn_BO_summary/bo_idetc_results_summary.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_BO import RunTuple, collect_runs
from plot_BO_IDETC import IDETC_PROBLEMS, MAXIMIZATION_PROBLEMS, NOISE_HIGH, NOISE_LOW
from plot_BO_IDETC_overleaf import (
    OVERLEAF_MODELS,
    PROBLEM_DIM_INFO,
    PROBLEM_NAME_OVERRIDES,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_MD = SCRIPT_DIR / "results_BO" / "gpVpfn_BO_summary" / "bo_idetc_results_summary.md"


@dataclass(frozen=True)
class SummaryStats:
    n: int
    mean: float
    median: float
    std: float
    vmin: float
    vmax: float


@dataclass(frozen=True)
class IterStats:
    n: int
    mean_iters: float
    median_iters: float


def _safe_stats(values: list[float]) -> SummaryStats | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    arr = np.asarray(finite, dtype=float)
    return SummaryStats(
        n=int(arr.size),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=0)),
        vmin=float(np.min(arr)),
        vmax=float(np.max(arr)),
    )


def _iter_stats(runs: list[RunTuple]) -> IterStats | None:
    lengths = [len(hist) for _rid, hist, _t in runs if hist]
    if not lengths:
        return None
    arr = np.asarray(lengths, dtype=float)
    return IterStats(
        n=int(arr.size),
        mean_iters=float(np.mean(arr)),
        median_iters=float(np.median(arr)),
    )


def _final_values(runs: list[RunTuple]) -> list[float]:
    out: list[float] = []
    for _run_id, hist, _t in runs:
        if not hist:
            continue
        out.append(float(hist[-1]))
    return out


def _fmt(x: float) -> str:
    # Compact formatting that still handles large-magnitude problems nicely.
    # Use scientific notation only when necessary.
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-3):
        # Scientific with 2 decimals -> ~3 significant digits, shorter strings.
        return f"{x:.2e}"
    # Standard notation with up to 4 significant digits to avoid wide cells.
    return f"{x:.4g}"


def _stats_cell(stats: SummaryStats | None) -> str:
    if stats is None:
        return ("—", "—", "—", "—", "—", "—")
    return (
        _fmt(stats.median),
        _fmt(stats.mean),
        _fmt(stats.std),
        _fmt(stats.vmin),
        _fmt(stats.vmax),
        str(stats.n),
    )


def _winner_labels(
    label_to_stats: dict[str, SummaryStats | None],
) -> list[str]:
    # Winner = lowest median (lower is better after sign convention).
    medians = {k: v.median for k, v in label_to_stats.items() if v is not None and np.isfinite(v.median)}
    if not medians:
        return []
    best = min(medians.values())
    eps = 1e-12
    return [k for k, m in medians.items() if abs(m - best) <= eps]


def _problem_display(folder: str, dim_key: str, subtitle: str) -> tuple[str, str]:
    name = PROBLEM_NAME_OVERRIDES.get((folder, dim_key), subtitle)
    dim = PROBLEM_DIM_INFO.get((folder, dim_key), dim_key or "")
    return name, dim


def _collect_by_model(models: list[tuple[Path, str]], use_clean_y: bool) -> list[dict[tuple[str, str, str], list[RunTuple]]]:
    all_data: list[dict[tuple[str, str, str], list[RunTuple]]] = []
    for model_dir, _label in models:
        data, _stems = collect_runs(model_dir, use_clean_y=use_clean_y)
        all_data.append(data)
    return all_data


def build_markdown() -> str:
    models = [(SCRIPT_DIR / rel, label) for rel, label in OVERLEAF_MODELS]
    model_labels = [label for _p, label in models]

    all_data_noisy = _collect_by_model(models, use_clean_y=False)
    all_data_clean = _collect_by_model(models, use_clean_y=True)

    noise_configs = [
        (NOISE_LOW, "0.002"),
        (NOISE_HIGH, "0.08"),
    ]

    # Global win counters (median-based) across all problems and noise configs.
    global_wins_noisy: dict[str, int] = {label: 0 for label in model_labels}
    global_wins_clean: dict[str, int] = {label: 0 for label in model_labels}

    # Per-run win counters across all problems/noise configs (compare run i vs run i only).
    per_run_wins: dict[str, int] = {label: 0 for label in model_labels}

    lines: list[str] = []
    lines.append("## BO results summary (final best values)")
    lines.append("")
    lines.append("- **What is summarized**: for each (problem, noise, model), stats over runs of the *final* value of best-so-far.")
    lines.append("- **Metrics**: `final_best_y` (noisy) and `final_best_y_clean` (clean, fallback to noisy if missing).")
    lines.append("- **Direction**: lower is better everywhere; for maximization problems we report `-y` (matching `plot_BO_IDETC_overleaf.py`).")
    lines.append("")

    for noise_key, noise_label in noise_configs:
        lines.append(f"### Noise = {noise_label}")
        lines.append("")

        # Per-noise win counters (should max out at 8 problems).
        wins_noisy: dict[str, int] = {label: 0 for label in model_labels}
        wins_clean: dict[str, int] = {label: 0 for label in model_labels}

        for folder, dim_key, subtitle in IDETC_PROBLEMS:
            name, dim_caption = _problem_display(folder, dim_key, subtitle)
            key = (folder, dim_key, noise_key)
            is_max = (folder, dim_key) in MAXIMIZATION_PROBLEMS
            metric_name = "Best (-y)" if is_max else "Best y"

            # Compute per-model stats (noisy + clean) + iteration stats.
            per_model_noisy: dict[str, SummaryStats | None] = {}
            per_model_clean: dict[str, SummaryStats | None] = {}
            per_model_iters: dict[str, IterStats | None] = {}

            for i, label in enumerate(model_labels):
                noisy_runs = all_data_noisy[i].get(key, [])
                clean_runs = all_data_clean[i].get(key, [])

                noisy_vals = _final_values(noisy_runs)
                clean_vals = _final_values(clean_runs)

                if is_max:
                    noisy_vals = [-v for v in noisy_vals]
                    clean_vals = [-v for v in clean_vals]

                per_model_noisy[label] = _safe_stats(noisy_vals)
                per_model_clean[label] = _safe_stats(clean_vals)
                per_model_iters[label] = _iter_stats(noisy_runs)

            winners_n = _winner_labels(per_model_noisy)
            winners_c = _winner_labels(per_model_clean)
            for w in winners_n:
                wins_noisy[w] += 1
            for w in winners_c:
                wins_clean[w] += 1
                # Also accumulate into global counters across both noise levels.
            for w in winners_n:
                global_wins_noisy[w] += 1
            for w in winners_c:
                global_wins_clean[w] += 1

            # For bolding best medians/means in the table.
            def _best_values(d: dict[str, SummaryStats | None]) -> tuple[float | None, float | None]:
                vals_med = [v.median for v in d.values() if v is not None and np.isfinite(v.median)]
                vals_mean = [v.mean for v in d.values() if v is not None and np.isfinite(v.mean)]
                best_med = min(vals_med) if vals_med else None
                best_mean = min(vals_mean) if vals_mean else None
                return best_med, best_mean

            best_n_med, best_n_mean = _best_values(per_model_noisy)
            best_c_med, best_c_mean = _best_values(per_model_clean)
            eps_best = 1e-12

            lines.append(f"#### {name} ({dim_caption}) — {metric_name}")
            lines.append("")
            lines.append("| Model | noisy median | noisy mean | noisy std | noisy min | noisy max | noisy n | clean median | clean mean | clean std | clean min | clean max | clean n | mean iters | median iters |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for label in model_labels:
                s_n = per_model_noisy[label]
                s_c = per_model_clean[label]
                it_s = per_model_iters[label]
                star_n = " **(best)**" if label in winners_n and len(winners_n) == 1 else (" **(tied best)**" if label in winners_n else "")
                star_c = " **(best)**" if label in winners_c and len(winners_c) == 1 else (" **(tied best)**" if label in winners_c else "")
                # Put both best markers (noisy/clean) into the model cell to keep the table narrow.
                model_cell = label
                if star_n or star_c:
                    # Prefer distinct tags when they differ.
                    tags: list[str] = []
                    if star_n:
                        tags.append("noisy " + star_n.strip("* ").strip())
                    if star_c:
                        tags.append("clean " + star_c.strip("* ").strip())
                    model_cell = f"{label} ({', '.join(tags)})"
                n_med, n_mean, n_std, n_min, n_max, n_n = _stats_cell(s_n)
                c_med, c_mean, c_std, c_min, c_max, c_n = _stats_cell(s_c)

                # Bold best medians and means and add an asterisk marker (inline, no extra spaces).
                if s_n is not None and best_n_med is not None and abs(s_n.median - best_n_med) <= eps_best:
                    n_med = f"**{n_med}***"
                if s_n is not None and best_n_mean is not None and abs(s_n.mean - best_n_mean) <= eps_best:
                    n_mean = f"**{n_mean}***"
                if s_c is not None and best_c_med is not None and abs(s_c.median - best_c_med) <= eps_best:
                    c_med = f"**{c_med}***"
                if s_c is not None and best_c_mean is not None and abs(s_c.mean - best_c_mean) <= eps_best:
                    c_mean = f"**{c_mean}***"
                if it_s is None:
                    it_mean_str = "—"
                    it_med_str = "—"
                else:
                    it_mean_str = _fmt(it_s.mean_iters)
                    it_med_str = _fmt(it_s.median_iters)
                lines.append(
                    f"| {model_cell} | {n_med} | {n_mean} | {n_std} | {n_min} | {n_max} | {n_n} | "
                    f"{c_med} | {c_mean} | {c_std} | {c_min} | {c_max} | {c_n} | {it_mean_str} | {it_med_str} |"
                )
            lines.append("")

        # Noise-level win summary.
        lines.append("#### Median-win count across the 8 problems")
        lines.append("")
        lines.append("| Model | noisy wins | clean wins |")
        lines.append("|---|---:|---:|")
        for label in model_labels:
            lines.append(f"| {label} | {wins_noisy[label]} | {wins_clean[label]} |")
        lines.append("")

        # Per-problem per-run wins and accumulate into global per_run_wins.
        lines.append("#### Per-run wins (noisy final best, run-wise comparisons)")
        lines.append("")
        lines.append("| Problem | " + " | ".join(model_labels) + " |")
        lines.append("|---|" + "|".join([":---:"] * len(model_labels)) + "|")

        for folder, dim_key, subtitle in IDETC_PROBLEMS:
            key = (folder, dim_key, noise_key)
            is_max = (folder, dim_key) in MAXIMIZATION_PROBLEMS

            # Gather aligned runs per model for this problem/noise.
            runs_per_model: list[list[RunTuple]] = []
            for i, _label in enumerate(model_labels):
                runs = all_data_noisy[i].get(key, [])
                runs_per_model.append(runs)

            if not runs_per_model or any(len(r) == 0 for r in runs_per_model):
                # If any model has no runs here, skip this problem for per-run comparison.
                continue

            # Only compare up to the minimum number of runs shared by all models.
            n_paired = min(len(r) for r in runs_per_model)
            per_model_local_wins = {label: 0 for label in model_labels}

            for run_idx in range(n_paired):
                vals: dict[str, float] = {}
                for model_idx, label in enumerate(model_labels):
                    _rid, hist, _t = runs_per_model[model_idx][run_idx]
                    if not hist:
                        continue
                    v = float(hist[-1])
                    if is_max:
                        v = -v  # flip so lower is better
                    vals[label] = v
                if not vals:
                    continue
                best_val = min(vals.values())
                eps = 1e-12
                winners = [lbl for lbl, v in vals.items() if abs(v - best_val) <= eps]
                for w in winners:
                    per_model_local_wins[w] += 1
                    per_run_wins[w] += 1

            problem_name, dim_caption = _problem_display(folder, dim_key, subtitle)
            row_counts = [str(per_model_local_wins[label]) for label in model_labels]
            lines.append(f"| {problem_name} ({dim_caption}) | " + " | ".join(row_counts) + " |")
        lines.append("")

    # Global median-win summary across all problems and noise configs.
    lines.append("### Overall median-win count across all problems and noise levels")
    lines.append("")
    lines.append("| Model | noisy wins | clean wins |")
    lines.append("|---|---:|---:|")
    for label in model_labels:
        lines.append(f"| {label} | {global_wins_noisy[label]} | {global_wins_clean[label]} |")
    lines.append("")

    # Global per-run win summary across all problems and noise configs.
    lines.append("### Overall per-run wins across all problems/noise (noisy final best)")
    lines.append("")
    lines.append("| Model | total run-wise wins |")
    lines.append("|---|---:|")
    for label in model_labels:
        lines.append(f"| {label} | {per_run_wins[label]} |")
    lines.append("")

    # High-level narrative (very compact; the table is the source of truth).
    lines.append("### Quick takeaways")
    lines.append("")
    lines.append("- **Use the win tables as a first-pass summary** (median of final best across runs, per problem).")
    lines.append("- **If clean/noisy winners disagree**, it’s a hint the method is exploiting observation noise (or the clean signal is behaving differently).")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    md = build_markdown()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[Wrote markdown] {OUT_MD}")


if __name__ == "__main__":
    main()

