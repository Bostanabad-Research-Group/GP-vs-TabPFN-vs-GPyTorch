"""
Create a scatter plot comparing Gaussian, Uniform, and Student-t noise samples.

The uniform and Student-t branches match the same target standard deviation:
    uniform_noise = (rand - 0.5) * 2 * noise_scale * sqrt(3)
    student_t_noise = t(df) * noise_scale / sqrt(df / (df - 2))
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def make_noise_scatter_plot(
    n_samples: int,
    noise_scale: float,
    student_t_df: float,
    seed: int,
    output_path: Path,
) -> Path:
    """Generate and save a side-by-side scatter figure."""
    if student_t_df <= 2.0:
        raise ValueError("student_t_df must be > 2.0 for finite variance.")

    torch.manual_seed(seed)
    x_idx = torch.arange(n_samples, dtype=torch.float32)

    gaussian_noise = torch.randn(n_samples) * noise_scale
    uniform_noise = (torch.rand(n_samples) - 0.5) * 2.0 * noise_scale * sqrt(3.0)
    student_t_raw = torch.distributions.StudentT(df=student_t_df).sample((n_samples,))
    student_t_noise = student_t_raw * (noise_scale / sqrt(student_t_df / (student_t_df - 2.0)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140, sharey=True)

    axes[0].scatter(x_idx.numpy(), gaussian_noise.numpy(), s=16, alpha=0.7, c="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_title(
        f"Gaussian noise\nmean={gaussian_noise.mean().item():.4f}, std={gaussian_noise.std(unbiased=False).item():.4f}"
    )
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Noise value")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(x_idx.numpy(), uniform_noise.numpy(), s=16, alpha=0.7, c="#ff7f0e")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_title(
        f"Uniform noise\nmean={uniform_noise.mean().item():.4f}, std={uniform_noise.std(unbiased=False).item():.4f}"
    )
    axes[1].set_xlabel("Sample index")
    axes[1].grid(True, alpha=0.25)

    axes[2].scatter(x_idx.numpy(), student_t_noise.numpy(), s=16, alpha=0.7, c="#2ca02c")
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[2].set_title(
        f"Student-t noise (df={student_t_df:g})\nmean={student_t_noise.mean().item():.4f}, std={student_t_noise.std(unbiased=False).item():.4f}"
    )
    axes[2].set_xlabel("Sample index")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(
        f"Gaussian vs Uniform vs Student-t Noise Scatter (target std = {noise_scale:.4f}, n = {n_samples})",
        fontsize=12,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a scatter plot comparing Gaussian, Uniform, and Student-t noise distributions."
    )
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of noise samples to draw.")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Target standard deviation of noise.")
    parser.add_argument("--student_t_df", type=float, default=4.0, help="Degrees of freedom for Student-t noise (>2).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments_revisions_april/plots_noise/noise_scatter_gaussian_uniform_student_t.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_noise_scatter_plot(
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        student_t_df=args.student_t_df,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
