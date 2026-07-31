#!/usr/bin/env python3
"""Plot paired internal source-stage to final-endpoint MSE changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attribution(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = np.asarray(summary["source"]["mse_matrix"], dtype=float)
    final = np.asarray(summary["matrices"]["final_mse"], dtype=float)
    if source.shape != (5, 5) or final.shape != (5, 5):
        raise ValueError("expected balanced 5 x 5 matrices")

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for index, (a, b) in enumerate(zip(source.ravel(), final.ravel())):
        color = "#2878B5" if b < a else "#C82423"
        axes[0].plot([0, 1], [a, b], color=color, alpha=0.7, linewidth=1.4)
        axes[0].scatter([0, 1], [a, b], color=color, s=12)
    axes[0].set_xticks([0, 1], ["internal source stage", "final reopened endpoint"])
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Paired endpoint attribution (25 cells)")
    axes[0].grid(axis="y", alpha=0.25)

    delta = final - source
    bound = float(np.abs(delta).max())
    image = axes[1].imshow(delta, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
    axes[1].set_xticks(range(5), [str(seed)[-3:] for seed in summary["seeds"]])
    axes[1].set_yticks(range(5), summary["rows"])
    axes[1].set_xlabel("raw seed suffix")
    axes[1].set_ylabel("evaluation case row")
    axes[1].set_title("Final minus source-stage MSE")
    for row in range(5):
        for col in range(5):
            axes[1].text(col, row, f"{delta[row,col]:+.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes[1], shrink=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot paired internal-stage to final-endpoint MSE attribution."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results/final_25_summary.json",
        help="machine-readable 5 x 5 summary JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "assets/figures/source_to_multiback_attribution.png",
        help="output PNG path",
    )
    args = parser.parse_args()
    plot_attribution(args.summary, args.output)


if __name__ == "__main__":
    main()
