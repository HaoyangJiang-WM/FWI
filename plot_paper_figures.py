#!/usr/bin/env python3
"""Generate the two requested paper figures.

Figure 1 compares frozen baselines with an explicitly post-hoc row-oracle
summary of the final five-seed panel. Figure 2 is the frozen 5 x 5 model panel
and is not regenerated here because its underlying field tensors are not part
of the minimal public release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def row_oracle(final: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mse = np.asarray(final["matrices"]["final_mse"], dtype=float)
    f1 = np.asarray(final["matrices"]["final_f1"], dtype=float)
    if mse.shape != (5, 5) or f1.shape != (5, 5):
        raise ValueError("expected final metric matrices with shape 5 x 5")
    indices = np.argmin(mse, axis=1)
    rows = np.arange(mse.shape[0])
    return mse[rows, indices], f1[rows, indices], [final["seeds"][i] for i in indices]


def plot_comparison(final_path: Path, baseline_path: Path, output_path: Path) -> None:
    final = json.loads(final_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if final["rows"] != baseline["rows"]:
        raise ValueError("baseline and final row order differ")

    ours_mse, ours_f1, chosen_seeds = row_oracle(final)
    methods = {"Ours (best seed per case)": {"mse": ours_mse, "f1": ours_f1}}
    methods.update(baseline["methods"])

    names = list(methods)
    means_mse = np.asarray([np.mean(methods[name]["mse"]) for name in names])
    means_f1 = np.asarray([np.mean(methods[name]["f1"]) for name in names])
    colors = ["#1676B7", "#F28E2B", "#D62728", "#8C564B", "#E377C2", "#BCBD22", "#17BECF"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for i, name in enumerate(names):
        marker = "*" if i == 0 else "o"
        size = 180 if i == 0 else 70
        axes[0].scatter(means_mse[i], means_f1[i], marker=marker, s=size,
                        color=colors[i], edgecolor="#333333", linewidth=0.7, label=name, zorder=3)
    axes[0].set_xlabel("Mean MSE across five cases (lower is better)")
    axes[0].set_ylabel("Mean boundary F1, tolerance 2 (higher is better)")
    axes[0].set_title("Aggregate accuracy and interface quality")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    x = np.arange(5)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        axes[1].bar(x + (i - (len(names) - 1) / 2) * width,
                    methods[name]["mse"], width=width, color=colors[i], label=name)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, baseline["rows"])
    axes[1].set_xlabel("Evaluation case")
    axes[1].set_ylabel("MSE (log scale)")
    axes[1].set_title("Per-case MSE")
    axes[1].grid(axis="y", alpha=0.25)

    seed_suffixes = ", ".join(str(seed)[-3:] for seed in chosen_seeds)
    fig.suptitle(
        "Frozen baselines vs post-hoc row-oracle ours\n"
        f"Ours selects the lowest-MSE seed per case (seed suffixes: {seed_suffixes})",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate requested baseline comparison figure.")
    parser.add_argument("--final", type=Path, default=root / "results/final_25_summary.json")
    parser.add_argument("--baselines", type=Path, default=root / "results/baseline_fivecase_metrics.json")
    parser.add_argument("--output", type=Path, default=root / "assets/figures/method_comparison.png")
    args = parser.parse_args()
    plot_comparison(args.final, args.baselines, args.output)


if __name__ == "__main__":
    main()
