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


def plot_bars(final_path: Path, baseline_path: Path, output_path: Path) -> None:
    final = json.loads(final_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if final["rows"] != baseline["rows"]:
        raise ValueError("baseline and final row order differ")

    ours_mse, ours_f1, chosen_seeds = row_oracle(final)
    methods = {"Ours (best seed per case)": {"mse": ours_mse, "f1": ours_f1}}
    methods.update({name: values for name, values in baseline["methods"].items() if name != "UPFWI-style"})

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


def plot_reconstructions(fields_path: Path, output_path: Path) -> None:
    names = ("truth", "ours", "unet", "fno", "pinn_unet", "velocitygan", "inversionnet")
    titles = ("Truth", "Ours\n(best seed per case)", "UNet", "FNO", "PINN-UNet", "VelocityGAN", "InversionNet")
    with np.load(fields_path, allow_pickle=False) as archive:
        rows = archive["rows"].astype(int).tolist()
        chosen_seeds = archive["chosen_seeds"].astype(int).tolist()
        arrays = {name: np.asarray(archive[name], dtype=float) for name in names}
    if any(array.shape != (5, 70, 70) for array in arrays.values()):
        raise ValueError("all reconstruction arrays must have shape 5 x 70 x 70")

    fig, axes = plt.subplots(5, 7, figsize=(14.2, 10.2), constrained_layout=True)
    for row_index, row in enumerate(rows):
        truth = arrays["truth"][row_index]
        for col_index, (name, title) in enumerate(zip(names, titles)):
            ax = axes[row_index, col_index]
            field = arrays[name][row_index]
            ax.imshow(field, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title, fontsize=11, color="#116B2A" if name == "ours" else "black",
                             fontweight="bold" if name == "ours" else "normal")
            if col_index == 0:
                ax.set_ylabel(f"case {row}", fontsize=10, fontweight="bold")
            if name != "truth":
                mse = float(np.mean((field - truth) ** 2))
                label = f"MSE {mse:.4f}"
                if name == "ours":
                    label += f"  seed {str(chosen_seeds[row_index])[-3:]}"
                ax.text(0.5, -0.035, label, transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.5, color="#116B2A" if name == "ours" else "black",
                        fontweight="bold" if name == "ours" else "normal")
    fig.suptitle(
        "Velocity-model reconstruction: row-oracle ours vs inference baselines\n"
        "Ours selects the lowest-truth-MSE seed separately within each case",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate requested baseline comparison figure.")
    parser.add_argument("--final", type=Path, default=root / "results/final_25_summary.json")
    parser.add_argument("--baselines", type=Path, default=root / "results/baseline_fivecase_metrics.json")
    parser.add_argument("--fields", type=Path, default=root / "results/figure1_fields.npz")
    parser.add_argument("--output", type=Path, default=root / "assets/figures/method_comparison.png")
    parser.add_argument("--bars-output", type=Path, default=root / "assets/figures/method_comparison_bars.png")
    args = parser.parse_args()
    plot_reconstructions(args.fields, args.output)
    plot_bars(args.final, args.baselines, args.bars_output)


if __name__ == "__main__":
    main()
