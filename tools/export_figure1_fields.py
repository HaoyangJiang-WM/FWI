#!/usr/bin/env python3
"""Export the exact fields used by the public row-oracle comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROWS = (29864, 29748, 29544, 29952, 29694)
CHOSEN_SEEDS = (20268332, 20268332, 20268432, 20268432, 20268232)
OURS_RELATIVE_PATHS = {
    29864: "artifacts/SOURCE_ORBIT_PROFILED_GATE_0731s/SOURCE_ORBIT_PROFILED_SEED20268332_ROW29864_PANEL1_0731s/row_29864.npz",
    29748: "artifacts/SOURCE_ORBIT_PROFILED_FINAL_ROW29748_SEED20268332_0731u/row_29748.npz",
    29544: "artifacts/SOURCE_ORBIT_PROFILED_FINAL_ROW29544_SEED20268432_0731u/row_29544.npz",
    29952: "artifacts/SOURCE_ORBIT_PROFILED_FINAL_ROW29952_SEED20268432_0731u/row_29952.npz",
    29694: "artifacts/SOURCE_ORBIT_PROFILED_FINAL_ROW29694_SEED20268232_0731u/row_29694.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = (
        args.research_root
        / "artifacts/fwi_ml_baselines_complete_unseen5_0729zz36/predictions.npz"
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline_rows = tuple(int(value) for value in archive["rows"])
        if baseline_rows != ROWS:
            raise ValueError(f"unexpected baseline row order: {baseline_rows}")
        exported = {
            key: np.asarray(archive[key], dtype=np.float32)
            for key in ("truth", "unet", "fno", "pinn_unet", "velocitygan", "inversionnet")
        }

    ours = []
    ours_truth = []
    for row in ROWS:
        with np.load(args.research_root / OURS_RELATIVE_PATHS[row], allow_pickle=False) as archive:
            ours.append(np.asarray(archive["selected"], dtype=np.float32))
            ours_truth.append(np.asarray(archive["truth"], dtype=np.float32))
    exported["ours"] = np.stack(ours)
    ours_truth_array = np.stack(ours_truth)
    if not np.allclose(ours_truth_array, exported["truth"], atol=1e-6):
        raise ValueError("ours and baseline truth tensors differ")

    summary = json.loads(args.final_summary.read_text(encoding="utf-8"))
    expected = np.min(np.asarray(summary["matrices"]["final_mse"], dtype=float), axis=1)
    observed = np.mean((exported["ours"] - exported["truth"]) ** 2, axis=(1, 2))
    if not np.allclose(observed, expected, atol=1e-6):
        raise ValueError(f"row-oracle field MSE mismatch: {observed} versus {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rows=np.asarray(ROWS, dtype=np.int64),
        chosen_seeds=np.asarray(CHOSEN_SEEDS, dtype=np.int64),
        **exported,
    )


if __name__ == "__main__":
    main()
