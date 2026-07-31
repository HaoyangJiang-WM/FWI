#!/usr/bin/env python3
"""Validate and print the published balanced 5 x 5 aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate(summary: dict) -> None:
    rows, seeds = summary["rows"], summary["seeds"]
    if summary.get("shape") != [5, 5] or len(rows) != 5 or len(seeds) != 5:
        raise ValueError("expected a balanced 5 x 5 panel")
    mse = np.asarray(summary["matrices"]["final_mse"], dtype=float)
    f1 = np.asarray(summary["matrices"]["final_f1"], dtype=float)
    if mse.shape != (5, 5) or f1.shape != (5, 5):
        raise ValueError("metric matrix shape mismatch")
    if not np.isclose(mse.mean(), summary["final"]["mean_mse"]):
        raise ValueError("mean MSE mismatch")
    if not np.isclose(mse.max(), summary["final"]["max_mse"]):
        raise ValueError("max MSE mismatch")
    if not np.isclose(f1.mean(), summary["final"]["mean_boundary_f1_tol2"]):
        raise ValueError("mean F1 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, nargs="?", default=Path("results/final_25_summary.json"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate(summary)
    print(json.dumps({"rows": summary["rows"], "seeds": summary["seeds"], "final": summary["final"], "goal": summary["goal"]}, indent=2))


if __name__ == "__main__":
    main()
