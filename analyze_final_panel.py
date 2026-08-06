#!/usr/bin/env python3
"""Validate and print the published balanced 5 x 5 aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flowmap_multiback.public_h_selection import PublicHRecord, rank_public_h
from flowmap_multiback.source_orbit import D4_ORDER


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matrix(summary: dict, group: str, name: str) -> np.ndarray:
    value = np.asarray(summary[group][name], dtype=float)
    if value.shape != (5, 5):
        raise ValueError(f"{group}.{name} must have shape 5 x 5")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{group}.{name} contains nonfinite values")
    return value


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected):
        raise ValueError(f"{name} mismatch: {actual} versus {expected}")


def validate(summary: dict) -> None:
    rows, seeds = summary["rows"], summary["seeds"]
    if summary.get("shape") != [5, 5] or len(rows) != 5 or len(seeds) != 5:
        raise ValueError("expected a balanced 5 x 5 panel")
    if len(set(rows)) != 5 or len(set(seeds)) != 5:
        raise ValueError("rows and seeds must be unique")
    if summary.get("truth_role") != "postdecision_only":
        raise ValueError("unexpected truth role")

    mse = _matrix(summary, "matrices", "final_mse")
    f1 = _matrix(summary, "matrices", "final_f1")
    right = _matrix(summary, "matrices", "final_right_third_mse")
    runtime = _matrix(summary, "matrices", "runtime_seconds")
    source = _matrix(summary, "source", "mse_matrix")

    _assert_close(float(mse.mean()), summary["final"]["mean_mse"], "mean MSE")
    _assert_close(float(mse.max()), summary["final"]["max_mse"], "maximum MSE")
    _assert_close(
        float(f1.mean()),
        summary["final"]["mean_boundary_f1_tol2"],
        "mean boundary F1",
    )
    _assert_close(
        float(f1.min()),
        summary["final"]["min_boundary_f1_tol2"],
        "minimum boundary F1",
    )
    _assert_close(
        float(right.mean()),
        summary["final"]["mean_right_third_mse"],
        "mean right-third MSE",
    )
    _assert_close(
        float(right.max()),
        summary["final"]["max_right_third_mse"],
        "maximum right-third MSE",
    )
    _assert_close(float(source.mean()), summary["source"]["mean_mse"], "source mean MSE")
    _assert_close(float(source.max()), summary["source"]["max_mse"], "source max MSE")

    delta = mse - source
    attribution = summary["source_to_final_attribution"]
    if int(np.count_nonzero(delta < 0.0)) != attribution["improved_mse_count"]:
        raise ValueError("improved MSE count mismatch")
    if int(np.count_nonzero(delta > 0.0)) != attribution["regressed_mse_count"]:
        raise ValueError("regressed MSE count mismatch")
    _assert_close(float(delta.mean()), attribution["mean_mse_delta"], "mean MSE delta")
    if attribution.get("causal_interpretation") is not False:
        raise ValueError("source-to-final attribution must remain noncausal")
    if not 0 <= attribution["improved_f1_count"] <= 25:
        raise ValueError("invalid improved F1 count")

    speed = summary["speed"]
    _assert_close(float(runtime.mean()), speed["mean_seconds"], "mean runtime")
    _assert_close(float(np.median(runtime)), speed["median_seconds"], "median runtime")
    _assert_close(float(np.percentile(runtime, 95)), speed["p95_seconds"], "p95 runtime")
    _assert_close(float(runtime.max()), speed["max_seconds"], "maximum runtime")
    _assert_close(
        float(3600.0 / runtime.mean()),
        speed["mean_single_gpu_samples_per_hour"],
        "single-GPU throughput",
    )


def validate_decisions(audit: dict, summary: dict) -> None:
    expected = {(row, seed) for row in summary["rows"] for seed in summary["seeds"]}
    observed = {(record["row"], record["seed"]) for record in audit["records"]}
    if observed != expected or len(audit["records"]) != 25:
        raise ValueError("D4 decision ledger does not match the 5 x 5 panel")
    for record in audit["records"]:
        if record["source_orbit_manifest_sha256"] != summary["source_manifest_sha256"]:
            raise ValueError("source manifest mismatch")
        supplied = tuple(
            PublicHRecord(
                transform=name,
                fit_key=tuple(record["records"][name]["fit_key"]),
                heldout_key=tuple(record["records"][name]["heldout_key"]),
                q_feasible=bool(record["records"][name]["q_feasible"]),
            )
            for name in D4_ORDER
        )
        ranked = rank_public_h(D4_ORDER, supplied)
        names = [item.transform for item in ranked]
        if names != record["ranked_transforms"] or names[0] != record["winner"]:
            raise ValueError(f"public-H ranking mismatch for {(record['row'], record['seed'])}")
        stored_hash = record["decision_record_sha256"]
        unhashed = {
            key: value
            for key, value in record.items()
            if key != "decision_record_sha256"
        }
        if canonical_sha256(unhashed) != stored_hash:
            raise ValueError("decision record hash mismatch")
    release_hash = audit["release_sha256"]
    unhashed_release = {
        key: value for key, value in audit.items() if key != "release_sha256"
    }
    if canonical_sha256(unhashed_release) != release_hash:
        raise ValueError("decision release hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("results/final_25_summary.json"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("results/d4_public_h_decision_audit.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate(summary)
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    validate_decisions(decisions, summary)
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "seeds": summary["seeds"],
                "source": {
                    "mean_mse": summary["source"]["mean_mse"],
                    "max_mse": summary["source"]["max_mse"],
                },
                "final": summary["final"],
                "source_to_final_attribution": summary[
                    "source_to_final_attribution"
                ],
                "speed": summary["speed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
