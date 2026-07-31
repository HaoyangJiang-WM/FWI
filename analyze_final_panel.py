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


def validate_decisions(audit: dict, summary: dict) -> None:
    expected = {(row, seed) for row in summary["rows"] for seed in summary["seeds"]}
    observed = {(record["row"], record["seed"]) for record in audit["records"]}
    if observed != expected or len(audit["records"]) != 25:
        raise ValueError("D4 decision ledger does not match the 5 x 5 panel")
    for record in audit["records"]:
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
        unhashed = {key: value for key, value in record.items() if key != "decision_record_sha256"}
        if canonical_sha256(unhashed) != stored_hash:
            raise ValueError("decision record hash mismatch")
    release_hash = audit["release_sha256"]
    unhashed_release = {key: value for key, value in audit.items() if key != "release_sha256"}
    if canonical_sha256(unhashed_release) != release_hash:
        raise ValueError("decision release hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, nargs="?", default=Path("results/final_25_summary.json"))
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
    print(json.dumps({"rows": summary["rows"], "seeds": summary["seeds"], "final": summary["final"], "goal": summary["goal"]}, indent=2))


if __name__ == "__main__":
    main()
