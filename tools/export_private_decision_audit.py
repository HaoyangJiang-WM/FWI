#!/usr/bin/env python3
"""Export a sanitized, machine-checkable D4 decision ledger.

Run this only against the private experiment tree. The output intentionally
excludes filesystem paths, model arrays, logs, and user/cluster identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        (row, seed): float(summary["matrices"]["final_mse"][i][j])
        for i, row in enumerate(summary["rows"])
        for j, seed in enumerate(summary["seeds"])
    }
    candidates: dict[tuple[int, int], list[dict]] = {key: [] for key in expected}
    for path in args.private_root.glob("artifacts/**/row_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        key = (int(payload.get("row", -1)), int(payload.get("raw_seed", -1)))
        if key not in expected or "source_orbit" not in payload:
            continue
        if payload.get("source_orbit_manifest_sha256") != summary["source_manifest_sha256"]:
            continue
        if not np.isclose(float(payload.get("selected_mse", np.nan)), expected[key], atol=1e-6, rtol=0):
            continue
        source = payload["source_orbit"]
        record = {
            "row": key[0],
            "seed": key[1],
            "winner": source["winner"],
            "ranked_transforms": source["ranked_transforms"],
            "retained_transforms": source["retained_transforms"],
            "records": source["records"],
            "selected_mse_postdecision": float(payload["selected_mse"]),
            "event_log": payload["event_log"],
            "truth_access": payload["truth_access"],
            "source_orbit_manifest_sha256": payload["source_orbit_manifest_sha256"],
            "selection_manifest_sha256": payload["selection_manifest_sha256"],
            "materialization_manifest_sha256": payload["materialization_manifest_sha256"],
            "truth_sha256_postdecision": payload["preparation"]["truth_sha256_postdecision"],
        }
        record["decision_record_sha256"] = canonical_sha256(record)
        candidates[key].append(record)
    chosen = []
    for key in expected:
        unique = {canonical_sha256(record): record for record in candidates[key]}
        if len(unique) != 1:
            raise RuntimeError(f"expected one unique decision record for {key}, found {len(unique)}")
        chosen.append(next(iter(unique.values())))
    release = {
        "schema": "d4_public_h_decision_audit_v1",
        "ranking_rule": "lexicographic(heldout_key, fit_key, frozen_D4_order)",
        "key_components": ["broadband_mean", "broadband_max", "dt2", "dt1", "phase_proxy"],
        "truth_role": "postdecision_report_only_never_selector",
        "manifest": manifest,
        "records": chosen,
    }
    release["release_sha256"] = canonical_sha256(release)
    args.output.write_text(json.dumps(release, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
