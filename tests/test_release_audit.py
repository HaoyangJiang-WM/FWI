"""Regression tests for the published aggregate, ledger, and plotting CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_summary_and_decision_ledger() -> None:
    audit_module = _load_module("release_analyzer", ROOT / "analyze_final_panel.py")
    summary = json.loads(
        (ROOT / "results/final_25_summary.json").read_text(encoding="utf-8")
    )
    decisions = json.loads(
        (ROOT / "results/d4_public_h_decision_audit.json").read_text(encoding="utf-8")
    )
    audit_module.validate(summary)
    audit_module.validate_decisions(decisions, summary)


def test_plot_cli_help_has_no_side_effect(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.png"
    result = subprocess.run(
        [sys.executable, str(ROOT / "plot_attribution.py"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--summary" in result.stdout
    assert "--output" in result.stdout
    assert not output.exists()


def test_plot_cli_writes_requested_output(tmp_path: Path) -> None:
    output = tmp_path / "attribution.png"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_attribution.py"),
            "--summary",
            str(ROOT / "results/final_25_summary.json"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
    )
    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_paper_figure_uses_lowest_mse_seed_per_case() -> None:
    figure_module = _load_module("paper_figures", ROOT / "plot_paper_figures.py")
    summary = json.loads(
        (ROOT / "results/final_25_summary.json").read_text(encoding="utf-8")
    )
    mse, f1, seeds = figure_module.row_oracle(summary)
    assert seeds == [20268332, 20268332, 20268432, 20268432, 20268232]
    matrix = np.asarray(summary["matrices"]["final_mse"], dtype=float)
    assert np.allclose(mse, matrix.min(axis=1))
    assert mse.shape == f1.shape == (5,)
