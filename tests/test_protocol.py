from __future__ import annotations

import pytest
import torch

from flowmap_multiback.protocol import BANDS, boundary_f1


def test_minimal_protocol_has_fixed_bands_and_identity_f1() -> None:
    assert BANDS == (0.125, 0.25, 0.5, 1.0)
    field = torch.zeros(1, 1, 8, 8)
    field[..., 3:, :] = 1.0
    assert boundary_f1(field, field) == 1.0


def test_two_boundary_free_fields_score_one() -> None:
    model = torch.zeros(1, 1, 8, 8)
    truth = torch.ones_like(model)
    assert boundary_f1(model, truth) == 1.0


def test_boundary_free_vs_nonempty_boundary_scores_zero() -> None:
    model = torch.zeros(1, 1, 8, 8)
    truth = torch.zeros_like(model)
    truth[..., 4:, :] = 1.0
    assert boundary_f1(model, truth) == 0.0


def test_boundary_f1_rejects_multiple_fields() -> None:
    model = torch.zeros(2, 1, 8, 8)
    truth = torch.zeros_like(model)
    with pytest.raises(ValueError, match="exactly one 2D field"):
        boundary_f1(model, truth)
