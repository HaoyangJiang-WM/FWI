from __future__ import annotations

import torch

from flowmap_multiback.public_h_metrics import public_h_key


def test_identical_public_observation_has_zero_key() -> None:
    value = torch.randn(1, 2, 8, 5, dtype=torch.double)
    key = public_h_key(value, value)
    assert all(abs(component) < 1e-12 for component in key)


def test_public_key_has_five_finite_components() -> None:
    prediction = torch.randn(1, 2, 8, 5)
    observation = torch.randn_like(prediction)
    assert len(public_h_key(prediction, observation)) == 5
