from __future__ import annotations

import torch

from flowmap_multiback.protocol import BANDS, boundary_f1


def test_minimal_protocol_has_fixed_bands_and_identity_f1() -> None:
    assert BANDS == (0.125, 0.25, 0.5, 1.0)
    field = torch.zeros(1, 1, 8, 8)
    field[..., 3:, :] = 1.0
    assert boundary_f1(field, field) == 1.0
