from __future__ import annotations

import torch

from flowmap_multiback.source_orbit import D4_ORDER, d4_transform


def test_all_d4_actions_preserve_shape_and_norm() -> None:
    raw = torch.arange(64, dtype=torch.double).reshape(1, 1, 8, 8)
    for name in D4_ORDER:
        moved = d4_transform(raw, name)
        assert moved.shape == raw.shape
        torch.testing.assert_close(moved.square().sum(), raw.square().sum())
