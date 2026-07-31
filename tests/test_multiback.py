from __future__ import annotations

import torch

from flowmap_multiback.multiback import anchored_back


def test_zero_control_is_identity_despite_cycle_defect() -> None:
    state = torch.tensor([1.0, -2.0], dtype=torch.double)
    forward = lambda x: 2.0 * x + 0.3
    return_with_defect = lambda x: 0.4 * x - 0.7
    result = anchored_back(
        state,
        torch.zeros_like(state),
        forward_leg=forward,
        return_leg=return_with_defect,
    )
    torch.testing.assert_close(result, state, rtol=0, atol=0)


def test_control_gradient_remains_live() -> None:
    state = torch.randn(4, dtype=torch.double)
    control = torch.zeros_like(state, requires_grad=True)
    output = anchored_back(
        state,
        control,
        forward_leg=lambda x: 1.7 * x,
        return_leg=lambda x: torch.tanh(x),
    )
    output.sum().backward()
    assert control.grad is not None
    assert bool(torch.isfinite(control.grad).all())
