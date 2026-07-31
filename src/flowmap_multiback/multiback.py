"""Generic identity-anchored recourse for a differentiable frozen flow."""

from __future__ import annotations

from collections.abc import Callable

import torch

FlowLeg = Callable[[torch.Tensor], torch.Tensor]


def anchored_back(
    state: torch.Tensor,
    control: torch.Tensor,
    *,
    forward_leg: FlowLeg,
    return_leg: FlowLeg,
) -> torch.Tensor:
    """Apply controlled-minus-passive recourse with exact zero-control identity.

    The two flow legs may be learned numerical maps and need not be inverses.
    ``control`` is inserted at the high/noisy end of the cycle.
    """

    high = forward_leg(state)
    if high.shape != control.shape:
        raise ValueError("control and high-state shapes must match")
    passive = return_leg(high)
    controlled = return_leg(high + control)
    delta = controlled - passive
    if bool((control.detach() == 0).all()):
        # Value-exact identity at zero while retaining the controlled-cycle
        # Jacobian for an optimizer initialized at zero action.
        return state + delta - delta.detach()
    return state + delta


__all__ = ["anchored_back"]
