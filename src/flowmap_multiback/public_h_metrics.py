"""Public observation key used for truth-free source-orbit ranking."""

from __future__ import annotations

import torch


def public_h_key(prediction: torch.Tensor, observation: torch.Tensor) -> tuple[float, ...]:
    """Return (broadband mean/max, dt2, dt1, phase proxy), lexicographically minimized.

    Inputs have shape ``[batch, acquisition, time, receiver]`` and must
    already be restricted to the declared fit or heldout acquisition split.
    """

    if prediction.shape != observation.shape or prediction.ndim != 4:
        raise ValueError("public-H tensors must share [batch, acquisition, time, receiver] shape")
    residual = prediction - observation
    broadband = residual.square().mean(dim=(2, 3)).sqrt()
    dt1 = residual.diff(dim=2).square().mean().sqrt()
    dt2 = residual.diff(n=2, dim=2).square().mean().sqrt()
    pred_dt = prediction.diff(dim=2).flatten(2)
    obs_dt = observation.diff(dim=2).flatten(2)
    denominator = (
        pred_dt.square().sum(2).sqrt() * obs_dt.square().sum(2).sqrt()
    ).clamp_min(1.0e-12)
    phase = (1.0 - (pred_dt * obs_dt).sum(2) / denominator).mean()
    values = (broadband.mean(), broadband.max(), dt2, dt1, phase)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("public-H key became nonfinite")
    return tuple(float(value.detach().cpu()) for value in values)


__all__ = ["public_h_key"]
