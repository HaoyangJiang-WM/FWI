"""Truth-free D4 source-orbit utilities used by the reported protocol."""

from __future__ import annotations

import torch

D4_ORDER = (
    "identity",
    "flip_x",
    "flip_y",
    "rot180",
    "transpose",
    "transpose_flip_x",
    "transpose_flip_y",
    "transpose_rot180",
)


def d4_transform(source: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one norm-preserving square-grid D4 action."""

    if source.ndim < 2 or source.shape[-2] != source.shape[-1]:
        raise ValueError("D4 source must have square trailing dimensions")
    actions = {
        "identity": lambda x: x,
        "flip_x": lambda x: torch.flip(x, (-1,)),
        "flip_y": lambda x: torch.flip(x, (-2,)),
        "rot180": lambda x: torch.flip(x, (-2, -1)),
        "transpose": lambda x: x.transpose(-2, -1),
        "transpose_flip_x": lambda x: torch.flip(x.transpose(-2, -1), (-1,)),
        "transpose_flip_y": lambda x: torch.flip(x.transpose(-2, -1), (-2,)),
        "transpose_rot180": lambda x: torch.flip(x.transpose(-2, -1), (-2, -1)),
    }
    try:
        return actions[name](source).contiguous()
    except KeyError as exc:
        raise ValueError(f"unknown D4 action: {name}") from exc


__all__ = ["D4_ORDER", "d4_transform"]
