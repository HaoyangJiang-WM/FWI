"""Truth-free D4 source-orbit utilities used by the reported protocol."""

from __future__ import annotations

from collections.abc import Callable

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


def rank_source_orbit(
    source: torch.Tensor,
    public_score: Callable[[torch.Tensor], tuple[float, ...]],
) -> tuple[str, list[tuple[str, tuple[float, ...]]]]:
    """Rank all fixed D4 sources using only a caller-supplied public score.

    The stable D4 order is the final tie breaker. Ground-truth metrics must
    never be supplied through ``public_score`` in the reported protocol.
    """

    records = [(name, tuple(public_score(d4_transform(source, name)))) for name in D4_ORDER]
    records.sort(key=lambda item: (item[1], D4_ORDER.index(item[0])))
    return records[0][0], records


__all__ = ["D4_ORDER", "d4_transform", "rank_source_orbit"]
