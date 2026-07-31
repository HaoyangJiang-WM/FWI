"""Exact deterministic public-H ranking rule used by the D4 screen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PublicHRecord:
    transform: str
    fit_key: tuple[float, ...]
    heldout_key: tuple[float, ...]
    q_feasible: bool = True


def rank_public_h(
    declared_transforms: Iterable[str],
    records: Iterable[PublicHRecord],
) -> tuple[PublicHRecord, ...]:
    """Rank feasible records by heldout key, fit key, then declared order."""

    declared = tuple(declared_transforms)
    supplied = tuple(records)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("declared transforms must be unique and nonempty")
    if tuple(record.transform for record in supplied) != declared:
        raise ValueError("records must follow the frozen declaration order")
    if any(
        not record.fit_key
        or not record.heldout_key
        or not all(math.isfinite(value) for value in (*record.fit_key, *record.heldout_key))
        for record in supplied
    ):
        raise ValueError("public-H keys must be finite and nonempty")
    order = {name: index for index, name in enumerate(declared)}
    ranked = tuple(
        sorted(
            (record for record in supplied if record.q_feasible),
            key=lambda record: (
                record.heldout_key,
                record.fit_key,
                order[record.transform],
            ),
        )
    )
    if not ranked:
        raise ValueError("no feasible source-orbit record")
    return ranked


__all__ = ["PublicHRecord", "rank_public_h"]
