"""Exact deterministic public-H ranking rule used by the D4 screen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, init=False)
class PublicHRecord:
    transform: str
    fit_key: tuple[float, ...]
    selection_key: tuple[float, ...]
    q_feasible: bool = True

    def __init__(
        self,
        transform: str,
        fit_key: tuple[float, ...],
        selection_key: tuple[float, ...] | None = None,
        q_feasible: bool = True,
        *,
        heldout_key: tuple[float, ...] | None = None,
    ) -> None:
        """Create a ranking record.

        ``heldout_key`` remains accepted as a compatibility alias for archived
        ledgers. The split is used for source selection and is not a test set.
        """

        if selection_key is None:
            selection_key = heldout_key
        elif heldout_key is not None and tuple(selection_key) != tuple(heldout_key):
            raise ValueError("selection_key and heldout_key disagree")
        if selection_key is None:
            raise TypeError("selection_key is required")

        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "fit_key", tuple(fit_key))
        object.__setattr__(self, "selection_key", tuple(selection_key))
        object.__setattr__(self, "q_feasible", bool(q_feasible))

    @property
    def heldout_key(self) -> tuple[float, ...]:
        """Compatibility alias for the archived JSON field name."""

        return self.selection_key


def rank_public_h(
    declared_transforms: Iterable[str],
    records: Iterable[PublicHRecord],
) -> tuple[PublicHRecord, ...]:
    """Rank feasible records by selection key, fit key, then declared order."""

    declared = tuple(declared_transforms)
    supplied = tuple(records)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("declared transforms must be unique and nonempty")
    if tuple(record.transform for record in supplied) != declared:
        raise ValueError("records must follow the frozen declaration order")
    if any(
        not record.fit_key
        or not record.selection_key
        or not all(
            math.isfinite(value)
            for value in (*record.fit_key, *record.selection_key)
        )
        for record in supplied
    ):
        raise ValueError("public-H keys must be finite and nonempty")
    order = {name: index for index, name in enumerate(declared)}
    ranked = tuple(
        sorted(
            (record for record in supplied if record.q_feasible),
            key=lambda record: (
                record.selection_key,
                record.fit_key,
                order[record.transform],
            ),
        )
    )
    if not ranked:
        raise ValueError("no feasible source-orbit record")
    return ranked


__all__ = ["PublicHRecord", "rank_public_h"]
