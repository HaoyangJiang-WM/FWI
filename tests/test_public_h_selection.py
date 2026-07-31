from __future__ import annotations

from flowmap_multiback.public_h_selection import PublicHRecord, rank_public_h


def test_heldout_then_fit_then_declaration_order() -> None:
    declared = ("identity", "flip_x", "flip_y")
    records = (
        PublicHRecord("identity", (2.0,), (1.0,)),
        PublicHRecord("flip_x", (1.0,), (1.0,)),
        PublicHRecord("flip_y", (1.0,), (0.5,)),
    )
    assert [r.transform for r in rank_public_h(declared, records)] == [
        "flip_y",
        "flip_x",
        "identity",
    ]


def test_infeasible_record_is_excluded() -> None:
    declared = ("identity", "flip_x")
    records = (
        PublicHRecord("identity", (1.0,), (1.0,)),
        PublicHRecord("flip_x", (0.0,), (0.0,), q_feasible=False),
    )
    assert rank_public_h(declared, records)[0].transform == "identity"
