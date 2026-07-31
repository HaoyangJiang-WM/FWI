"""Publication-minimal immutable reporting protocol for canonical shooting."""

from __future__ import annotations

import numpy as np
import torch


BANDS = (0.125, 0.25, 0.5, 1.0)
EDGE_FRACTION = 0.10
EDGE_TOLERANCE_PIXELS = 2.0


def _as_2d_field(value: torch.Tensor, *, name: str) -> np.ndarray:
    field = np.asarray(value.detach().cpu(), dtype=float).squeeze()
    if field.ndim != 2:
        raise ValueError(f"{name} must contain exactly one 2D field after squeezing")
    if not np.isfinite(field).all():
        raise FloatingPointError(f"{name} contains nonfinite values")
    return field


def _top_fraction_edges(
    gradient: np.ndarray,
    *,
    fraction: float = EDGE_FRACTION,
) -> np.ndarray:
    """Select an exact, deterministic top fraction of positive gradients."""

    if gradient.ndim != 2:
        raise ValueError("gradient must be two-dimensional")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if not np.isfinite(gradient).all():
        raise FloatingPointError("gradient contains nonfinite values")

    flat = gradient.reshape(-1)
    positive = np.flatnonzero(flat > 0.0)
    edges = np.zeros_like(flat, dtype=bool)
    if not len(positive):
        return edges.reshape(gradient.shape)

    count = min(len(positive), max(1, int(np.ceil(fraction * flat.size))))
    order = np.argsort(-flat[positive], kind="stable")[:count]
    edges[positive[order]] = True
    return edges.reshape(gradient.shape)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_coordinates = np.argwhere(source)
    target_coordinates = np.argwhere(target)
    if not len(source_coordinates):
        return np.empty(0, dtype=float)
    if not len(target_coordinates):
        return np.full(len(source_coordinates), np.inf)
    squared = np.square(
        source_coordinates[:, None, :] - target_coordinates[None, :, :]
    ).sum(axis=2)
    return np.sqrt(squared.min(axis=1))


def boundary_f1(model: torch.Tensor, truth: torch.Tensor) -> float:
    """Return the postdecision tolerance-two boundary F1.

    Boundaries are the exact top 10% of strictly positive gradient-magnitude
    pixels in each field. Two boundary-free fields score 1; a boundary-free
    field compared with a nonempty boundary set scores 0.
    """

    model_field = _as_2d_field(model, name="model")
    truth_field = _as_2d_field(truth, name="truth")
    model_dy, model_dx = np.gradient(model_field)
    truth_dy, truth_dx = np.gradient(truth_field)
    model_edge = _top_fraction_edges(np.hypot(model_dx, model_dy))
    truth_edge = _top_fraction_edges(np.hypot(truth_dx, truth_dy))

    model_has_edges = bool(model_edge.any())
    truth_has_edges = bool(truth_edge.any())
    if not model_has_edges and not truth_has_edges:
        return 1.0
    if model_has_edges != truth_has_edges:
        return 0.0

    precision = float(
        np.mean(_nearest_distances(model_edge, truth_edge) <= EDGE_TOLERANCE_PIXELS)
    )
    recall = float(
        np.mean(_nearest_distances(truth_edge, model_edge) <= EDGE_TOLERANCE_PIXELS)
    )
    return 2.0 * precision * recall / max(precision + recall, 1.0e-12)


__all__ = [
    "BANDS",
    "EDGE_FRACTION",
    "EDGE_TOLERANCE_PIXELS",
    "boundary_f1",
]
