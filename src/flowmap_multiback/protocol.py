"""Publication-minimal immutable reporting protocol for canonical shooting."""

from __future__ import annotations

import numpy as np
import torch


BANDS = (0.125, 0.25, 0.5, 1.0)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_coordinates = np.argwhere(source)
    target_coordinates = np.argwhere(target)
    if not len(source_coordinates) or not len(target_coordinates):
        return np.full(max(len(source_coordinates), 1), np.inf)
    squared = np.square(
        source_coordinates[:, None, :]
        - target_coordinates[None, :, :]
    ).sum(axis=2)
    return np.sqrt(squared.min(axis=1))


def boundary_f1(model: torch.Tensor, truth: torch.Tensor) -> float:
    """Tolerance-two boundary F1 used only after the fit decision closes."""

    model_field = np.asarray(model.detach().cpu()).squeeze()
    truth_field = np.asarray(truth.detach().cpu()).squeeze()
    model_dy, model_dx = np.gradient(model_field)
    truth_dy, truth_dx = np.gradient(truth_field)
    model_gradient = np.hypot(model_dx, model_dy)
    truth_gradient = np.hypot(truth_dx, truth_dy)
    model_edge = model_gradient >= np.quantile(model_gradient, 0.90)
    truth_edge = truth_gradient >= np.quantile(truth_gradient, 0.90)
    precision = float(
        np.mean(_nearest_distances(model_edge, truth_edge) <= 2.0)
    )
    recall = float(
        np.mean(_nearest_distances(truth_edge, model_edge) <= 2.0)
    )
    return 2.0 * precision * recall / max(precision + recall, 1.0e-12)


__all__ = ["BANDS", "boundary_f1"]
