"""Spatial interpolation of source counts and residual delay density."""

import math
import numpy as np
from scipy.spatial import cKDTree
from .types import AnchorMap
from .kernels import _student_density


def survey_kernel_bandwidth(model: AnchorMap) -> float:
    """Return a survey-resolution bandwidth without a room-scale parameter.

    A nonpositive configured value activates the adaptive rule.  With the
    Gaussian written as ``exp(-d^2/(2 h^2))``, ``h=s/sqrt(2)`` gives weight
    ``exp(-1)`` at the median nearest-neighbour survey spacing ``s``.
    """

    if model.parameters.field_length_m > 0.0:
        return float(model.parameters.field_length_m)
    if len(model.reference_xy) < 2:
        return 1.0 / math.sqrt(2.0)
    tree = cKDTree(model.reference_xy)
    distance, _ = tree.query(model.reference_xy, k=2)
    spacing = float(np.median(distance[:, 1]))
    return max(spacing / math.sqrt(2.0), 1.0e-3)


def predict_intensity(model: AnchorMap, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    difference = candidates[:, None, :] - model.reference_xy[None, :, :]
    bandwidth = survey_kernel_bandwidth(model)
    weights = np.exp(-np.sum(np.square(difference), axis=2) / (2.0 * bandwidth**2))
    prior_weight = 0.5
    numerator = weights @ model.soft_counts + prior_weight * model.global_counts
    denominator = np.sum(weights, axis=1, keepdims=True) + prior_weight
    return np.clip(
        numerator / denominator,
        model.parameters.intensity_floor,
        model.parameters.intensity_ceiling,
    )


def _prepare_background(
    model: AnchorMap,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if (
        model.background_delay is None
        or model.background_mask is None
        or model.background_responsibility is None
    ):
        return None
    difference = candidates[:, None, :] - model.reference_xy[None, :, :]
    bandwidth = survey_kernel_bandwidth(model)
    spatial = np.exp(-np.sum(np.square(difference), axis=2) / (2.0 * bandwidth**2))
    prior_weight = 0.5
    denominator = np.sum(spatial, axis=1) + prior_weight
    row_count = np.sum(model.background_responsibility, axis=1)
    global_count = float(np.mean(row_count))
    integral = (spatial @ row_count + prior_weight * global_count) / denominator
    return spatial, denominator, integral


def _background_density(model, prepared, observed):
    if prepared is None:
        return None, None
    spatial, denominator, integral = prepared
    values = np.asarray(observed, dtype=np.float64)
    density = _student_density(
        values[None, None, :],
        model.background_delay[:, :, None],
        model.parameters.association_scale_m,
        model.parameters.range_limit_m,
    )
    density = np.where(model.background_mask[:, :, None], density, 0.0)
    row_density = np.sum(model.background_responsibility[:, :, None] * density, axis=1)
    global_density = np.mean(row_density, axis=0)
    background = (spatial @ row_density + 0.5 * global_density[None, :]) / denominator[:, None]
    return background, integral
