"""Range surfaces and normalized delay kernels."""

import math
import numpy as np
from scipy.special import stdtr, gammaln


def _pack_sets(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    maximum = max((len(row) for row in sets), default=0)
    values = np.full((len(sets), maximum), np.nan, dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for index, row in enumerate(sets):
        current = np.sort(np.asarray(row, dtype=np.float64), kind="stable")
        current = current[np.isfinite(current)]
        values[index, : len(current)] = current
        mask[index, : len(current)] = True
    return values, mask


def range_surfaces(xy: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    return np.sqrt(
        np.sum(np.square(xy[:, None, :] - anchors[None, :, :2]), axis=2)
        + np.square(anchors[None, :, 2])
    )


def _student_density(
    observed: np.ndarray,
    location: np.ndarray,
    scale_m: float,
    range_limit_m: float,
) -> np.ndarray:
    scale = max(float(scale_m), 0.01)
    upper = (float(range_limit_m) - location) / scale
    lower = -location / scale
    normalizer = np.maximum(
        stdtr(3, upper) - stdtr(3, lower),
        1.0e-15,
    )
    constant = math.exp(
        float(gammaln(2.0) - gammaln(1.5)) - 0.5 * math.log(3.0 * math.pi) - math.log(scale)
    )
    residual = (observed - location) / scale
    return constant * np.power(1.0 + np.square(residual) / 3.0, -2.0) / normalizer
