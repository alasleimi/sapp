"""Weighted geometric median, using the paper stopping criteria."""

import numpy as np
from numba import njit


@njit(cache=True)
def geometric_median(points, weights, initial):
    estimate = initial.copy()
    for _ in range(40):
        numerator = np.zeros(2)
        denominator = 0.0
        nearest = np.inf
        nearest_index = 0
        for i in range(len(points)):
            dx, dy = points[i, 0] - estimate[0], points[i, 1] - estimate[1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance < nearest:
                nearest = distance
                nearest_index = i
            if distance > 0:
                w = weights[i] / distance
                numerator[0] += w * points[i, 0]
                numerator[1] += w * points[i, 1]
                denominator += w
        if nearest < 1e-10:
            return points[nearest_index].copy()
        updated = numerator / denominator
        movement = np.sqrt(np.sum((updated - estimate) ** 2))
        estimate = updated
        if movement < 1e-7:
            break
    return estimate
