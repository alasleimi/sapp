"""Multipath component analysis (MCA) delay-set localization."""

from typing import Sequence
import numpy as np


def localize(
    query: Sequence[float],
    reference_fingerprints: Sequence[Sequence[float]],
    reference_positions_m: np.ndarray,
    *,
    epsilon_m: float,
    k: int = 1,
) -> np.ndarray:
    """Localize one query with MCA scores and weighted reference coordinates."""
    query_array = np.asarray(query, dtype=np.float64)
    scores = []
    for reference in reference_fingerprints:
        reference_array = np.asarray(reference, dtype=np.float64)
        if not len(query_array) or not len(reference_array):
            scores.append(0.0)
            continue
        nearest = np.min(np.abs(query_array[:, None] - reference_array[None, :]), axis=1)
        accepted = nearest < epsilon_m
        scores.append(float(np.sum((epsilon_m - nearest[accepted]) ** 2)))
    scores_array = np.asarray(scores)
    chosen = np.argsort(-scores_array, kind="stable")[: min(k, len(scores_array))]
    if k == 1 or not np.any(scores_array[chosen] > 0):
        return np.asarray(reference_positions_m, dtype=np.float64)[chosen[0]].copy()
    weights = np.maximum(scores_array[chosen], 0.0)
    weights /= weights.sum()
    return np.sum(np.asarray(reference_positions_m)[chosen] * weights[:, None], axis=0)
