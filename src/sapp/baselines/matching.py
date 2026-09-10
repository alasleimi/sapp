"""MCA and MPUrge-MAP reference implementations used in the study."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Match:
    """One bijective MPC match, retaining original vector indices."""

    index_a: int
    index_b: int
    value_a: float
    value_b: float
    dissimilarity: float
    iteration: int


def _ordered(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("MPC vectors must contain only finite values")
    order = np.argsort(array, kind="stable")
    return (array[order], order.astype(np.int64))


def _best_ordered_subset(longer: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact exhaustive-subset optimum using monotone dynamic programming.

    Size unification enumerates all subsets of the
    longer sorted vector and selecting the one closest to the shorter vector.
    Dynamic programming returns exactly that optimum without an exponential
    materialization of all subsets.
    """
    n, m = (len(longer), len(target))
    if m > n:
        raise ValueError("target cannot be longer than the candidate vector")
    if m == 0:
        return np.empty(0, dtype=np.int64)
    cost = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    cost[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            skip = cost[i, j - 1]
            choose = cost[i - 1, j - 1] + abs(longer[j - 1] - target[i - 1])
            if choose <= skip:
                cost[i, j] = choose
                take[i, j] = True
            else:
                cost[i, j] = skip
    selected: list[int] = []
    i, j = (m, n)
    while i:
        if j == 0:
            raise RuntimeError("Failed to reconstruct size-unification subset")
        if take[i, j]:
            selected.append(j - 1)
            i -= 1
        j -= 1
    return np.asarray(selected[::-1], dtype=np.int64)


def size_unify(
    values_a: Sequence[float], values_b: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the minimum-distance subset and retain its original indices."""
    a, ia = _ordered(values_a)
    b, ib = _ordered(values_b)
    if len(a) > len(b):
        keep = _best_ordered_subset(a, b)
        a, ia = (a[keep], ia[keep])
    elif len(b) > len(a):
        keep = _best_ordered_subset(b, a)
        b, ib = (b[keep], ib[keep])
    return (a, b, ia, ib)


def _padded_windows(values: np.ndarray, p: int) -> np.ndarray:
    """Zero-centred P=2p+1 windows with negligible border increments."""
    if p < 0:
        raise ValueError("p must be non-negative")
    if not len(values):
        return np.empty((0, 2 * p + 1), dtype=np.float64)
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0)
    tiny = np.finfo(np.float64).eps * scale * 16.0
    lower = values[0] - tiny * np.arange(p, 0, -1, dtype=np.float64)
    upper = values[-1] + tiny * np.arange(1, p + 1, dtype=np.float64)
    padded = np.concatenate((lower, values, upper))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * p + 1).copy()
    return windows - values[:, None]


def dissimilarity_matrix(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    p: int,
    alpha: float,
) -> np.ndarray:
    """MPUrge-MAP distance and normalized local-pattern dissimilarity."""
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not len(a) or not len(b):
        return np.empty((len(a), len(b)), dtype=np.float64)
    wa = _padded_windows(a, p)
    wb = _padded_windows(b, p)
    upper = np.triu_indices(2 * p + 1, k=1)
    pairwise_a = np.abs(wa[:, :, None] - wa[:, None, :])[:, upper[0], upper[1]]
    pairwise_b = np.abs(wb[:, :, None] - wb[:, None, :])[:, upper[0], upper[1]]
    pattern = np.sum(np.abs(pairwise_a[:, None, :] - pairwise_b[None, :, :]), axis=2)
    accumulation = sum(range(1, 2 * p + 1))
    if accumulation:
        pattern = pattern / float(accumulation)
    distance = np.abs(a[:, None] - b[None, :])
    return alpha * distance + (1.0 - alpha) * pattern


def published_mca_localize(
    query: Sequence[float],
    reference_fingerprints: Sequence[Sequence[float]],
    reference_positions_m: np.ndarray,
    *,
    epsilon_m: float,
    k: int = 1,
) -> np.ndarray:
    """Published MCA score followed by the paper's k-best weighted head."""
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


def centroid(reference_xy: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(reference_xy, dtype=np.float64), axis=0)


def pairwise_match(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    half_window_p: int,
    alpha: float,
) -> list[Match]:
    """Resolve crossing correspondences in increasing dissimilarity order."""
    a, b, ia, ib = size_unify(values_a, values_b)
    accepted: list[Match] = []
    iteration = 0

    def crosses(left: Match, right: Match) -> bool:
        return (left.value_a - right.value_a) * (left.value_b - right.value_b) < 0.0

    while len(a) and len(b):
        delta = dissimilarity_matrix(a, b, p=half_window_p, alpha=alpha)
        row_best = np.argmin(delta, axis=1)
        col_best = np.argmin(delta, axis=0)
        potential_indices = [
            (i, int(j)) for i, j in enumerate(row_best) if int(col_best[int(j)]) == i
        ]
        potential = [
            Match(int(ia[i]), int(ib[j]), float(a[i]), float(b[j]), float(delta[i, j]), iteration)
            for i, j in potential_indices
        ]
        remove_a = {i for i, _ in potential_indices}
        remove_b = {j for _, j in potential_indices}
        keep_a = np.asarray([i not in remove_a for i in range(len(a))])
        keep_b = np.asarray([j not in remove_b for j in range(len(b))])
        a, ia = (a[keep_a], ia[keep_a])
        b, ib = (b[keep_b], ib[keep_b])
        prior_ok = [
            candidate
            for candidate in potential
            if not any((crosses(candidate, prior) for prior in accepted))
        ]
        current: list[Match] = []
        for candidate in sorted(
            prior_ok, key=lambda item: (item.dissimilarity, item.index_a, item.index_b)
        ):
            if not any((crosses(candidate, other) for other in current)):
                current.append(candidate)
        accepted.extend(current)
        iteration += 1
    return accepted


def mpurge_scores(
    query: np.ndarray,
    references: Sequence[np.ndarray],
    *,
    half_window_p: int = 2,
    alpha: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.sort(np.asarray(query, dtype=np.float64))
    scores = np.full(len(references), np.inf, dtype=np.float64)
    coverage = np.zeros(len(references), dtype=np.float64)
    for index, reference in enumerate(references):
        reference = np.sort(np.asarray(reference, dtype=np.float64))
        matches = pairwise_match(
            query,
            reference,
            half_window_p=half_window_p,
            alpha=alpha,
        )
        if not matches:
            continue
        mean = float(np.mean([item.dissimilarity for item in matches]))
        coverage[index] = len(matches) / max(len(query), len(reference), 1)
        scores[index] = mean / max(coverage[index], np.finfo(np.float64).eps)
    return (scores, coverage)


def inverse_topk(scores: np.ndarray, reference_xy: np.ndarray, k: int = 3) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return centroid(reference_xy)
    selected = finite[np.argsort(scores[finite], kind="stable")[: min(k, len(finite))]]
    values = np.maximum(scores[selected], np.finfo(np.float64).eps)
    weights = 1.0 / values
    weights /= weights.sum()
    return np.sum(reference_xy[selected] * weights[:, None], axis=0)


def localize(queries, survey, reference_xy):
    result = []
    for query in queries:
        scores, _ = mpurge_scores(query, survey)
        result.append(inverse_topk(scores, np.asarray(reference_xy), k=3))
    return np.asarray(result)
