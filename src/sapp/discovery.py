"""Consensus fitting of anonymous survey delays."""

import math
import numpy as np
from scipy.optimize import least_squares
from .types import AnchorParameters
from .kernels import range_surfaces, _pack_sets


def _nearest_residuals(
    predictions: np.ndarray,
    packed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    difference = np.abs(predictions[:, :, None] - packed[None, :, :])
    difference = np.where(mask[None, :, :], difference, np.inf)
    return np.min(difference, axis=2)


def _score_hypotheses(
    anchors: np.ndarray,
    reference_xy: np.ndarray,
    packed: np.ndarray,
    mask: np.ndarray,
    scale_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.sqrt(
        np.sum(
            np.square(reference_xy[None, :, :] - anchors[:, None, :2]),
            axis=2,
        )
        + np.square(anchors[:, None, 2])
    )
    residual = _nearest_residuals(predictions, packed, mask)
    support = np.sum(residual <= scale_m, axis=1)
    robust = np.sum(np.exp(-0.5 * np.square(residual / scale_m)), axis=1)
    median = np.median(np.minimum(residual, 4.0 * scale_m), axis=1)
    score = support + 0.35 * robust - 0.10 * median / scale_m
    return score, support, residual


def _solve_three(
    xy: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray | None:
    delta = xy[1:] - xy[0]
    matrix = 2.0 * delta
    if abs(float(np.linalg.det(matrix))) < 0.04:
        return None
    right = (
        np.sum(np.square(xy[1:]), axis=1)
        - np.sum(np.square(xy[0]))
        - (np.square(ranges[1:]) - ranges[0] ** 2)
    )
    centre = np.linalg.solve(matrix, right)
    height_square = ranges[0] ** 2 - float(np.sum(np.square(xy[0] - centre)))
    if height_square < -0.02:
        return None
    return np.asarray([centre[0], centre[1], math.sqrt(max(0.0, height_square))])


def _refine_anchor(
    anchor: np.ndarray,
    reference_xy: np.ndarray,
    packed: np.ndarray,
    mask: np.ndarray,
    threshold_m: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    prediction = range_surfaces(reference_xy, anchor[None, :])[:, 0]
    difference = np.abs(prediction[:, None] - packed)
    difference = np.where(mask, difference, np.inf)
    closest = np.argmin(difference, axis=1)
    residual = difference[np.arange(len(reference_xy)), closest]
    selected = residual <= threshold_m
    if np.sum(selected) < 5:
        return anchor
    target = packed[np.arange(len(reference_xy)), closest]
    if weights is None:
        local_weight = np.exp(-0.5 * np.square(residual / threshold_m))
    else:
        local_weight = np.asarray(weights, dtype=np.float64)
    selected &= local_weight > 1.0e-4
    if np.sum(selected) < 5:
        return anchor
    lower_xy = np.min(reference_xy, axis=0) - 4.0 * np.ptp(reference_xy, axis=0) - 5.0
    upper_xy = np.max(reference_xy, axis=0) + 4.0 * np.ptp(reference_xy, axis=0) + 5.0

    def residual_function(value: np.ndarray) -> np.ndarray:
        predicted = np.sqrt(
            np.sum(np.square(reference_xy[selected] - value[:2]), axis=1) + value[2] ** 2
        )
        return np.sqrt(local_weight[selected]) * (predicted - target[selected])

    fitted = least_squares(
        residual_function,
        np.asarray(anchor, dtype=np.float64),
        bounds=(
            np.asarray([lower_xy[0], lower_xy[1], 0.0]),
            np.asarray([upper_xy[0], upper_xy[1], 15.0]),
        ),
        loss="soft_l1",
        f_scale=max(threshold_m / 2.0, 0.03),
        max_nfev=120,
    )
    return np.asarray(fitted.x, dtype=np.float64)


def discover_anchors(
    reference_xy: np.ndarray,
    sets: list[np.ndarray],
    parameters: AnchorParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine reproducible range surfaces from unordered survey delays."""

    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    packed, mask = _pack_sets(sets)
    if packed.shape[1] == 0:
        raise ValueError("anchor discovery requires nonempty survey delay sets")
    rng = np.random.default_rng(parameters.seed)
    length = np.sum(mask, axis=1)
    eligible = np.flatnonzero(length > 0)
    if len(eligible) < 3:
        raise ValueError("anchor discovery requires three nonempty survey positions")

    candidates: list[np.ndarray] = []
    scores: list[float] = []
    supports: list[int] = []
    residual_rows: list[np.ndarray] = []

    trial_count = 0
    while trial_count < parameters.ransac_trials:
        batch_size = min(parameters.ransac_batch, parameters.ransac_trials - trial_count)
        hypotheses = []
        for _ in range(batch_size):
            indices = rng.choice(eligible, size=3, replace=False)
            values = np.asarray(
                [packed[index, rng.integers(0, length[index])] for index in indices]
            )
            candidate = _solve_three(reference_xy[indices], values)
            if candidate is None:
                continue
            span = np.maximum(np.ptp(reference_xy, axis=0), 1.0)
            if np.any(candidate[:2] < np.min(reference_xy, axis=0) - 5.0 * span):
                continue
            if np.any(candidate[:2] > np.max(reference_xy, axis=0) + 5.0 * span):
                continue
            if candidate[2] > 15.0:
                continue
            hypotheses.append(candidate)
        trial_count += batch_size
        if not hypotheses:
            continue
        hypothesis_array = np.asarray(hypotheses, dtype=np.float64)
        score, support, residual = _score_hypotheses(
            hypothesis_array,
            reference_xy,
            packed,
            mask,
            parameters.inlier_scale_m,
        )
        keep = np.argsort(score)[-min(12, len(score)) :]
        candidates.extend(hypothesis_array[keep])
        scores.extend(score[keep].tolist())
        supports.extend(support[keep].tolist())
        residual_rows.extend(residual[keep])

    candidate_array = np.asarray(candidates, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    support_array = np.asarray(supports, dtype=np.int64)
    residual_array = np.asarray(residual_rows, dtype=np.float64)
    minimum_support = max(
        5,
        int(
            math.ceil(
                parameters.minimum_support_fraction * len(reference_xy),
            )
        ),
    )
    del score_array, support_array, residual_array
    selected: list[np.ndarray] = []
    selected_support: list[int] = []
    selected_residual: list[float] = []
    selected_surface: list[np.ndarray] = []
    active_mask = mask.copy()
    available = np.ones(len(candidate_array), dtype=bool)
    for _ in range(parameters.source_count):
        active_index = np.flatnonzero(available)
        if not len(active_index):
            break
        best_index = None
        best_key = None
        for start in range(0, len(active_index), parameters.ransac_batch):
            index = active_index[start : start + parameters.ransac_batch]
            score, support, _ = _score_hypotheses(
                candidate_array[index],
                reference_xy,
                packed,
                active_mask,
                parameters.inlier_scale_m,
            )
            local = int(np.argmax(score))
            key = (float(score[local]), int(support[local]))
            if best_key is None or key > best_key:
                best_key = key
                best_index = int(index[local])
        if best_index is None or best_key[1] < minimum_support:
            break
        candidate = _refine_anchor(
            candidate_array[best_index],
            reference_xy,
            packed,
            active_mask,
            2.0 * parameters.inlier_scale_m,
        )
        surface = range_surfaces(reference_xy, candidate[None, :])[:, 0]
        if (
            selected_surface
            and min(float(np.sqrt(np.mean(np.square(surface - old)))) for old in selected_surface)
            < parameters.surface_separation_m
        ):
            available[best_index] = False
            continue
        _, support, residual = _score_hypotheses(
            candidate[None, :],
            reference_xy,
            packed,
            active_mask,
            parameters.inlier_scale_m,
        )
        if int(support[0]) < minimum_support:
            available[best_index] = False
            continue
        selected.append(candidate)
        selected_surface.append(surface)
        selected_support.append(int(support[0]))
        selected_residual.append(
            float(
                np.median(
                    residual[0, residual[0] <= parameters.inlier_scale_m],
                )
            )
        )
        difference = np.where(
            active_mask,
            np.abs(surface[:, None] - packed),
            np.inf,
        )
        closest = np.argmin(difference, axis=1)
        closest_residual = difference[np.arange(len(reference_xy)), closest]
        rows = np.flatnonzero(closest_residual <= parameters.inlier_scale_m)
        active_mask[rows, closest[rows]] = False
        available_index = np.flatnonzero(available)
        predicted = range_surfaces(reference_xy, candidate_array[available_index])
        rms = np.sqrt(np.mean(np.square(predicted - surface[:, None]), axis=0))
        available[available_index[rms < parameters.surface_separation_m]] = False
    if len(selected) < 3:
        raise RuntimeError(f"anchor discovery found only {len(selected)} surfaces")
    return (
        np.asarray(selected, dtype=np.float64),
        np.asarray(selected_support, dtype=np.int64),
        np.asarray(selected_residual, dtype=np.float64),
    )
