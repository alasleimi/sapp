"""Fit sources, expected counts and residual survey weights."""

import numpy as np
from .types import AnchorMap, AnchorParameters
from .discovery import discover_anchors
from .kernels import range_surfaces, _student_density, _pack_sets


def soft_association_counts(
    reference_xy: np.ndarray,
    sets: list[np.ndarray],
    anchors: np.ndarray,
    parameters: AnchorParameters,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate source and clutter counts without forced peak assignment."""

    source_count = len(anchors)
    global_count = np.full(source_count, 0.35, dtype=np.float64)
    clutter = 1.0
    counts = np.zeros((len(reference_xy), source_count), dtype=np.float64)
    for _ in range(iterations):
        locations = range_surfaces(reference_xy, anchors)
        clutter_counts = []
        counts.fill(0.0)
        for row, observed in enumerate(sets):
            values = np.asarray(observed, dtype=np.float64)
            if not len(values):
                clutter_counts.append(0.0)
                continue
            density = _student_density(
                values[:, None],
                locations[row][None, :],
                parameters.association_scale_m,
                parameters.range_limit_m,
            )
            weighted = density * global_count[None, :]
            clutter_density = clutter / parameters.range_limit_m
            denominator = clutter_density + np.sum(weighted, axis=1)
            responsibility = weighted / np.maximum(denominator[:, None], 1.0e-15)
            counts[row] = np.sum(responsibility, axis=0)
            clutter_counts.append(float(np.sum(clutter_density / denominator)))
        global_count = np.clip(
            (np.sum(counts, axis=0) + 0.25) / (len(reference_xy) + 0.5),
            parameters.intensity_floor,
            parameters.intensity_ceiling,
        )
        clutter = float(
            np.clip(
                (np.sum(clutter_counts) + 0.5) / (len(reference_xy) + 0.5),
                0.05,
                6.0,
            )
        )
    return counts.copy(), global_count, clutter


def fit(
    reference_xy: np.ndarray,
    sets: list[np.ndarray],
    parameters: AnchorParameters = AnchorParameters(),
) -> AnchorMap:
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    anchors, support, residual = discover_anchors(reference_xy, sets, parameters)
    counts, global_counts, clutter = soft_association_counts(
        reference_xy,
        sets,
        anchors,
        parameters,
    )
    fitted = AnchorMap(
        anchors=anchors,
        reference_xy=reference_xy,
        soft_counts=counts,
        global_counts=global_counts,
        clutter_rate=clutter,
        parameters=parameters,
        discovery_support=support,
        discovery_residual_m=residual,
    )
    calibrate_background(fitted, sets)
    return fitted


def calibrate_background(model: AnchorMap, sets: list[np.ndarray]) -> None:
    """Store survey peaks not explained by inferred propagation surfaces.

    Responsibilities are computed under the final survey association model.
    """

    packed, mask = _pack_sets(sets)
    locations = range_surfaces(model.reference_xy, model.anchors)
    responsibility = np.zeros_like(packed)
    for row in range(len(model.reference_xy)):
        valid = mask[row]
        if not np.any(valid):
            continue
        density = _student_density(
            packed[row, valid, None],
            locations[row][None, :],
            model.parameters.association_scale_m,
            model.parameters.range_limit_m,
        )
        source = density * model.global_counts[None, :]
        clutter_density = model.clutter_rate / model.parameters.range_limit_m
        denominator = clutter_density + np.sum(source, axis=1)
        responsibility[row, valid] = clutter_density / np.maximum(
            denominator,
            1.0e-15,
        )
    model.background_delay = packed
    model.background_mask = mask
    model.background_responsibility = responsibility
