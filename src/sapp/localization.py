"""Candidate support, likelihood evaluation and position decoding."""

import numpy as np
from scipy.spatial import cKDTree
from .types import AnchorMap
from .kernels import range_surfaces
from .fields import predict_intensity, _prepare_background, _background_density
from .likelihood import (
    semiparametric_marginal_log_density,
    semiparametric_fixed_nuisance_log_density,
    semiparametric_conditional_log_density,
)


def candidate_grid(
    reference_xy: np.ndarray,
    step_m: float,
) -> tuple[np.ndarray, float]:
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    tree = cKDTree(reference_xy)
    distance, _ = tree.query(reference_xy, k=min(2, len(reference_xy)))
    if len(reference_xy) > 1:
        spacing = float(np.median(distance[:, -1]))
    else:
        spacing = 1.0
    padding = max(0.5 * spacing, step_m)
    support_radius = max(0.82 * spacing, 2.0 * step_m)
    lower = np.min(reference_xy, axis=0) - padding
    upper = np.max(reference_xy, axis=0) + padding
    x = np.arange(lower[0], upper[0] + step_m / 2.0, step_m)
    y = np.arange(lower[1], upper[1] + step_m / 2.0, step_m)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    nearest, _ = tree.query(grid, k=1)
    return grid[nearest <= support_radius], support_radius


def localize(
    sets: list[np.ndarray],
    model: AnchorMap,
    *,
    likelihood=semiparametric_fixed_nuisance_log_density,
) -> np.ndarray:
    parameters = model.parameters
    if parameters.decode not in {"map", "posterior_median"}:
        raise ValueError("decode must be 'map' or 'posterior_median'")
    semiparametric = likelihood in {
        semiparametric_marginal_log_density,
        semiparametric_fixed_nuisance_log_density,
        semiparametric_conditional_log_density,
    }
    coarse, support_radius = candidate_grid(model.reference_xy, parameters.coarse_step_m)
    coarse_ranges = range_surfaces(coarse, model.anchors)
    coarse_intensity = predict_intensity(model, coarse)
    coarse_background = _prepare_background(model, coarse) if semiparametric else None
    reference_tree = cKDTree(model.reference_xy)
    output = []
    for observed in sets:
        values = np.asarray(observed, dtype=np.float64)
        if not len(values):
            output.append(np.mean(model.reference_xy, axis=0))
            continue
        extra = {}
        if semiparametric:
            extra = {
                "survival_alpha": parameters.survival_alpha,
                "survival_beta": parameters.survival_beta,
                "clutter_shape": parameters.clutter_shape,
                "clutter_prior_rate": parameters.clutter_prior_rate,
            }
            background_density, background_integral = _background_density(
                model, coarse_background, values
            )
            extra.update(
                {
                    "background_density": background_density,
                    "background_integral": background_integral,
                }
            )
        score = likelihood(
            values,
            coarse_ranges,
            coarse_intensity,
            scale_m=parameters.association_scale_m,
            clutter_rate=model.clutter_rate,
            range_limit_m=parameters.range_limit_m,
            **extra,
        )
        if parameters.decode == "posterior_median":
            shifted = (score - np.max(score)) / max(parameters.posterior_temperature, 1e-06)
            weight = np.exp(np.clip(shifted, -80.0, 0.0))
            weight /= np.sum(weight)
            estimate = weight @ coarse
            active = weight > 1e-10
            points = coarse[active]
            local_weight = weight[active]
            for _ in range(40):
                distance = np.linalg.norm(points - estimate, axis=1)
                if np.min(distance) < 1e-10:
                    estimate = points[int(np.argmin(distance))]
                    break
                updated = np.sum((local_weight / distance)[:, None] * points, axis=0) / np.sum(
                    local_weight / distance
                )
                if np.linalg.norm(updated - estimate) < 1e-07:
                    estimate = updated
                    break
                estimate = updated
            nearest_support, _ = reference_tree.query(estimate[None, :], k=1)
            if float(nearest_support[0]) > support_radius:
                estimate = coarse[int(np.argmin(np.linalg.norm(coarse - estimate, axis=1)))]
            output.append(np.asarray(estimate, dtype=np.float64))
            continue
        centre = coarse[int(np.argmax(score))]
        radius = parameters.coarse_step_m
        x = np.arange(
            centre[0] - radius,
            centre[0] + radius + parameters.refine_step_m / 2.0,
            parameters.refine_step_m,
        )
        y = np.arange(
            centre[1] - radius,
            centre[1] + radius + parameters.refine_step_m / 2.0,
            parameters.refine_step_m,
        )
        xx, yy = np.meshgrid(x, y, indexing="xy")
        local = np.column_stack((xx.ravel(), yy.ravel()))
        nearest, _ = reference_tree.query(local, k=1)
        local = local[nearest <= support_radius]
        local_extra = dict(extra)
        if semiparametric:
            local_background = _prepare_background(model, local)
            background_density, background_integral = _background_density(
                model, local_background, values
            )
            local_extra.update(
                {
                    "background_density": background_density,
                    "background_integral": background_integral,
                }
            )
        local_score = likelihood(
            values,
            range_surfaces(local, model.anchors),
            predict_intensity(model, local),
            scale_m=parameters.association_scale_m,
            clutter_rate=model.clutter_rate,
            range_limit_m=parameters.range_limit_m,
            **local_extra,
        )
        output.append(local[int(np.argmax(local_score))])
    return np.asarray(output, dtype=np.float64)
