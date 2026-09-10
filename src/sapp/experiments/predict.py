"""Shared numerical paths for benchmark prediction."""

from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree

from sapp.acceleration import direct_scores, decode
from sapp.fields import predict_intensity, _prepare_background, _background_density
from sapp.kernels import range_surfaces
from sapp.likelihood import semiparametric_fixed_nuisance_log_density
from sapp.localization import candidate_grid, localize
from .common import unpack


def prepared(model, points):
    return (
        range_surfaces(points, model.anchors),
        predict_intensity(model, points),
        _prepare_background(model, points),
    )


def scores_at(model, points, observed, cache=None):
    ranges, counts, background = prepared(model, points) if cache is None else cache
    bg, integral = _background_density(model, background, observed)
    p = model.parameters
    return semiparametric_fixed_nuisance_log_density(
        observed,
        ranges,
        counts,
        scale_m=p.association_scale_m,
        range_limit_m=p.range_limit_m,
        clutter_rate=model.clutter_rate,
        survival_alpha=p.survival_alpha,
        survival_beta=p.survival_beta,
        clutter_shape=p.clutter_shape,
        clutter_prior_rate=p.clutter_prior_rate,
        background_density=bg,
        background_integral=integral,
    )


def modes(model, query, grid, radius, coarse):
    centre = coarse.argmax(axis=1)
    largest = np.partition(coarse, -2, axis=1)[:, -2:]
    ties = np.flatnonzero(largest[:, 1] - largest[:, 0] < 5e-5)
    cache = prepared(model, grid) if len(ties) else None
    for i in ties:
        centre[i] = scores_at(model, grid, query[i][np.isfinite(query[i])], cache).argmax()
    p = model.parameters
    tree = cKDTree(model.reference_xy)
    out = np.empty((len(query), 2))
    for ci in np.unique(centre):
        xy = grid[ci]
        width, step = p.coarse_step_m, p.refine_step_m
        x = np.arange(xy[0] - width, xy[0] + width + step / 2, step)
        y = np.arange(xy[1] - width, xy[1] + width + step / 2, step)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        fine = np.column_stack((xx.ravel(), yy.ravel()))
        fine = fine[tree.query(fine)[0] <= radius]
        cache = prepared(model, fine)
        for i in np.flatnonzero(centre == ci):
            observed = np.sort(query[i][np.isfinite(query[i])])
            out[i] = fine[scores_at(model, fine, observed, cache).argmax()]
    out[~np.isfinite(query).any(axis=1)] = model.reference_xy.mean(axis=0)
    return out


def predict(model, query, check_path, mode=False, temperature=1.0, backend="fast"):
    if backend == "numpy":
        m = replace(
            model,
            parameters=replace(
                model.parameters,
                decode="map" if mode else "posterior_median",
                posterior_temperature=temperature,
            ),
        )
        return localize(unpack(query), m)
    grid, radius = candidate_grid(model.reference_xy, model.parameters.coarse_step_m)
    scores = direct_scores(model, grid, query, check_path)
    if mode:
        return modes(model, query, grid, radius, scores)
    out = decode(scores, grid, temperature, radius, model.reference_xy)
    out[~np.isfinite(query).any(axis=1)] = model.reference_xy.mean(axis=0)
    return out
