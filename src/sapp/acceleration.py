"""Batched analytic scoring, UWB delay tables and geometric-median decoding."""

import math
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.special import stdtr, gammaln
from scipy.signal import fftconvolve
import torch
from .fields import (
    survey_kernel_bandwidth,
    predict_intensity,
    _prepare_background,
    _background_density,
)
from .kernels import range_surfaces
from .likelihood import semiparametric_fixed_nuisance_log_density
from ._median_numba import geometric_median as median
from .experiments.common import write_json as write

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def spatial(model, points):
    h = survey_kernel_bandwidth(model)
    distance = cKDTree(points).sparse_distance_matrix(
        cKDTree(model.reference_xy), h * math.sqrt(-2 * math.log(1e-12)), output_type="coo_matrix"
    )
    distance.data = np.exp(-(distance.data**2) / (2 * h * h))
    return distance.tocsr()


def field(model, grid, step=0.0125):
    """Return total intensity and integral on the existing SAPP candidate grid."""
    p = model.parameters
    sigma = p.association_scale_m
    length = int(math.ceil(p.range_limit_m / step)) + 1
    axis = np.arange(length) * step
    const = math.exp(
        float(gammaln(2) - gammaln(1.5)) - 0.5 * math.log(3 * math.pi) - math.log(sigma)
    )
    offsets = np.arange(-(length - 1), length) * step
    kernel = const / (1 + (offsets / sigma) ** 2 / 3) ** 2
    delay = model.background_delay
    mask = model.background_mask
    resp = model.background_responsibility
    rows, cols = np.nonzero(mask)
    location = delay[rows, cols]
    normalization = np.maximum(
        stdtr(3, (p.range_limit_m - location) / sigma) - stdtr(3, -location / sigma), 1e-15
    )
    mass = resp[rows, cols] / normalization
    fraction = location / step
    left = np.floor(fraction).astype(int)
    alpha = fraction - left
    histogram = csr_matrix(
        (
            np.r_[mass * (1 - alpha), mass * alpha],
            (np.r_[rows, rows], np.r_[left, np.minimum(left + 1, length - 1)]),
        ),
        shape=(len(delay), length),
    )
    global_histogram = np.asarray(histogram.mean(axis=0)).ravel()
    row_integral = resp.sum(axis=1)
    global_integral = float(row_integral.mean())
    # Chunking bounds memory even for the full training trajectory.
    table = np.empty((len(grid), length), dtype=np.float32)
    integrated = np.empty(len(grid), dtype=np.float64)
    dev_axis = torch.as_tensor(axis, dtype=torch.float32, device=DEVICE)
    for lo in range(0, len(grid), 64):
        hi = min(lo + 64, len(grid))
        points = grid[lo:hi]
        w = spatial(model, points)
        denominator = np.asarray(w.sum(axis=1)).ravel() + 0.5
        hist = (w @ histogram).toarray()
        hist += 0.5 * global_histogram[None, :]
        hist /= denominator[:, None]
        background = fftconvolve(hist, kernel[None, :], mode="same", axes=1)
        intensity = np.clip(
            (w @ model.soft_counts + 0.5 * model.global_counts[None, :]) / denominator[:, None],
            p.intensity_floor,
            p.intensity_ceiling,
        )
        integrated[lo:hi] = (
            w @ row_integral + 0.5 * global_integral
        ) / denominator + intensity.sum(axis=1)
        if len(model.anchors):
            ranges = range_surfaces(points, model.anchors)
            norm = np.maximum(
                stdtr(3, (p.range_limit_m - ranges) / sigma) - stdtr(3, -ranges / sigma), 1e-15
            )
            coefficient = torch.as_tensor(
                const * intensity / norm, dtype=torch.float32, device=DEVICE
            )
            r = torch.as_tensor(ranges, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                delta = (dev_axis[None, :, None] - r[:, None, :]) / sigma
                source = (
                    (coefficient[:, None, :] / (1 + delta.square() / 3).square())
                    .sum(dim=2)
                    .cpu()
                    .numpy()
                )
            background += source
        table[lo:hi] = np.maximum(background, 0).astype(np.float32)
    return table, integrated


def scores(table, integrated, query, limit, step=0.0125):
    # Return query-by-grid log likelihoods; no position prior is counted per anchor.
    values = np.where(np.isfinite(query), query, 0)
    fraction = values / step
    left = np.minimum(np.floor(fraction).astype(np.int64), table.shape[1] - 2)
    alpha = fraction - left
    mask = np.isfinite(query)
    result = np.empty((len(query), len(table)), dtype=np.float32)
    dev_table = torch.as_tensor(table, device=DEVICE)
    dev_integral = torch.as_tensor(integrated, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for lo in range(0, len(query), 32):
            hi = min(lo + 32, len(query))
            ix = torch.as_tensor(left[lo:hi], device=DEVICE)
            weight = torch.as_tensor(alpha[lo:hi], dtype=torch.float32, device=DEVICE)
            valid = torch.as_tensor(mask[lo:hi], device=DEVICE)
            density = (
                dev_table[:, ix] * (1 - weight[None, :, :])
                + dev_table[:, ix + 1] * weight[None, :, :]
            )
            score = (
                (
                    torch.log(torch.clamp(0.75 * density + 2.0 / limit, min=1e-30))
                    * valid[None, :, :]
                ).sum(dim=2)
                - 0.75 * dev_integral[:, None]
                - 2.0
            )
            result[lo:hi] = score.T.cpu().numpy()
    del dev_table
    return result


@njit(cache=True, parallel=True)
def _decode(score, grid, temperature):
    out = np.empty((len(score), 2))
    for i in prange(len(score)):
        row = score[i].astype(np.float64)
        weights = np.exp(np.maximum((row - row.max()) / temperature, -80.0))
        weights /= weights.sum()
        initial = np.zeros(2)
        for j in range(len(grid)):
            initial[0] += weights[j] * grid[j, 0]
            initial[1] += weights[j] * grid[j, 1]
        active = weights > 1e-10
        out[i] = median(grid[active], weights[active], initial)
    return out


def decode(score, grid, temperature, radius, xy):
    out = _decode(score, grid, float(temperature))
    outside = cKDTree(xy).query(out)[0] > radius
    out[outside] = grid[cKDTree(grid).query(out[outside])[1]]
    return out


def score_model(model, grid, query, path):
    table, integrated = field(model, grid, step=0.0125)
    ix = np.unique(np.linspace(0, len(grid) - 1, min(41, len(grid)), dtype=int))
    qi = np.unique(np.linspace(0, len(query) - 1, min(7, len(query)), dtype=int))
    points = grid[ix]
    p = model.parameters
    prep = _prepare_background(model, points)
    ranges = range_surfaces(points, model.anchors)
    counts = predict_intensity(model, points)
    actual = scores(table[ix], integrated[ix], query[qi], p.range_limit_m, step=0.0125).astype(
        float
    )
    expected = []
    for row in query[qi]:
        obs = row[np.isfinite(row)]
        bg, bi = _background_density(model, prep, obs)
        expected.append(
            semiparametric_fixed_nuisance_log_density(
                obs,
                ranges,
                counts,
                scale_m=p.association_scale_m,
                clutter_rate=model.clutter_rate,
                range_limit_m=p.range_limit_m,
                background_density=bg,
                background_integral=bi,
            )
        )
    error = float(np.max(np.abs(actual - np.asarray(expected))))
    record = dict(
        status="PASS" if error < 0.01 else "FAIL",
        step_m=0.0125,
        max_absolute_log_likelihood_error=error,
    )
    write(path, record)
    assert record["status"] == "PASS", record
    return scores(table, integrated, query, model.parameters.range_limit_m, step=0.0125)


def direct_scores(model, grid, query, path):
    """Evaluate analytic Student intensities in GPU batches, without delay tables."""
    import torch
    import math
    from scipy.special import stdtr, gammaln

    p = model.parameters
    sigma = p.association_scale_m
    limit = p.range_limit_m
    device = DEVICE

    def asdev(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    w = spatial(model, grid).toarray()
    den = w.sum(axis=1) + 0.5
    w = (w + 0.5 / len(model.reference_xy)) / den[:, None]
    weights = asdev(w)
    counts = predict_intensity(model, grid)
    ranges = range_surfaces(grid, model.anchors)
    norm = np.maximum(stdtr(3, (limit - ranges) / sigma) - stdtr(3, -ranges / sigma), 1e-15)
    const = math.exp(
        float(gammaln(2) - gammaln(1.5)) - 0.5 * math.log(3 * math.pi) - math.log(sigma)
    )
    coefficient = asdev(counts * const / norm)
    rg = asdev(ranges)
    integrated = asdev(w @ model.background_responsibility.sum(axis=1) + counts.sum(axis=1))
    bd = model.background_delay
    mask = model.background_mask
    location = np.where(mask, bd, 0.0)
    bn = np.maximum(stdtr(3, (limit - location) / sigma) - stdtr(3, -location / sigma), 1e-15)
    bc = np.where(mask, model.background_responsibility * const / bn, 0.0)
    result = np.empty((len(query), len(grid)), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(query), 16):
            q = query[lo : lo + 16]
            valid = np.isfinite(q)
            obs = np.where(valid, q, 0.0).ravel()
            density = (
                bc[:, :, None]
                / (1 + ((obs[None, None, :] - location[:, :, None]) / sigma) ** 2 / 3) ** 2
            ).sum(axis=1)
            bg = weights @ asdev(density)
            y = asdev(obs)
            for j in range(len(model.anchors)):
                bg += (
                    coefficient[:, j, None]
                    / (1 + ((y[None, :] - rg[:, j, None]) / sigma).square() / 3).square()
                )
            score = (
                (
                    torch.log(torch.clamp(0.75 * bg + 2.0 / limit, min=1e-30)).reshape(
                        len(grid), len(q), q.shape[1]
                    )
                    * asdev(valid)[None, :, :]
                ).sum(axis=2)
                - 0.75 * integrated[:, None]
                - 2.0
            )
            result[lo : lo + len(q)] = score.T.cpu().numpy()
    ix = np.unique(np.linspace(0, len(grid) - 1, min(41, len(grid)), dtype=int))
    qi = np.unique(np.linspace(0, len(query) - 1, min(7, len(query)), dtype=int))
    points = grid[ix]
    prep = _prepare_background(model, points)
    expected = []
    for row in query[qi]:
        obs = row[np.isfinite(row)]
        bg, bi = _background_density(model, prep, obs)
        expected.append(
            semiparametric_fixed_nuisance_log_density(
                obs,
                ranges[ix],
                counts[ix],
                scale_m=sigma,
                clutter_rate=model.clutter_rate,
                range_limit_m=limit,
                background_density=bg,
                background_integral=bi,
            )
        )
    error = float(np.max(np.abs(result[np.ix_(qi, ix)] - np.asarray(expected))))
    record = dict(
        status="PASS" if error < 0.001 else "FAIL",
        max_absolute_log_likelihood_error=error,
        evaluation="analytic GPU batches",
    )
    write(path, record)
    assert record["status"] == "PASS", record
    return result
