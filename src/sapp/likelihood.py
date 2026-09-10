"""Finite-set query scores and the two count-model ablations."""

import math
import numpy as np
from scipy.special import betaln, gammaln, hyp1f1, logsumexp
from .kernels import _student_density


def _marginal_log_density_from_source(
    source_density: np.ndarray,
    integrated_source: np.ndarray,
    *,
    range_limit_m: float,
    survival_alpha: float,
    survival_beta: float,
    clutter_shape: float,
    clutter_prior_rate: float,
) -> np.ndarray:
    """Integrate beta survival and gamma clutter from source intensities."""

    source_density = np.maximum(
        np.asarray(source_density, dtype=np.float64),
        1.0e-300,
    )
    integrated_source = np.maximum(
        np.asarray(integrated_source, dtype=np.float64),
        0.0,
    )
    candidate_count, count = source_density.shape
    log_coefficient = np.full((candidate_count, count + 1), -np.inf)
    log_coefficient[:, 0] = 0.0
    for observation_index in range(1, count + 1):
        log_source = np.log(source_density[:, observation_index - 1])
        previous = log_coefficient.copy()
        for degree in range(1, observation_index + 1):
            log_coefficient[:, degree] = np.logaddexp(
                previous[:, degree],
                previous[:, degree - 1] + log_source,
            )

    source_terms = []
    for degree in range(count + 1):
        beta_moment = (
            betaln(survival_alpha + degree, survival_beta)
            - betaln(survival_alpha, survival_beta)
            + np.log(
                np.maximum(
                    hyp1f1(
                        survival_alpha + degree,
                        survival_alpha + survival_beta + degree,
                        -integrated_source,
                    ),
                    1.0e-300,
                )
            )
        )
        clutter_degree = count - degree
        gamma_moment = (
            clutter_shape * math.log(clutter_prior_rate)
            - gammaln(clutter_shape)
            + gammaln(clutter_shape + clutter_degree)
            - (clutter_shape + clutter_degree) * math.log(clutter_prior_rate + 1.0)
        )
        source_terms.append(
            log_coefficient[:, degree]
            - clutter_degree * math.log(range_limit_m)
            + beta_moment
            + gamma_moment
        )
    return logsumexp(np.column_stack(source_terms), axis=1)


def semiparametric_marginal_log_density(
    observed: np.ndarray,
    candidate_ranges: np.ndarray,
    source_intensity: np.ndarray,
    *,
    scale_m: float,
    clutter_rate: float,
    range_limit_m: float,
    survival_alpha: float = 6.0,
    survival_beta: float = 2.0,
    clutter_shape: float = 2.0,
    clutter_prior_rate: float = 1.0,
    background_density: np.ndarray | None = None,
    background_integral: np.ndarray | None = None,
) -> np.ndarray:
    """Marginal likelihood for one physical-plus-residual source intensity."""

    del clutter_rate
    source_density, integrated = _semiparametric_components(
        observed,
        candidate_ranges,
        source_intensity,
        scale_m=scale_m,
        range_limit_m=range_limit_m,
        background_density=background_density,
        background_integral=background_integral,
    )
    return _marginal_log_density_from_source(
        source_density,
        integrated,
        range_limit_m=range_limit_m,
        survival_alpha=survival_alpha,
        survival_beta=survival_beta,
        clutter_shape=clutter_shape,
        clutter_prior_rate=clutter_prior_rate,
    )


def _semiparametric_components(
    observed,
    candidate_ranges,
    source_intensity,
    *,
    scale_m,
    range_limit_m,
    background_density,
    background_integral,
):
    values = np.asarray(observed, dtype=np.float64)
    intensity = np.maximum(np.asarray(source_intensity, dtype=np.float64), 0.0)
    ranges = np.asarray(candidate_ranges, dtype=np.float64)
    density = _student_density(values[None, None, :], ranges[:, :, None], scale_m, range_limit_m)
    source_density = np.sum(intensity[:, :, None] * density, axis=1)
    if background_density is not None:
        source_density += background_density
    integrated = np.sum(intensity, axis=1)
    if background_integral is not None:
        integrated += background_integral
    return source_density, integrated


def semiparametric_fixed_nuisance_log_density(
    observed: np.ndarray,
    candidate_ranges: np.ndarray,
    source_intensity: np.ndarray,
    *,
    scale_m: float,
    clutter_rate: float,
    range_limit_m: float,
    survival_alpha: float = 6.0,
    survival_beta: float = 2.0,
    clutter_shape: float = 2.0,
    clutter_prior_rate: float = 1.0,
    background_density: np.ndarray | None = None,
    background_integral: np.ndarray | None = None,
) -> np.ndarray:
    """Finite-set likelihood with development-selected fixed nuisance means."""

    del clutter_rate
    density, integrated = _semiparametric_components(
        observed,
        candidate_ranges,
        source_intensity,
        scale_m=scale_m,
        range_limit_m=range_limit_m,
        background_density=background_density,
        background_integral=background_integral,
    )
    survival = survival_alpha / (survival_alpha + survival_beta)
    innovation = clutter_shape / clutter_prior_rate
    return (
        -survival * integrated
        - innovation
        + np.log(np.maximum(survival * density + innovation / range_limit_m, 1e-300)).sum(axis=1)
    )


def semiparametric_conditional_log_density(
    observed,
    candidate_ranges,
    source_intensity,
    *,
    scale_m,
    clutter_rate,
    range_limit_m,
    survival_alpha=6.0,
    survival_beta=2.0,
    clutter_shape=2.0,
    clutter_prior_rate=1.0,
    background_density=None,
    background_integral=None,
):
    """Condition on observed cardinality, removing the Poisson count term."""
    density, integrated = _semiparametric_components(
        observed,
        candidate_ranges,
        source_intensity,
        scale_m=scale_m,
        range_limit_m=range_limit_m,
        background_density=background_density,
        background_integral=background_integral,
    )
    survival = survival_alpha / (survival_alpha + survival_beta)
    innovation = clutter_shape / clutter_prior_rate
    return np.log(np.maximum(survival * density + innovation / range_limit_m, 1e-300)).sum(
        axis=1
    ) - len(observed) * np.log(survival * integrated + innovation)
