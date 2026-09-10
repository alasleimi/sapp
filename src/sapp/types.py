"""Parameters and fitted survey map."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AnchorParameters:
    source_count: int = 20
    ransac_trials: int = 30_000
    ransac_batch: int = 400
    inlier_scale_m: float = 0.18
    minimum_support_fraction: float = 0.08
    surface_separation_m: float = 0.12
    association_scale_m: float = 0.16
    field_length_m: float = 0.0
    intensity_floor: float = 0.015
    intensity_ceiling: float = 1.8
    survival_alpha: float = 6.0
    survival_beta: float = 2.0
    clutter_shape: float = 2.0
    clutter_prior_rate: float = 1.0
    range_limit_m: float = 25.0
    coarse_step_m: float = 0.10
    refine_step_m: float = 0.025
    decode: str = "map"
    posterior_temperature: float = 1.0
    seed: int = 2026090401


@dataclass
class AnchorMap:
    anchors: np.ndarray
    reference_xy: np.ndarray
    soft_counts: np.ndarray
    global_counts: np.ndarray
    clutter_rate: float
    parameters: AnchorParameters
    discovery_support: np.ndarray
    discovery_residual_m: np.ndarray
    background_delay: np.ndarray | None = None
    background_mask: np.ndarray | None = None
    background_responsibility: np.ndarray | None = None
