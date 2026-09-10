#!/usr/bin/env python3
"""Deterministic, receiver-facing delay-peak extraction for NIST Q-D frames.

The NIST Q-D output supplies specular ray delay, complex gain, and angle.  This
module converts those rays into the output of a disclosed finite-bandwidth
front end.  Rays are coherently superposed at an eight-element array, passed
through the impulse response of an ideal rectangular band, sampled at the
Nyquist delay interval, corrupted by complex AWGN, and reduced to local delay
peaks.  A localizer receives only the resulting sorted delays.

The implementation deliberately does not read ray identities, reflection
orders, object geometry, or receiver coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


C_MPS = 3.0e8


def cube_array_positions(carrier_hz: float = 60.0e9) -> np.ndarray:
    """Eight half-wavelength-spaced elements on a 2x2x2 cube."""

    wavelength = 299_792_458.0 / float(carrier_hz)
    half = wavelength / 2.0
    return np.asarray(
        [[ix * half, iy * half, iz * half] for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)],
        dtype=np.float64,
    )


def nist_angle_to_unit(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Use NIST's polar-elevation convention from angle2vector.m."""

    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    return np.column_stack(
        (
            np.sin(elevation) * np.cos(azimuth),
            np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        )
    )


@dataclass(frozen=True)
class PeakFrontend:
    bandwidth_hz: float = 2.0e9
    carrier_hz: float = 60.0e9
    maximum_range_m: float = 25.0
    snr_db: float = 40.0
    threshold_above_noise_db: float = 8.0
    dynamic_range_db: float = 45.0
    maximum_peaks: int = 24
    parabolic_refinement: bool = True

    def audit(self) -> dict[str, float | int | bool]:
        return {
            **asdict(self),
            "nominal_range_resolution_m": C_MPS / self.bandwidth_hz,
            "array_elements": 8,
        }


def _parabolic_offset(log_power: np.ndarray, index: int) -> float:
    """Return a bounded three-sample quadratic peak offset in bins."""

    if index <= 0 or index >= len(log_power) - 1:
        return 0.0
    left, center, right = map(float, log_power[index - 1 : index + 2])
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator >= -1.0e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def extract_delay_peaks(
    frame,
    *,
    rng: np.random.Generator,
    config: PeakFrontend,
    power_order: bool = False,
) -> np.ndarray:
    """Extract a sorted delay-range set from one native NIST channel frame."""

    resolution_m = C_MPS / float(config.bandwidth_hz)
    sample_ranges = np.arange(
        0.0,
        float(config.maximum_range_m) + 0.5 * resolution_m,
        resolution_m,
        dtype=np.float64,
    )
    clean = np.zeros((8, len(sample_ranges)), dtype=np.complex128)
    if int(frame.path_count):
        path_ranges = np.asarray(frame.delay_s, dtype=np.float64) * C_MPS
        valid = (
            np.isfinite(path_ranges)
            & (path_ranges >= 0.0)
            & (path_ranges <= float(config.maximum_range_m))
        )
        if np.any(valid):
            directions = nist_angle_to_unit(
                np.asarray(frame.aoa_az_deg, dtype=np.float64)[valid],
                np.asarray(frame.aoa_el_deg, dtype=np.float64)[valid],
            )
            amplitudes = np.power(
                10.0,
                np.asarray(frame.gain_db, dtype=np.float64)[valid] / 20.0,
            ) * np.exp(1j * np.asarray(frame.phase_rad, dtype=np.float64)[valid])
            array_positions = cube_array_positions(float(config.carrier_hz))
            wavelength = C_MPS / float(config.carrier_hz)
            for path_range, amplitude, direction in zip(
                path_ranges[valid],
                amplitudes,
                directions,
                strict=True,
            ):
                steering = np.exp(
                    -2j * np.pi * (array_positions @ direction) / wavelength,
                )
                pulse = np.sinc((sample_ranges - path_range) / resolution_m)
                clean += amplitude * steering[:, None] * pulse[None, :]

    clean_power = np.mean(np.abs(clean) ** 2, axis=0)
    strongest_clean = max(float(np.max(clean_power)), 1.0e-20)
    noise_variance = max(
        strongest_clean / np.power(10.0, float(config.snr_db) / 10.0),
        1.0e-20,
    )
    observed = clean + math.sqrt(noise_variance / 2.0) * (
        rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    )
    power = np.mean(np.abs(observed) ** 2, axis=0)
    noise_floor = max(float(np.median(power)) / math.log(2.0), noise_variance, 1.0e-20)
    strongest = max(float(np.max(power)), 1.0e-20)
    threshold = max(
        noise_floor * np.power(10.0, float(config.threshold_above_noise_db) / 10.0),
        strongest * np.power(10.0, -float(config.dynamic_range_db) / 10.0),
    )
    local = np.zeros(len(power), dtype=bool)
    if len(power) >= 3:
        local[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
        local[0] = power[0] > power[1]
        local[-1] = power[-1] > power[-2]
    peaks = np.flatnonzero(local & (power >= threshold))
    if len(peaks) > int(config.maximum_peaks):
        peaks = peaks[np.argsort(-power[peaks], kind="stable")[: int(config.maximum_peaks)]]
    if power_order:
        peaks = peaks[np.argsort(-power[peaks], kind="stable")]
    log_power = np.log(np.maximum(power, 1.0e-30))
    ranges = []
    for peak in peaks:
        offset = _parabolic_offset(log_power, int(peak)) if config.parabolic_refinement else 0.0
        value = (float(peak) + offset) * resolution_m
        if 0.0 < value <= float(config.maximum_range_m):
            ranges.append(value)
    result = np.asarray(ranges, dtype=np.float64)
    return result if power_order else np.sort(result, kind="stable")
