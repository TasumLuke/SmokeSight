"""Uncertainty propagation helpers.

The MC helper is seeded with default_rng(42) so two runs on the same
inputs produce identical output; callers don't get to override the seed.
The "sigma" reported by monte_carlo is the (p84-p16)/2 half-range, not
std() of the samples — that's the convention SmokeSight uses everywhere
it ships an MC-derived uncertainty.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, Tuple, Union

import numpy as np

from smokesight._types import FloatArray

ArrayOrFloat = Union[FloatArray, float]

_MC_SEED = 42
_MC_MIN_SAMPLES = 1000


class _SensorNoise(Protocol):
    """The slice of SensorModel that radiance_uncertainty needs."""

    gain: float
    noise_equivalent_radiance: float
    flat_field_relative_uncertainty: float


class _AtmosNoise(Protocol):
    def uncertainty(self, L: FloatArray) -> FloatArray: ...


def radiance_uncertainty(
    L: FloatArray,
    sensor: _SensorNoise,
    atmos: _AtmosNoise,
) -> FloatArray:
    """sigma_L = sqrt(shot^2 + read^2 + flat^2 + atmos^2), per pixel."""
    L = np.asarray(L)
    L_pos = np.maximum(L, 0.0)  # negative DN can't generate shot noise

    shot = np.sqrt(L_pos * sensor.gain)
    read = np.full_like(L_pos, sensor.noise_equivalent_radiance, dtype=L_pos.dtype)
    flat = L_pos * sensor.flat_field_relative_uncertainty
    atmos_sigma = np.asarray(atmos.uncertainty(L_pos), dtype=L_pos.dtype)

    sigma = np.sqrt(shot**2 + read**2 + flat**2 + atmos_sigma**2)
    return np.asarray(sigma.astype(np.float32, copy=False))


def tau_uncertainty(
    L: FloatArray,
    sigma_L: FloatArray,
    L0: FloatArray,
    sigma_L0: FloatArray,
) -> FloatArray:
    """sigma_tau = sqrt((sigma_L/L)^2 + (sigma_L0/L0)^2), NaN where invalid.

    First-order propagation through tau = -ln(L / L0). Invalid =
    L<=0, L0<=0, or non-finite — in those cases sigma_tau is NaN, which
    matches the contract that masked tau pixels carry no numeric sigma.
    """
    L = np.asarray(L, dtype=np.float64)
    sigma_L = np.asarray(sigma_L, dtype=np.float64)
    L0 = np.asarray(L0, dtype=np.float64)
    sigma_L0 = np.asarray(sigma_L0, dtype=np.float64)

    invalid = (L <= 0) | (L0 <= 0) | ~np.isfinite(L) | ~np.isfinite(L0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt((sigma_L / L) ** 2 + (sigma_L0 / L0) ** 2)
    sigma = np.where(invalid, np.nan, sigma)
    return np.asarray(sigma.astype(np.float32, copy=False))


def centroid_uncertainty(tau: FloatArray, sigma_tau: FloatArray) -> Tuple[float, float]:
    """1-sigma uncertainty on a tau-weighted 2-D centroid (delta method)."""
    tau = np.asarray(tau, dtype=np.float64)
    sigma_tau = np.asarray(sigma_tau, dtype=np.float64)
    if tau.shape != sigma_tau.shape:
        raise ValueError(
            f"tau and sigma_tau must share shape; got {tau.shape} vs {sigma_tau.shape}"
        )
    if tau.ndim != 2:
        raise ValueError(f"tau must be 2-D, got shape {tau.shape}")

    valid = np.isfinite(tau) & np.isfinite(sigma_tau) & (tau > 0)
    if not valid.any():
        return float("nan"), float("nan")

    weights = np.where(valid, tau, 0.0)
    sigmas = np.where(valid, sigma_tau, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float("nan"), float("nan")

    yy, xx = np.mgrid[0 : tau.shape[0], 0 : tau.shape[1]]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)

    # d(cx)/d(w_i) = (x_i - cx) / total, then square and sum.
    var_x = float(((xx - cx) ** 2 * sigmas**2).sum() / total**2)
    var_y = float(((yy - cy) ** 2 * sigmas**2).sum() / total**2)
    return float(np.sqrt(var_x)), float(np.sqrt(var_y))


def gaussian_fit_uncertainty(pcov: FloatArray) -> FloatArray:
    """sqrt(diag(pcov)) with non-finite/negative diagonals coerced to NaN.

    curve_fit returns ill-conditioned covariances as inf/-inf or with
    negative diagonals; rather than emit imaginary uncertainties we mark
    those parameters NaN so the caller treats them as not-determined.
    """
    pcov = np.asarray(pcov, dtype=np.float64)
    if pcov.ndim != 2 or pcov.shape[0] != pcov.shape[1]:
        raise ValueError(f"pcov must be square; got shape {pcov.shape}")
    diag = np.diag(pcov)
    safe = np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan)
    return np.asarray(np.sqrt(safe))


def monte_carlo(
    func: Callable[..., FloatArray],
    inputs: Sequence[ArrayOrFloat],
    sigmas: Sequence[ArrayOrFloat],
    n: int = _MC_MIN_SAMPLES,
) -> Tuple[FloatArray, FloatArray]:
    """Run ``func`` on n Gaussian-perturbed copies of inputs.

    Returns (mean, sigma) where sigma is (p84 - p16) / 2 across samples.
    The RNG is seeded internally — pass everything you need through
    ``inputs`` and ``sigmas``, not via globals.
    """
    if n < _MC_MIN_SAMPLES:
        raise ValueError(
            f"monte_carlo requires n >= {_MC_MIN_SAMPLES} samples; got {n}"
        )
    if len(inputs) != len(sigmas):
        raise ValueError(
            f"inputs and sigmas must be the same length; "
            f"got {len(inputs)} and {len(sigmas)}"
        )

    rng = np.random.default_rng(_MC_SEED)
    values = [np.asarray(v, dtype=np.float64) for v in inputs]
    deviations = [np.asarray(s, dtype=np.float64) for s in sigmas]

    samples = np.empty((n,) + np.shape(func(*values)), dtype=np.float64)
    for i in range(n):
        perturbed = [
            v + rng.standard_normal(v.shape) * s for v, s in zip(values, deviations)
        ]
        samples[i] = np.asarray(func(*perturbed), dtype=np.float64)

    mean = samples.mean(axis=0)
    p16, p84 = np.percentile(samples, [16, 84], axis=0)
    return mean, (p84 - p16) / 2.0
