"""Uncertainty propagation utilities.

This is the most critical private module. Every measurement quantity that
SmokeSight returns must have a documented uncertainty path that goes
through one of the helpers below.

The Monte Carlo helper is *internally seeded* with ``np.random.default_rng(42)``;
callers must not pass an external seed. This is so two runs on the same
inputs produce identical uncertainty estimates and CI is deterministic.

When a Monte Carlo result is reported as ``sigma``, it is computed as
``(p84 - p16) / 2`` (the 16th-84th percentile half-range), not the
sample standard deviation. This matches the SmokeSight reporting rule
in the design guide section 6.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np

from smokesight._types import FloatArray

ArrayOrFloat = Union[FloatArray, float]
MC_SEED = 42
MC_DEFAULT_N = 1000


def radiance_uncertainty(
    L: FloatArray,
    sensor: Any,
    atmos: Any,
) -> FloatArray:
    """Combined 1-sigma radiance uncertainty.

    ``sigma_L = sqrt(shot^2 + read^2 + flatfield^2 + atmos^2)``

    * Shot noise: ``sqrt(L * gain)`` (radiance-domain Poisson).
    * Read noise: ``sensor.noise_equivalent_radiance``, broadcast.
    * Flat-field uncertainty: ``L * sensor.flat_field_relative_uncertainty``.
    * Atmospheric uncertainty: ``atmos.uncertainty(L)``.
    """
    L = np.asarray(L)
    L_clipped = np.maximum(L, 0.0)
    gain = float(getattr(sensor, "gain"))
    ner = float(getattr(sensor, "noise_equivalent_radiance"))
    ff_rel = float(getattr(sensor, "flat_field_relative_uncertainty", 0.01))

    sigma_shot = np.sqrt(L_clipped * gain)
    sigma_read = np.full_like(L_clipped, ner, dtype=L_clipped.dtype)
    sigma_flat = np.abs(L_clipped) * ff_rel
    sigma_atmos = np.asarray(atmos.uncertainty(L_clipped), dtype=L_clipped.dtype)

    sigma_L = np.sqrt(sigma_shot**2 + sigma_read**2 + sigma_flat**2 + sigma_atmos**2)
    return np.asarray(sigma_L.astype(np.float32, copy=False))


def tau_uncertainty(
    L: FloatArray,
    sigma_L: FloatArray,
    L0: FloatArray,
    sigma_L0: FloatArray,
) -> FloatArray:
    """Analytic Beer-Lambert uncertainty.

    For ``tau = -ln(L / L0)``, first-order error propagation gives::

        sigma_tau = sqrt( (sigma_L / L)^2 + (sigma_L0 / L0)^2 )

    Pixels where either ``L`` or ``L0`` is zero or non-finite produce NaN
    in the output (consistent with the masking rules in the design guide
    section 6).
    """
    L = np.asarray(L, dtype=np.float64)
    sigma_L = np.asarray(sigma_L, dtype=np.float64)
    L0 = np.asarray(L0, dtype=np.float64)
    sigma_L0 = np.asarray(sigma_L0, dtype=np.float64)

    invalid = (L <= 0) | (L0 <= 0) | ~np.isfinite(L) | ~np.isfinite(L0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_L = sigma_L / L
        rel_L0 = sigma_L0 / L0
        sigma_tau = np.sqrt(rel_L**2 + rel_L0**2)
    sigma_tau = np.where(invalid, np.nan, sigma_tau)
    return np.asarray(sigma_tau.astype(np.float32, copy=False))


def centroid_uncertainty(
    tau: FloatArray,
    sigma_tau: FloatArray,
) -> Tuple[float, float]:
    """Delta-method 1-sigma uncertainty on the tau-weighted centroid.

    For a centroid ``x_c = sum(w_i * x_i) / sum(w_i)`` with weights
    ``w_i = tau_i`` and weight uncertainties ``sigma_i``, the propagated
    uncertainty on ``x_c`` is::

        sigma_x_c^2 = sum( (d x_c / d w_i)^2 * sigma_i^2 )
                    = sum( ((x_i - x_c) / W)^2 * sigma_i^2 )

    where ``W = sum(w_i)``. Returns ``(sigma_x, sigma_y)`` for a 2-D image.
    NaN-valued pixels are dropped from both the centroid and uncertainty.
    """
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

    h, w = tau.shape
    yy, xx = np.mgrid[0:h, 0:w]
    weights = np.where(valid, tau, 0.0)
    sigma = np.where(valid, sigma_tau, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float("nan"), float("nan")

    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)

    var_x = float(((xx - cx) ** 2 * sigma**2).sum() / total**2)
    var_y = float(((yy - cy) ** 2 * sigma**2).sum() / total**2)
    return float(np.sqrt(var_x)), float(np.sqrt(var_y))


def gaussian_fit_uncertainty(pcov: FloatArray) -> FloatArray:
    """1-sigma parameter uncertainties from a scipy.optimize.curve_fit covariance.

    Returns ``sqrt(diag(pcov))``. Diagonal entries that are non-finite or
    negative (which curve_fit can produce when the fit is ill-conditioned)
    are returned as NaN rather than imaginary numbers.
    """
    pcov = np.asarray(pcov, dtype=np.float64)
    if pcov.ndim != 2 or pcov.shape[0] != pcov.shape[1]:
        raise ValueError(f"pcov must be a square matrix; got shape {pcov.shape}")
    diag = np.diag(pcov)
    safe = np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan)
    return np.asarray(np.sqrt(safe))


def monte_carlo(
    func: Callable[..., FloatArray],
    inputs: Sequence[ArrayOrFloat],
    sigmas: Sequence[ArrayOrFloat],
    n: int = MC_DEFAULT_N,
) -> Tuple[FloatArray, FloatArray]:
    """Generic Monte Carlo propagation through ``func``.

    Draws ``n`` Gaussian-perturbed copies of each input (using the
    internal seeded RNG, ``default_rng(42)``), evaluates ``func`` on each,
    and returns ``(mean, sigma)`` where ``sigma`` is the percentile-based
    1-sigma half-range ``(p84 - p16) / 2`` per the design guide section 6.

    Notes
    -----
    Callers must not pass an external seed. Reproducibility is guaranteed
    only if all callers obtain randomness from this seeded RNG.
    """
    if n < MC_DEFAULT_N:
        raise ValueError(f"monte_carlo requires n >= {MC_DEFAULT_N} samples; got {n}")
    if len(inputs) != len(sigmas):
        raise ValueError(
            f"inputs and sigmas must have the same length; "
            f"got {len(inputs)} and {len(sigmas)}"
        )

    rng = np.random.default_rng(MC_SEED)
    samples = []
    for _ in range(n):
        perturbed = []
        for value, sigma in zip(inputs, sigmas):
            value_arr = np.asarray(value, dtype=np.float64)
            sigma_arr = np.asarray(sigma, dtype=np.float64)
            noise = rng.standard_normal(value_arr.shape) * sigma_arr
            perturbed.append(value_arr + noise)
        samples.append(np.asarray(func(*perturbed), dtype=np.float64))

    stacked = np.stack(samples, axis=0)
    mean = stacked.mean(axis=0)
    p16 = np.percentile(stacked, 16, axis=0)
    p84 = np.percentile(stacked, 84, axis=0)
    sigma_out = (p84 - p16) / 2.0
    return mean, sigma_out
