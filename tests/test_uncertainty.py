"""Tests for smokesight._uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from smokesight._atmos import IdentityAtmos
from smokesight._uncertainty import (
    centroid_uncertainty,
    gaussian_fit_uncertainty,
    monte_carlo,
    radiance_uncertainty,
    tau_uncertainty,
)


@dataclass
class _StubSensor:
    gain: float = 0.012
    noise_equivalent_radiance: float = 0.002
    flat_field_relative_uncertainty: float = 0.01


def test_radiance_uncertainty_finite_and_positive() -> None:
    L = np.array([[0.0, 1.0, 10.0], [0.5, 2.5, 100.0]], dtype=np.float32)
    sigma = radiance_uncertainty(L, _StubSensor(), IdentityAtmos())
    assert sigma.shape == L.shape
    assert sigma.dtype == np.float32
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)


def test_radiance_uncertainty_read_floor_at_zero_signal() -> None:
    L = np.zeros((4, 4), dtype=np.float32)
    sensor = _StubSensor()
    sigma = radiance_uncertainty(L, sensor, IdentityAtmos())
    np.testing.assert_allclose(sigma, sensor.noise_equivalent_radiance, rtol=1e-5)


def test_radiance_uncertainty_shot_dominates_at_high_signal() -> None:
    sensor = _StubSensor()
    L = np.array([1e4], dtype=np.float32)
    sigma = radiance_uncertainty(L, sensor, IdentityAtmos())
    expected_shot = np.sqrt(L * sensor.gain)
    expected_flat = L * sensor.flat_field_relative_uncertainty
    expected_combined = float(np.sqrt(expected_shot[0] ** 2 + expected_flat[0] ** 2))
    np.testing.assert_allclose(float(sigma[0]), expected_combined, rtol=1e-3)


def test_tau_uncertainty_matches_analytic() -> None:
    L = np.array([0.5, 0.8, 1.0])
    sigma_L = np.array([0.05, 0.04, 0.03])
    L0 = np.array([1.0, 1.0, 1.0])
    sigma_L0 = np.array([0.02, 0.02, 0.02])
    expected = np.sqrt((sigma_L / L) ** 2 + (sigma_L0 / L0) ** 2)
    out = tau_uncertainty(L, sigma_L, L0, sigma_L0)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_tau_uncertainty_nan_where_invalid() -> None:
    L = np.array([0.0, 1.0, np.nan])
    sigma_L = np.array([0.01, 0.01, 0.01])
    L0 = np.array([1.0, 0.0, 1.0])
    sigma_L0 = np.array([0.01, 0.01, 0.01])
    out = tau_uncertainty(L, sigma_L, L0, sigma_L0)
    assert np.all(np.isnan(out))


def test_centroid_uncertainty_finite_for_gaussian() -> None:
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w]
    tau = np.exp(-((xx - 15.0) ** 2 + (yy - 16.0) ** 2) / (2 * 4.0**2)).astype(
        np.float32
    )
    sigma_tau = np.full_like(tau, 0.01)
    sx, sy = centroid_uncertainty(tau, sigma_tau)
    assert np.isfinite(sx) and sx > 0
    assert np.isfinite(sy) and sy > 0


def test_centroid_uncertainty_nan_when_no_signal() -> None:
    tau = np.zeros((8, 8), dtype=np.float32)
    sigma_tau = np.full_like(tau, 0.01)
    sx, sy = centroid_uncertainty(tau, sigma_tau)
    assert np.isnan(sx) and np.isnan(sy)


def test_gaussian_fit_uncertainty_sqrt_diag() -> None:
    pcov = np.array([[4.0, 0.5], [0.5, 9.0]])
    out = gaussian_fit_uncertainty(pcov)
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_gaussian_fit_uncertainty_negative_diag_to_nan() -> None:
    pcov = np.array([[4.0, 0.0], [0.0, -1.0]])
    out = gaussian_fit_uncertainty(pcov)
    assert out[0] == pytest.approx(2.0)
    assert np.isnan(out[1])


def test_monte_carlo_is_reproducible() -> None:
    L = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.1, 0.1])
    mean1, sig1 = monte_carlo(lambda x: x * 2, [L], [sigma])
    mean2, sig2 = monte_carlo(lambda x: x * 2, [L], [sigma])
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(sig1, sig2)


def test_monte_carlo_agrees_with_analytic_for_identity() -> None:
    L = np.array([1.0])
    sigma_L = np.array([0.05])
    mean, sig = monte_carlo(lambda x: x, [L], [sigma_L], n=2000)
    assert mean[0] == pytest.approx(L[0], abs=0.01)
    assert sig[0] == pytest.approx(sigma_L[0], rel=0.15)


def test_monte_carlo_rejects_low_n() -> None:
    L = np.array([1.0])
    sigma = np.array([0.1])
    with pytest.raises(ValueError, match="n >="):
        monte_carlo(lambda x: x, [L], [sigma], n=10)


def test_monte_carlo_input_sigma_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        monte_carlo(lambda x: x, [np.array([1.0])], [])
