"""Unit tests for smokesight._atmos."""

from __future__ import annotations

import numpy as np
import pytest

from smokesight._atmos import AtmosModel, IdentityAtmos, make_atmos


def test_identity_correct_returns_input_unchanged() -> None:
    L = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = IdentityAtmos().correct(L)
    np.testing.assert_array_equal(out, L)


def test_identity_uncertainty_is_zero() -> None:
    L = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sigma = IdentityAtmos().uncertainty(L)
    assert sigma.shape == L.shape
    np.testing.assert_array_equal(sigma, np.zeros_like(L))


def test_atmos_model_correct_math() -> None:
    cfg = {
        "model": "modtran",
        "transmittance": 0.8,
        "path_radiance": 0.5,
        "relative_uncertainty": 0.05,
    }
    try:
        am = AtmosModel(cfg)
    except ImportError:
        pytest.skip("pymodtran not installed")
    L = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = (L - 0.5) / 0.8
    np.testing.assert_allclose(am.correct(L), expected, rtol=1e-6)


def test_atmos_model_uncertainty_proportional_to_signal() -> None:
    cfg = {
        "model": "modtran",
        "transmittance": 0.9,
        "path_radiance": 0.0,
        "relative_uncertainty": 0.07,
    }
    try:
        am = AtmosModel(cfg)
    except ImportError:
        pytest.skip("pymodtran not installed")
    L = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(am.uncertainty(L), L * 0.07, rtol=1e-6)


def test_atmos_model_rejects_identity_backend() -> None:
    with pytest.raises(ValueError, match="identity"):
        AtmosModel({"model": "identity"})


def test_make_atmos_no_section_returns_identity() -> None:
    assert isinstance(make_atmos({}), IdentityAtmos)


def test_make_atmos_explicit_identity() -> None:
    assert isinstance(make_atmos({"atmosphere": {"model": "identity"}}), IdentityAtmos)


def test_make_atmos_falls_back_when_backend_missing() -> None:
    cfg = {
        "atmosphere": {
            "model": "6s",
            "transmittance": 0.8,
            "path_radiance": 0.0,
        }
    }
    try:
        import py6s  # noqa: F401
    except ImportError:
        with pytest.warns(UserWarning, match="atmospheric correction is disabled"):
            atmos = make_atmos(cfg)
        assert isinstance(atmos, IdentityAtmos)
    else:  # pragma: no cover - exercised only when py6s is installed
        atmos = make_atmos(cfg)
        assert isinstance(atmos, AtmosModel)
