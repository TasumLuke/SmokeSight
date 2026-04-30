"""Tests for smokesight._atmos."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from smokesight._atmos import AtmosModel, IdentityAtmos, make_atmos


def _atmos_or_skip(cfg: Mapping[str, Any]) -> AtmosModel:
    """AtmosModel(cfg), or pytest.skip if the backend isn't installed."""
    try:
        return AtmosModel(cfg)
    except ImportError:
        pytest.skip("py6s/pymodtran not installed (smokesight[calibrate])")


def test_identity_correct_returns_input_unchanged() -> None:
    L = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(IdentityAtmos().correct(L), L)


def test_identity_uncertainty_is_zero() -> None:
    L = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sigma = IdentityAtmos().uncertainty(L)
    assert sigma.shape == L.shape
    np.testing.assert_array_equal(sigma, np.zeros_like(L))


def test_atmos_model_correct_math() -> None:
    am = _atmos_or_skip(
        {
            "model": "modtran",
            "transmittance": 0.8,
            "path_radiance": 0.5,
            "relative_uncertainty": 0.05,
        }
    )
    L = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(am.correct(L), (L - 0.5) / 0.8, rtol=1e-6)


def test_atmos_model_uncertainty_proportional_to_signal() -> None:
    am = _atmos_or_skip(
        {
            "model": "modtran",
            "transmittance": 0.9,
            "path_radiance": 0.0,
            "relative_uncertainty": 0.07,
        }
    )
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
