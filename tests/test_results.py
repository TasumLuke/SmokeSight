"""Tests for smokesight._results."""

from __future__ import annotations

import numpy as np
import pytest

from smokesight._atmos import IdentityAtmos
from smokesight._results import (
    BackgroundResult,
    CalibrationResult,
    DynamicsResult,
    RetrievalResult,
)
from smokesight._sensor import SensorModel


def _sensor() -> SensorModel:
    return SensorModel.from_config(
        {"sensor": {"gain": 0.012, "bit_depth": 14, "ner": 0.002}}
    )


def _calibration_result(
    t: int = 3, h: int = 4, w: int = 5, n: int = 1
) -> CalibrationResult:
    L = np.ones((t, h, w, n), dtype=np.float32)
    return CalibrationResult(
        L=L,
        sigma_L=np.full_like(L, 0.01),
        metadata={"fps": 25.0, "n_frames": t, "video_path": "fake.tif"},
        sensor=_sensor(),
        atmos=IdentityAtmos(),
    )


def test_calibration_result_repr_shows_dimensions() -> None:
    r = _calibration_result(t=10, h=64, w=80, n=3)
    assert repr(r) == "<CalibrationResult T=10 H=64 W=80 λ=3>"


def test_calibration_result_repr_handles_unexpected_shape() -> None:
    L = np.zeros((5, 5), dtype=np.float32)
    r = CalibrationResult(
        L=L,
        sigma_L=L.copy(),
        metadata={},
        sensor=_sensor(),
        atmos=IdentityAtmos(),
    )
    assert "shape=(5, 5)" in repr(r)


def test_background_result_repr_includes_method_and_confidence() -> None:
    confidence = np.full((4, 5), 0.75, dtype=np.float32)
    r = BackgroundResult(
        L0=np.ones((4, 5, 1), dtype=np.float32),
        sigma_L0=np.full((4, 5, 1), 0.01, dtype=np.float32),
        confidence=confidence,
        method="temporal_median",
        n_frames_used=100,
    )
    text = repr(r)
    assert "method='temporal_median'" in text
    assert "frames=100" in text
    assert "0.75" in text


def test_retrieval_result_reports_valid_fraction() -> None:
    mask = np.ones((2, 4, 5), dtype=bool)
    mask[0, 0, 0] = False  # 1/40 = 2.5% invalid
    r = RetrievalResult(
        tau=np.zeros((2, 4, 5), dtype=np.float32),
        sigma_tau=np.zeros((2, 4, 5), dtype=np.float32),
        mask=mask.astype(np.float32),
        metadata={},
    )
    text = repr(r)
    assert "T=2 H=4 W=5" in text
    assert "97.5%" in text
    assert "multiband=no" in text


def test_retrieval_result_flags_multiband_when_t_lambda_present() -> None:
    shape = (2, 4, 5)
    r = RetrievalResult(
        tau=np.zeros(shape, dtype=np.float32),
        sigma_tau=np.zeros(shape, dtype=np.float32),
        mask=np.ones(shape, dtype=np.float32),
        metadata={},
        T_lambda=np.zeros((*shape, 4), dtype=np.float32),
    )
    assert "multiband=yes" in repr(r)


def test_dynamics_result_formats_finite_rise() -> None:
    r = DynamicsResult(
        rise_velocity=2.34,
        sigma_rise_velocity=0.12,
        sigma_y_coeffs=np.array([0.1, 0.9], dtype=np.float32),
        sigma_z_coeffs=np.array([0.06, 0.7], dtype=np.float32),
        sigma_y_cov=np.eye(2, dtype=np.float32),
        sigma_z_cov=np.eye(2, dtype=np.float32),
        centroid_track=np.zeros((10, 2), dtype=np.float32),
        stability_class="D",
    )
    assert repr(r) == "<DynamicsResult rise=2.34±0.12 m/s stability=D>"


def test_dynamics_result_formats_nan_rise() -> None:
    r = DynamicsResult(
        rise_velocity=float("nan"),
        sigma_rise_velocity=float("nan"),
        sigma_y_coeffs=np.array([np.nan, np.nan], dtype=np.float32),
        sigma_z_coeffs=np.array([np.nan, np.nan], dtype=np.float32),
        sigma_y_cov=np.full((2, 2), np.nan, dtype=np.float32),
        sigma_z_cov=np.full((2, 2), np.nan, dtype=np.float32),
        centroid_track=np.zeros((1, 2), dtype=np.float32),
    )
    assert "rise=NaN" in repr(r)
    assert "stability=n/a" in repr(r)


def test_to_netcdf_raises_until_io_module_lands() -> None:
    """Sanity check for the Phase-4 wiring: result objects know that
    smokesight.io.to_netcdf doesn't exist yet and say so cleanly."""
    r = _calibration_result()
    with pytest.raises(NotImplementedError, match="Phase 4"):
        r.to_netcdf("/tmp/should-not-be-written.nc")
