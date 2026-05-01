"""Tests for SmokeSight NetCDF IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from smokesight.io import to_netcdf


@dataclass
class DummyRetrievalResult:
    """Small retrieval-like object used for IO tests."""

    tau: np.ndarray
    sigma_tau: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]


def make_dummy_retrieval_result() -> DummyRetrievalResult:
    """Create a small retrieval-like result for NetCDF tests."""
    tau = np.full((2, 4, 4), 0.5, dtype=np.float32)
    sigma_tau = np.full((2, 4, 4), 0.05, dtype=np.float32)
    mask = np.ones((2, 4, 4), dtype=bool)

    return DummyRetrievalResult(
        tau=tau,
        sigma_tau=sigma_tau,
        mask=mask,
        metadata={
            "fps": 25.0,
            "n_frames": 2,
            "height": 4,
            "width": 4,
            "institution": "SmokeSight Test",
        },
    )


def test_to_netcdf_creates_file(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    assert output_path.exists()


def test_to_netcdf_cf_conventions(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert ds.attrs["Conventions"] == "CF-1.9"
    finally:
        ds.close()


def test_to_netcdf_tau_variable_present(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert "tau" in ds
    finally:
        ds.close()


def test_to_netcdf_sigma_tau_ancillary(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert "sigma_tau" in ds
        assert ds["sigma_tau"].attrs["ancillary_variables"] == "tau"
    finally:
        ds.close()


def test_to_netcdf_opens_with_xarray(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert ds is not None
    finally:
        ds.close()


def test_xarray_accessor(tmp_path):
    result = make_dummy_retrieval_result()
    output_path = tmp_path / "output.nc"

    to_netcdf(result, output_path)

    ds = xr.open_dataset(output_path)
    try:
        assert ds.smokesight.tau is not None
    finally:
        ds.close()

def test_to_netcdf_requires_tau_and_sigma_tau(tmp_path):
    """to_netcdf should reject objects missing tau or sigma_tau."""
    output_path = tmp_path / "output.nc"

    class MissingFields:
        metadata = {}

    try:
        to_netcdf(MissingFields(), output_path)
    except ValueError as exc:
        assert "result.tau and result.sigma_tau" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing tau and sigma_tau.")


def test_to_netcdf_requires_matching_shapes(tmp_path):
    """to_netcdf should reject tau and sigma_tau with different shapes."""
    output_path = tmp_path / "output.nc"
    result = make_dummy_retrieval_result()
    result.sigma_tau = np.ones((1, 4, 4), dtype=np.float32)

    try:
        to_netcdf(result, output_path)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes.")


def test_to_netcdf_requires_3d_tau(tmp_path):
    """to_netcdf should reject tau that is not shaped as time, y, x."""
    output_path = tmp_path / "output.nc"
    result = make_dummy_retrieval_result()
    result.tau = np.ones((4, 4), dtype=np.float32)
    result.sigma_tau = np.ones((4, 4), dtype=np.float32)

    try:
        to_netcdf(result, output_path)
    except ValueError as exc:
        assert "shape (time, y, x)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-3D tau.")        
