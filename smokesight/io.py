"""NetCDF IO utilities for SmokeSight."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Union

import numpy as np
import xarray as xr

PathLike = Union[str, os.PathLike[str]]

TAU_STANDARD_NAME = "atmosphere_optical_thickness_due_to_" "aerosol"


def to_netcdf(
    result: Any,
    path: PathLike,
    *,
    complevel: int = 4,
    mode: str = "w",
) -> None:
    """Write a SmokeSight retrieval-like result object to a NetCDF file."""
    metadata = getattr(result, "metadata", {}) or {}

    tau = getattr(result, "tau", None)
    sigma_tau = getattr(result, "sigma_tau", None)

    if tau is None or sigma_tau is None:
        message = "to_netcdf requires result.tau and " "result.sigma_tau."
        raise ValueError(message)

    tau_arr = np.asarray(tau, dtype=np.float32)
    sigma_tau_arr = np.asarray(sigma_tau, dtype=np.float32)

    if tau_arr.shape != sigma_tau_arr.shape:
        message = "tau and sigma_tau must have the same shape."
        raise ValueError(message)

    if tau_arr.ndim != 3:
        message = "tau and sigma_tau must have shape " "(time, y, x)."
        raise ValueError(message)

    history = f"{datetime.now(timezone.utc).isoformat()} " "SmokeSight Python API"

    ds = xr.Dataset(
        data_vars={
            "tau": (("time", "y", "x"), tau_arr),
            "sigma_tau": (("time", "y", "x"), sigma_tau_arr),
        },
        coords={
            "time": np.arange(tau_arr.shape[0], dtype=np.float32),
            "y": np.arange(tau_arr.shape[1], dtype=np.float32),
            "x": np.arange(tau_arr.shape[2], dtype=np.float32),
        },
        attrs={
            "Conventions": "CF-1.9",
            "title": "SmokeSight retrieval output",
            "institution": metadata.get("institution", ""),
            "source": "SmokeSight v0.1.0",
            "history": history,
            "references": "https://github.com/TasumLuke/smokesight",
        },
    )

    ds["tau"].attrs.update(
        {
            "standard_name": TAU_STANDARD_NAME,
            "units": "1",
        }
    )
    ds["sigma_tau"].attrs.update(
        {
            "long_name": "1-sigma uncertainty on tau",
            "units": "1",
            "ancillary_variables": "tau",
        }
    )

    encoding = {
        variable_name: {"zlib": True, "complevel": complevel}
        for variable_name in ds.data_vars
    }

    ds.to_netcdf(path, mode=mode, encoding=encoding)


@xr.register_dataset_accessor("smokesight")
class SmokeSightAccessor:
    """Convenience accessor for SmokeSight xarray datasets."""

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @property
    def tau(self) -> xr.DataArray:
        """Return the optical depth variable."""
        return self._obj["tau"]

    @property
    def sigma_tau(self) -> xr.DataArray:
        """Return the optical depth uncertainty variable."""
        return self._obj["sigma_tau"]
