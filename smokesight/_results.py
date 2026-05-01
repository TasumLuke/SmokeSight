"""Result dataclasses returned by the public pipeline.

One private module so that everyone can import these without dragging
in `smokesight.io`. The `to_netcdf` convenience methods lazy-import io
on call — they'll be wired up properly once io.to_netcdf is implemented
in Phase 4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

from smokesight._atmos import AtmosLike
from smokesight._sensor import SensorModel
from smokesight._types import FloatArray

PathLike = Union[str, "os.PathLike[str]"]


class _Result:
    """Mixin: shared `to_netcdf` so each dataclass doesn't repeat itself."""

    def to_netcdf(self, path: PathLike) -> None:
        """Serialise to CF-1.9 NetCDF4. Delegates to `smokesight.io.to_netcdf`."""
        from smokesight import io as _io

        # io.to_netcdf is wired up in Phase 4; until then this surfaces a
        # clear error rather than the bare AttributeError you'd get from
        # an empty module.
        write = getattr(_io, "to_netcdf", None)
        if write is None:
            raise NotImplementedError(
                "smokesight.io.to_netcdf is not yet implemented (Phase 4)"
            )
        write(self, path)


@dataclass(repr=False)
class CalibrationResult(_Result):
    """Calibrated radiance cube + per-pixel sigma_L + provenance.

    L has shape (T, H, W, N_lambda). sigma_L is the same shape.
    metadata carries the bookkeeping fields (fps, frame count, source
    paths, calibration timestamp). sensor and atmos are kept around so
    downstream steps can recover the noise model that produced sigma_L.
    """

    L: FloatArray
    sigma_L: FloatArray
    metadata: Dict[str, Any]
    sensor: SensorModel
    atmos: AtmosLike

    def __repr__(self) -> str:
        if self.L.ndim == 4:
            t, h, w, n = self.L.shape
            return f"<CalibrationResult T={t} H={h} W={w} λ={n}>"
        return f"<CalibrationResult shape={self.L.shape}>"


@dataclass(repr=False)
class BackgroundResult(_Result):
    """Background radiance plate, its uncertainty, and a confidence map.

    L0 is shape (H, W, N_lambda); confidence is the (H, W) gate that
    every downstream tau pixel passes through.
    """

    L0: FloatArray
    sigma_L0: FloatArray
    confidence: FloatArray
    method: str
    n_frames_used: int

    def __repr__(self) -> str:
        h, w = self.confidence.shape
        mean_c = float(np.nanmean(self.confidence))
        return (
            f"<BackgroundResult method={self.method!r} H={h} W={w} "
            f"frames={self.n_frames_used} mean_confidence={mean_c:.2f}>"
        )


@dataclass(repr=False)
class RetrievalResult(_Result):
    """Optical depth tau, its uncertainty, and optional spectral / column outputs.

    tau and sigma_tau are NaN wherever the pixel was masked (low
    confidence, ratio out of physical range, tau > tau_max). T_lambda
    and N/sigma_N are populated only for multi-band input or when a
    species cross-section was provided.
    """

    tau: FloatArray
    sigma_tau: FloatArray
    mask: FloatArray  # bool (T, H, W) -- True where the pixel is valid
    metadata: Dict[str, Any]
    T_lambda: Optional[FloatArray] = None
    N: Optional[FloatArray] = None
    sigma_N: Optional[FloatArray] = None

    def __repr__(self) -> str:
        if self.mask.size:
            valid_pct = 100.0 * float(self.mask.sum()) / float(self.mask.size)
        else:
            valid_pct = float("nan")
        t, h, w = self.tau.shape
        return (
            f"<RetrievalResult T={t} H={h} W={w} valid={valid_pct:.1f}% "
            f"multiband={'yes' if self.T_lambda is not None else 'no'}>"
        )


@dataclass(repr=False)
class DynamicsResult(_Result):
    """Plume rise velocity and Pasquill-Gifford dispersion fits.

    rise_velocity / sigma_rise_velocity are NaN if the centroid track
    was too short or noisy for a meaningful linear fit. sigma_y_coeffs
    is (a, b) in sigma_y = a * x^b; sigma_z_coeffs likewise.
    """

    rise_velocity: float
    sigma_rise_velocity: float
    sigma_y_coeffs: FloatArray  # shape (2,)
    sigma_z_coeffs: FloatArray  # shape (2,)
    sigma_y_cov: FloatArray  # shape (2, 2)
    sigma_z_cov: FloatArray  # shape (2, 2)
    centroid_track: FloatArray  # shape (T, 2)
    stability_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if np.isnan(self.rise_velocity):
            rise = "NaN"
        else:
            rise = f"{self.rise_velocity:.2f}±{self.sigma_rise_velocity:.2f} m/s"
        klass = self.stability_class if self.stability_class else "n/a"
        return f"<DynamicsResult rise={rise} stability={klass}>"
