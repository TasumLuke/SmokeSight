"""Sensor model: dark current, flat field, gain, spectral response.

Private module. Not part of the public API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from smokesight._types import FloatArray

_VALID_BIT_DEPTHS = (8, 12, 14, 16)
_FLAT_MEAN_TOLERANCE = 1e-3


@dataclass
class SensorModel:
    """Calibrated sensor parameters loaded from a ``cal.yaml`` config.

    Parameters
    ----------
    dark : np.ndarray
        Dark current frame, shape (H, W) or (1, 1) for scalar broadcast,
        float32, units of DN.
    flat : np.ndarray
        Flat-field response, shape (H, W) or (1, 1), float32, normalised
        so the mean equals 1.0.
    gain : float
        DN-to-radiance scale factor, units of W m^-2 sr^-1 um^-1 per DN.
    spectral_response : np.ndarray
        Per-wavelength response, shape (N_lambda,), values in [0, 1].
    wavelengths : np.ndarray
        Centre wavelengths, shape (N_lambda,), units of micrometres.
    bit_depth : int
        Sensor ADC bit depth. Must be one of 8, 12, 14, 16.
    noise_equivalent_radiance : float
        NER, the read-noise floor in radiance units.
    """

    dark: FloatArray
    flat: FloatArray
    gain: float
    spectral_response: FloatArray
    wavelengths: FloatArray
    bit_depth: int
    noise_equivalent_radiance: float
    flat_field_relative_uncertainty: float = field(default=0.01)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> "SensorModel":
        """Build a SensorModel from a parsed cal.yaml mapping.

        Required keys: ``sensor.gain``, ``sensor.bit_depth``, ``sensor.ner``.
        Optional keys: ``sensor.flat_field`` (path), ``sensor.dark_current``
        (path), ``sensor.spectral_response`` (dict with ``wavelengths`` and
        ``response`` lists).

        ``frame_shape`` (H, W) is used to (a) construct default flat=ones /
        dark=zeros arrays when the config omits the corresponding files, and
        (b) validate that loaded files match the video resolution. When
        ``frame_shape`` is None and a file is omitted, a (1, 1) broadcast
        array is used; the caller must validate shape later.
        """
        if "sensor" not in config:
            raise ValueError("config is missing required 'sensor' section")
        sensor_cfg = config["sensor"]

        for required in ("gain", "bit_depth", "ner"):
            if required not in sensor_cfg:
                raise ValueError(f"config sensor.{required} is required")

        bit_depth = int(sensor_cfg["bit_depth"])
        if bit_depth not in _VALID_BIT_DEPTHS:
            raise ValueError(
                f"sensor.bit_depth must be one of {_VALID_BIT_DEPTHS}, got {bit_depth}"
            )

        flat = _load_or_default(
            sensor_cfg.get("flat_field"),
            frame_shape=frame_shape,
            default_value=1.0,
            kind="flat_field",
            normalize_mean=True,
        )
        dark = _load_or_default(
            sensor_cfg.get("dark_current"),
            frame_shape=frame_shape,
            default_value=0.0,
            kind="dark_current",
            normalize_mean=False,
        )

        sr_cfg = sensor_cfg.get("spectral_response")
        if sr_cfg is not None:
            wavelengths = np.asarray(sr_cfg["wavelengths"], dtype=np.float32)
            spectral_response = np.asarray(sr_cfg["response"], dtype=np.float32)
            if wavelengths.shape != spectral_response.shape:
                raise ValueError(
                    "sensor.spectral_response.wavelengths and .response "
                    f"must have equal length, got {wavelengths.shape} "
                    f"and {spectral_response.shape}"
                )
            if (spectral_response < 0).any() or (spectral_response > 1).any():
                raise ValueError(
                    "sensor.spectral_response.response values must be in [0, 1]"
                )
        else:
            wavelengths = np.array([0.55], dtype=np.float32)
            spectral_response = np.array([1.0], dtype=np.float32)

        return cls(
            dark=dark,
            flat=flat,
            gain=float(sensor_cfg["gain"]),
            spectral_response=spectral_response,
            wavelengths=wavelengths,
            bit_depth=bit_depth,
            noise_equivalent_radiance=float(sensor_cfg["ner"]),
        )

    @property
    def n_wavelengths(self) -> int:
        return int(self.spectral_response.shape[0])

    def validate_against_frame_shape(self, frame_shape: Tuple[int, int]) -> None:
        """Raise ValueError if loaded flat/dark have incompatible shape."""
        for name, arr in (("flat", self.flat), ("dark", self.dark)):
            if arr.shape == (1, 1):
                continue  # broadcast-friendly default
            if arr.shape != frame_shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match frame shape {frame_shape}"
                )


def _load_or_default(
    path: Optional[str],
    *,
    frame_shape: Optional[Tuple[int, int]],
    default_value: float,
    kind: str,
    normalize_mean: bool,
) -> FloatArray:
    if path is not None:
        arr = np.asarray(imageio.imread(Path(path)), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"sensor.{kind} must be a 2-D image, got shape {arr.shape}"
            )
        if frame_shape is not None and arr.shape != frame_shape:
            raise ValueError(
                f"sensor.{kind} shape {arr.shape} does not match "
                f"frame shape {frame_shape}"
            )
        if normalize_mean:
            mean = float(arr.mean())
            if mean <= 0:
                raise ValueError(f"sensor.{kind} mean is {mean}; must be positive")
            if abs(mean - 1.0) > _FLAT_MEAN_TOLERANCE:
                warnings.warn(
                    f"sensor.{kind} mean was {mean:.4f}; normalising to 1.0",
                    UserWarning,
                    stacklevel=2,
                )
                arr = arr / mean
        return arr

    if frame_shape is not None:
        return np.full(frame_shape, default_value, dtype=np.float32)
    return np.full((1, 1), default_value, dtype=np.float32)
