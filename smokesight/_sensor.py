"""Sensor model: dark current, flat field, gain, spectral response."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from smokesight._types import FloatArray

_VALID_BIT_DEPTHS = (8, 12, 14, 16)
# How far the flat-field mean is allowed to drift from 1.0 before we
# emit a warning and renormalise.
_FLAT_MEAN_TOLERANCE = 1e-3

Shape = Tuple[int, int]


@dataclass
class SensorModel:
    """Sensor parameters needed to convert DN to calibrated radiance.

    flat/dark are either full (H, W) frames or a (1, 1) broadcast scalar
    when the config didn't supply a calibration image. Use
    `validate_shape` once the video resolution is known if you want to
    catch a mismatch before it bites you mid-pipeline.
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
        frame_shape: Optional[Shape] = None,
    ) -> "SensorModel":
        """Build a SensorModel from a parsed cal.yaml mapping."""
        try:
            sensor_cfg = config["sensor"]
        except KeyError as exc:
            raise ValueError("config is missing required 'sensor' section") from exc

        for required in ("gain", "bit_depth", "ner"):
            if required not in sensor_cfg:
                raise ValueError(f"config sensor.{required} is required")

        bit_depth = int(sensor_cfg["bit_depth"])
        if bit_depth not in _VALID_BIT_DEPTHS:
            raise ValueError(
                f"sensor.bit_depth must be one of {_VALID_BIT_DEPTHS}, got {bit_depth}"
            )

        flat = _read_image(sensor_cfg.get("flat_field"), "flat_field", frame_shape)
        if flat is None:
            flat = _scalar_array(1.0, frame_shape)
        else:
            flat = _renormalise_flat(flat)

        dark = _read_image(sensor_cfg.get("dark_current"), "dark_current", frame_shape)
        if dark is None:
            dark = _scalar_array(0.0, frame_shape)

        wavelengths, response = _spectral_response_from_config(sensor_cfg)

        return cls(
            dark=dark,
            flat=flat,
            gain=float(sensor_cfg["gain"]),
            spectral_response=response,
            wavelengths=wavelengths,
            bit_depth=bit_depth,
            noise_equivalent_radiance=float(sensor_cfg["ner"]),
        )

    @property
    def n_wavelengths(self) -> int:
        return int(self.spectral_response.shape[0])

    def validate_shape(self, frame_shape: Shape) -> None:
        """Raise if flat or dark have a (non-broadcast) shape that doesn't match."""
        for name, arr in (("flat", self.flat), ("dark", self.dark)):
            if arr.shape == (1, 1):
                continue
            if arr.shape != frame_shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match frame shape {frame_shape}"
                )


def _read_image(
    path: Optional[str], kind: str, frame_shape: Optional[Shape]
) -> Optional[FloatArray]:
    if path is None:
        return None
    arr = np.asarray(imageio.imread(Path(path)), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"sensor.{kind} must be a 2-D image, got shape {arr.shape}")
    if frame_shape is not None and arr.shape != frame_shape:
        raise ValueError(
            f"sensor.{kind} shape {arr.shape} does not match frame shape {frame_shape}"
        )
    return arr


def _renormalise_flat(arr: FloatArray) -> FloatArray:
    mean = float(arr.mean())
    if mean <= 0:
        raise ValueError(f"sensor.flat_field mean is {mean}; must be positive")
    if abs(mean - 1.0) <= _FLAT_MEAN_TOLERANCE:
        return arr
    warnings.warn(
        f"sensor.flat_field mean was {mean:.4f}; normalising to 1.0",
        UserWarning,
        stacklevel=3,
    )
    return arr / mean


def _scalar_array(value: float, frame_shape: Optional[Shape]) -> FloatArray:
    """A (1, 1) broadcast or a full frame of `value`, depending on what we know."""
    shape: Shape = frame_shape if frame_shape is not None else (1, 1)
    return np.full(shape, value, dtype=np.float32)


def _spectral_response_from_config(
    sensor_cfg: Mapping[str, Any],
) -> Tuple[FloatArray, FloatArray]:
    sr_cfg = sensor_cfg.get("spectral_response")
    if sr_cfg is None:
        # Single-band default. The exact wavelength is arbitrary -- callers
        # who care about wavelength must supply a spectral_response.
        return np.array([0.55], dtype=np.float32), np.array([1.0], dtype=np.float32)

    wavelengths = np.asarray(sr_cfg["wavelengths"], dtype=np.float32)
    response = np.asarray(sr_cfg["response"], dtype=np.float32)
    if wavelengths.shape != response.shape:
        raise ValueError(
            f"sensor.spectral_response.wavelengths and .response must have equal "
            f"length, got {wavelengths.shape} and {response.shape}"
        )
    if (response < 0).any() or (response > 1).any():
        raise ValueError("sensor.spectral_response.response values must be in [0, 1]")
    return wavelengths, response
