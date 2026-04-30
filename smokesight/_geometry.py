"""Tilted-pinhole flat-earth camera model.

We don't try to do anything fancy here -- the plumes we measure live in
a single horizontal slab, the cameras look down at them at a known
altitude and tilt, and a per-pixel ground scale is enough resolution
for the dispersion fits downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from smokesight._types import FloatArray


@dataclass
class CameraGeometry:
    """Geometry parameters and the resulting per-pixel ground scale.

    The optional fields are kept around for diagnostic / serialisation
    purposes; the only thing the rest of the pipeline reads is
    ``pixel_scale`` (and ``tilt_deg`` for the along-track stretch in
    :func:`project_to_ground`).
    """

    focal_length_mm: Optional[float]
    sensor_width_mm: Optional[float]
    image_width_px: Optional[int]
    altitude_m: Optional[float]
    tilt_deg: Optional[float]
    pixel_scale: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CameraGeometry":
        """Build CameraGeometry from a parsed cal.yaml mapping.

        ``geometry.pixel_scale`` overrides the focal-length calculation if
        present. Otherwise we need focal length, sensor width, image width,
        and altitude to compute it.
        """
        geom = config.get("geometry") if isinstance(config, Mapping) else None
        if not geom:
            return cls(None, None, None, None, None, pixel_scale=1.0)

        focal_length_mm = _maybe_float(geom.get("focal_length_mm"))
        sensor_width_mm = _maybe_float(geom.get("sensor_width_mm"))
        image_width_px = _maybe_int(geom.get("image_width"))
        altitude_m = _maybe_float(geom.get("altitude_m"))
        tilt_deg = _maybe_float(geom.get("tilt_deg"))

        explicit = _maybe_float(geom.get("pixel_scale"))
        pixel_scale = (
            explicit
            if explicit is not None
            else compute_pixel_scale(
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                image_width_px=image_width_px,
                altitude_m=altitude_m,
                tilt_deg=tilt_deg,
            )
        )
        return cls(
            focal_length_mm,
            sensor_width_mm,
            image_width_px,
            altitude_m,
            tilt_deg,
            pixel_scale,
        )


def compute_pixel_scale(
    *,
    focal_length_mm: Optional[float],
    sensor_width_mm: Optional[float],
    image_width_px: Optional[int],
    altitude_m: Optional[float],
    tilt_deg: Optional[float],
) -> float:
    """Ground-plane pixel scale at image centre, in metres per pixel.

        ifov_per_pixel = (sensor_width_mm / image_width_px) / focal_length_mm
        slant_range    = altitude_m / cos(tilt)
        pixel_scale    = slant_range * ifov_per_pixel

    Reduces to altitude * IFOV at nadir (tilt=0).
    """
    missing = [
        name
        for name, value in (
            ("focal_length_mm", focal_length_mm),
            ("sensor_width_mm", sensor_width_mm),
            ("image_width", image_width_px),
            ("altitude_m", altitude_m),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "compute_pixel_scale needs " + ", ".join(missing) + " "
            "(or pass pixel_scale directly in the config)"
        )
    assert focal_length_mm and sensor_width_mm and image_width_px and altitude_m  # mypy

    ifov = (sensor_width_mm / image_width_px) / focal_length_mm
    cos_tilt = np.cos(np.deg2rad(tilt_deg or 0.0))
    return float((altitude_m / cos_tilt) * ifov)


def project_to_ground(pixels: FloatArray, geom: CameraGeometry) -> FloatArray:
    """Pixel offsets from image centre -> ground-plane metres.

    Last axis must be size 2 (x, y). Returns an array of the same shape.
    A non-zero tilt stretches the y (along-track) component by 1/cos(tilt).
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError(
            f"pixels must have last dimension of size 2, got shape {pixels.shape}"
        )
    cos_tilt = np.cos(np.deg2rad(geom.tilt_deg or 0.0))

    ground = np.empty_like(pixels)
    ground[..., 0] = pixels[..., 0] * geom.pixel_scale
    ground[..., 1] = pixels[..., 1] * geom.pixel_scale / cos_tilt
    return ground


def _maybe_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _maybe_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)
