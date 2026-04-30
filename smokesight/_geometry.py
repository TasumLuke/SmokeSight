"""Camera geometry and pixel-to-ground projection.

Private module. Implements a tilted-pinhole flat-earth camera model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import numpy.typing as npt

NDArrayAny = npt.NDArray[Any]


@dataclass
class CameraGeometry:
    """Pinhole camera geometry with tilt correction.

    All length fields are floats; angles are in degrees. ``pixel_scale`` is
    in metres-per-pixel at the plume plane and may either be derived from
    the focal-length / sensor-width / altitude / tilt fields or specified
    directly via :meth:`from_config` (in which case the geometry fields are
    optional placeholders).
    """

    focal_length_mm: Optional[float]
    sensor_width_mm: Optional[float]
    image_width_px: Optional[int]
    altitude_m: Optional[float]
    tilt_deg: Optional[float]
    pixel_scale: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CameraGeometry":
        """Build CameraGeometry from a parsed cal.yaml ``geometry`` mapping.

        If ``pixel_scale`` is given it is used verbatim. Otherwise all of
        ``focal_length_mm``, ``sensor_width_mm``, ``altitude_m`` and
        (optionally) ``image_width`` and ``tilt_deg`` must be provided so
        the value can be computed.
        """
        geom_cfg = config.get("geometry") if isinstance(config, Mapping) else None
        if not geom_cfg:
            return cls(
                focal_length_mm=None,
                sensor_width_mm=None,
                image_width_px=None,
                altitude_m=None,
                tilt_deg=None,
                pixel_scale=1.0,
            )

        focal_length_mm = _opt_float(geom_cfg.get("focal_length_mm"))
        sensor_width_mm = _opt_float(geom_cfg.get("sensor_width_mm"))
        image_width_px = _opt_int(geom_cfg.get("image_width"))
        altitude_m = _opt_float(geom_cfg.get("altitude_m"))
        tilt_deg = _opt_float(geom_cfg.get("tilt_deg"))

        explicit = _opt_float(geom_cfg.get("pixel_scale"))
        if explicit is not None:
            pixel_scale = explicit
        else:
            pixel_scale = compute_pixel_scale(
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                image_width_px=image_width_px,
                altitude_m=altitude_m,
                tilt_deg=tilt_deg,
            )

        return cls(
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            image_width_px=image_width_px,
            altitude_m=altitude_m,
            tilt_deg=tilt_deg,
            pixel_scale=pixel_scale,
        )


def compute_pixel_scale(
    *,
    focal_length_mm: Optional[float],
    sensor_width_mm: Optional[float],
    image_width_px: Optional[int],
    altitude_m: Optional[float],
    tilt_deg: Optional[float],
) -> float:
    """Compute ground-plane pixel scale [m/pixel] at image centre.

    Uses a tilted flat-earth pinhole model::

        ifov_per_pixel = (sensor_width_mm / image_width_px) / focal_length_mm
        slant_range    = altitude_m / cos(tilt)
        pixel_scale    = slant_range * ifov_per_pixel

    For a nadir-pointing camera (tilt=0) this reduces to the standard
    altitude-times-IFOV formula.
    """
    if not (focal_length_mm and sensor_width_mm and altitude_m):
        raise ValueError(
            "compute_pixel_scale needs focal_length_mm, sensor_width_mm, "
            "and altitude_m (or pass pixel_scale directly in the config)"
        )
    if not image_width_px:
        raise ValueError(
            "compute_pixel_scale needs image_width to convert sensor_width "
            "to per-pixel angular resolution"
        )
    ifov = (sensor_width_mm / image_width_px) / focal_length_mm
    tilt_rad = np.deg2rad(tilt_deg if tilt_deg is not None else 0.0)
    cos_tilt = np.cos(tilt_rad)
    if cos_tilt <= 0:
        raise ValueError(f"tilt_deg={tilt_deg} produces non-positive cos(tilt)")
    slant_range = altitude_m / cos_tilt
    return float(slant_range * ifov)


def project_to_ground(pixels: NDArrayAny, geom: CameraGeometry) -> NDArrayAny:
    """Project pixel coordinates to ground-plane metres.

    ``pixels`` is an array of shape (..., 2) holding (x, y) pixel
    coordinates measured from image centre. Returns an array of the same
    shape giving ground-plane (x, y) in metres.

    This is a flat-earth, small-angle approximation: ground_x = pixel_x *
    pixel_scale, ground_y = pixel_y * pixel_scale / cos(tilt) (the tilt
    factor stretches the along-track axis on a tilted look).
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError(
            f"pixels must have last dimension of size 2, got shape {pixels.shape}"
        )
    tilt_rad = np.deg2rad(geom.tilt_deg if geom.tilt_deg is not None else 0.0)
    cos_tilt = np.cos(tilt_rad)
    if cos_tilt <= 0:
        raise ValueError(f"tilt_deg={geom.tilt_deg} produces non-positive cos(tilt)")
    ground = np.empty_like(pixels)
    ground[..., 0] = pixels[..., 0] * geom.pixel_scale
    ground[..., 1] = pixels[..., 1] * geom.pixel_scale / cos_tilt
    return ground


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)
