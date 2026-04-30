"""Unit tests for smokesight._geometry."""

from __future__ import annotations

import numpy as np
import pytest

from smokesight._geometry import (
    CameraGeometry,
    compute_pixel_scale,
    project_to_ground,
)


def test_compute_pixel_scale_nadir_matches_formula() -> None:
    ps = compute_pixel_scale(
        focal_length_mm=50.0,
        sensor_width_mm=12.8,
        image_width_px=1024,
        altitude_m=300.0,
        tilt_deg=0.0,
    )
    ifov = (12.8 / 1024) / 50.0
    assert ps == pytest.approx(300.0 * ifov, rel=1e-6)


def test_compute_pixel_scale_tilt_increases_scale() -> None:
    nadir = compute_pixel_scale(
        focal_length_mm=50.0,
        sensor_width_mm=12.8,
        image_width_px=1024,
        altitude_m=300.0,
        tilt_deg=0.0,
    )
    tilted = compute_pixel_scale(
        focal_length_mm=50.0,
        sensor_width_mm=12.8,
        image_width_px=1024,
        altitude_m=300.0,
        tilt_deg=45.0,
    )
    assert tilted == pytest.approx(nadir / np.cos(np.deg2rad(45.0)), rel=1e-6)


def test_compute_pixel_scale_missing_inputs_raises() -> None:
    with pytest.raises(ValueError, match="focal_length_mm"):
        compute_pixel_scale(
            focal_length_mm=None,
            sensor_width_mm=12.8,
            image_width_px=1024,
            altitude_m=300.0,
            tilt_deg=0.0,
        )


def test_from_config_no_geometry_returns_default() -> None:
    geom = CameraGeometry.from_config({})
    assert geom.pixel_scale == 1.0
    assert geom.focal_length_mm is None


def test_from_config_explicit_pixel_scale_wins() -> None:
    cfg = {
        "geometry": {
            "pixel_scale": 0.25,
            "focal_length_mm": 50.0,
            "sensor_width_mm": 12.8,
            "altitude_m": 300.0,
            "image_width": 1024,
            "tilt_deg": 0.0,
        }
    }
    geom = CameraGeometry.from_config(cfg)
    assert geom.pixel_scale == 0.25


def test_from_config_computes_when_pixel_scale_omitted() -> None:
    cfg = {
        "geometry": {
            "focal_length_mm": 50.0,
            "sensor_width_mm": 12.8,
            "altitude_m": 300.0,
            "image_width": 1024,
            "tilt_deg": 30.0,
        }
    }
    geom = CameraGeometry.from_config(cfg)
    expected = compute_pixel_scale(
        focal_length_mm=50.0,
        sensor_width_mm=12.8,
        image_width_px=1024,
        altitude_m=300.0,
        tilt_deg=30.0,
    )
    assert geom.pixel_scale == pytest.approx(expected)


def test_project_to_ground_nadir() -> None:
    geom = CameraGeometry(
        focal_length_mm=None,
        sensor_width_mm=None,
        image_width_px=None,
        altitude_m=None,
        tilt_deg=0.0,
        pixel_scale=0.5,
    )
    pixels = np.array([[0.0, 0.0], [10.0, 20.0], [-3.0, 4.0]])
    ground = project_to_ground(pixels, geom)
    np.testing.assert_allclose(ground[..., 0], pixels[..., 0] * 0.5)
    np.testing.assert_allclose(ground[..., 1], pixels[..., 1] * 0.5)


def test_project_to_ground_tilt_stretches_y() -> None:
    geom = CameraGeometry(
        focal_length_mm=None,
        sensor_width_mm=None,
        image_width_px=None,
        altitude_m=None,
        tilt_deg=60.0,
        pixel_scale=0.5,
    )
    pixels = np.array([[10.0, 10.0]])
    ground = project_to_ground(pixels, geom)
    cos_tilt = np.cos(np.deg2rad(60.0))
    np.testing.assert_allclose(ground[0, 0], 10.0 * 0.5)
    np.testing.assert_allclose(ground[0, 1], 10.0 * 0.5 / cos_tilt)


def test_project_to_ground_rejects_wrong_shape() -> None:
    geom = CameraGeometry(
        focal_length_mm=None,
        sensor_width_mm=None,
        image_width_px=None,
        altitude_m=None,
        tilt_deg=0.0,
        pixel_scale=1.0,
    )
    with pytest.raises(ValueError, match="size 2"):
        project_to_ground(np.zeros((4, 3)), geom)
