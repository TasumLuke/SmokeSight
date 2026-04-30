"""Unit tests for smokesight._sensor."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from smokesight._sensor import SensorModel


def _minimal_config() -> dict:
    return {"sensor": {"gain": 0.012, "bit_depth": 14, "ner": 0.002}}


def test_from_config_minimal_no_frame_shape() -> None:
    sm = SensorModel.from_config(_minimal_config())
    assert sm.gain == pytest.approx(0.012)
    assert sm.bit_depth == 14
    assert sm.noise_equivalent_radiance == pytest.approx(0.002)
    assert sm.flat.shape == (1, 1)
    assert sm.dark.shape == (1, 1)
    assert sm.flat[0, 0] == pytest.approx(1.0)
    assert sm.dark[0, 0] == pytest.approx(0.0)
    assert sm.spectral_response.shape == (1,)


def test_from_config_minimal_with_frame_shape() -> None:
    sm = SensorModel.from_config(_minimal_config(), frame_shape=(64, 80))
    assert sm.flat.shape == (64, 80)
    assert sm.dark.shape == (64, 80)
    np.testing.assert_array_equal(sm.flat, np.ones((64, 80), dtype=np.float32))
    np.testing.assert_array_equal(sm.dark, np.zeros((64, 80), dtype=np.float32))


@pytest.mark.parametrize("missing", ["gain", "bit_depth", "ner"])
def test_missing_required_field_raises(missing: str) -> None:
    cfg = _minimal_config()
    del cfg["sensor"][missing]
    with pytest.raises(ValueError, match=missing):
        SensorModel.from_config(cfg)


def test_missing_sensor_section_raises() -> None:
    with pytest.raises(ValueError, match="sensor"):
        SensorModel.from_config({})


def test_invalid_bit_depth_raises() -> None:
    cfg = _minimal_config()
    cfg["sensor"]["bit_depth"] = 10
    with pytest.raises(ValueError, match="bit_depth"):
        SensorModel.from_config(cfg)


def test_flat_field_normalisation_warning(tmp_path: Path) -> None:
    flat = np.ones((16, 16), dtype=np.float32) * 2.0
    flat_path = tmp_path / "flat.tif"
    imageio.imwrite(flat_path, flat)

    cfg = _minimal_config()
    cfg["sensor"]["flat_field"] = str(flat_path)
    with pytest.warns(UserWarning, match="normalising"):
        sm = SensorModel.from_config(cfg, frame_shape=(16, 16))
    assert sm.flat.mean() == pytest.approx(1.0, rel=1e-5)


def test_flat_field_shape_mismatch_raises(tmp_path: Path) -> None:
    flat = np.ones((16, 16), dtype=np.float32)
    flat_path = tmp_path / "flat.tif"
    imageio.imwrite(flat_path, flat)

    cfg = _minimal_config()
    cfg["sensor"]["flat_field"] = str(flat_path)
    with pytest.raises(ValueError, match="flat_field shape"):
        SensorModel.from_config(cfg, frame_shape=(32, 32))


def test_spectral_response_round_trip() -> None:
    cfg = _minimal_config()
    cfg["sensor"]["spectral_response"] = {
        "wavelengths": [3.5, 4.0, 4.5, 5.0],
        "response": [0.3, 0.8, 0.9, 0.4],
    }
    sm = SensorModel.from_config(cfg)
    assert sm.n_wavelengths == 4
    np.testing.assert_allclose(sm.wavelengths, [3.5, 4.0, 4.5, 5.0])
    np.testing.assert_allclose(sm.spectral_response, [0.3, 0.8, 0.9, 0.4])


def test_spectral_response_length_mismatch_raises() -> None:
    cfg = _minimal_config()
    cfg["sensor"]["spectral_response"] = {
        "wavelengths": [3.5, 4.0, 4.5],
        "response": [0.3, 0.8],
    }
    with pytest.raises(ValueError, match="equal length"):
        SensorModel.from_config(cfg)


def test_spectral_response_out_of_range_raises() -> None:
    cfg = _minimal_config()
    cfg["sensor"]["spectral_response"] = {
        "wavelengths": [4.0, 4.5],
        "response": [0.5, 1.5],
    }
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SensorModel.from_config(cfg)


def test_validate_shape_passes_with_broadcast() -> None:
    sm = SensorModel.from_config(_minimal_config())
    sm.validate_shape((64, 64))


def test_validate_shape_detects_mismatch() -> None:
    sm = SensorModel.from_config(_minimal_config(), frame_shape=(32, 32))
    with pytest.raises(ValueError, match="does not match"):
        sm.validate_shape((64, 64))
