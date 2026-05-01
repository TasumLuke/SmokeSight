"""Tests for smokesight.calibrate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import pytest

from smokesight._results import CalibrationResult
from smokesight.calibrate import calibrate

# fixtures from conftest.py: synthetic_video, minimal_config, full_config


def test_returns_correct_shape(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    assert isinstance(cal, CalibrationResult)
    # 50 frames, 64x64, single-band default => N_lambda = 1
    assert cal.L.shape == (50, 64, 64, 1)
    assert cal.L.dtype == np.float32


def test_sigma_L_is_positive_finite(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    assert np.all(np.isfinite(cal.sigma_L))
    assert np.all(cal.sigma_L > 0)
    assert cal.sigma_L.shape == cal.L.shape


def test_dark_subtraction_lowers_radiance(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    full_config: Dict[str, Any],
) -> None:
    """A non-zero dark in full_config should yield lower L than the no-dark case."""
    cal_no_dark = calibrate(synthetic_video, minimal_config, progress=False)
    cal_with_dark = calibrate(synthetic_video, full_config, progress=False)
    expected_drop = full_config["sensor"]["gain"] * 100.0  # dark = 100 DN
    delta = cal_no_dark.L.mean() - cal_with_dark.L.mean()
    np.testing.assert_allclose(float(delta), expected_drop, rtol=1e-3)


def test_flat_field_applied_before_gain(
    synthetic_video: Path,
    minimal_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Recovered radiance at a hot flat-field pixel must equal F / flat * gain
    -- catches the silent bias from applying gain before the flat divide."""
    flat = np.ones((64, 64), dtype=np.float32)
    flat[10, 10] = 2.0  # twice as sensitive at this pixel
    flat = flat / float(flat.mean())  # mean=1.0 so the loader doesn't rescale
    flat_pixel = float(flat[10, 10])

    flat_path = tmp_path / "flat.tif"
    imageio.imwrite(flat_path, flat)

    cfg = dict(minimal_config)
    cfg["sensor"] = {**minimal_config["sensor"], "flat_field": str(flat_path)}

    cal = calibrate(synthetic_video, cfg, progress=False)

    reader = imageio.get_reader(str(synthetic_video))
    try:
        raw = np.asarray(reader.get_data(0), dtype=np.float32)
    finally:
        reader.close()

    expected = float(raw[10, 10]) / flat_pixel * cfg["sensor"]["gain"]
    np.testing.assert_allclose(float(cal.L[0, 10, 10, 0]), expected, rtol=1e-5)


def test_missing_required_field_raises(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cfg = {"sensor": {k: v for k, v in minimal_config["sensor"].items() if k != "gain"}}
    with pytest.raises(ValueError, match="gain"):
        calibrate(synthetic_video, cfg, progress=False)


def test_frame_range_inclusive(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = calibrate(
        synthetic_video, minimal_config, frame_range=(10, 20), progress=False
    )
    assert cal.L.shape[0] == 11  # inclusive on both ends
    assert cal.metadata["n_frames"] == 11


def test_metadata_recorded(
    synthetic_video: Path, minimal_config: Dict[str, Any]
) -> None:
    cal = calibrate(synthetic_video, minimal_config, progress=False)
    assert cal.metadata["height"] == 64
    assert cal.metadata["width"] == 64
    assert cal.metadata["video_path"].endswith("synthetic_plume.tif")
    assert "calibration_timestamp" in cal.metadata
