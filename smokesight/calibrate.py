"""DN-to-radiance calibration pipeline.

Reads a video frame-by-frame, applies the per-frame radiometry chain
(dark, flat, gain, spectral response, atmospheric correction) and
returns a (T, H, W, N_lambda) calibrated radiance cube plus a matching
sigma_L cube from the sensor noise model.

The flat-field correction has to happen *before* the gain multiply.
The two operations commute mathematically when both are scalar/uniform,
but a non-uniform flat with a non-zero dark current produces a 1-3%
bias if you reorder them. The pipeline below is written to make the
order obvious.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import yaml
from tqdm import tqdm

from smokesight._atmos import AtmosLike, make_atmos
from smokesight._results import CalibrationResult
from smokesight._sensor import SensorModel
from smokesight._types import FloatArray
from smokesight._uncertainty import radiance_uncertainty

PathLike = Union[str, "os.PathLike[str]"]
ConfigLike = Union[PathLike, Mapping[str, Any]]


class SmokeSightCalibrationError(RuntimeError):
    """Raised when the sensor or atmosphere model can't be initialised."""


def calibrate(
    video_path: PathLike,
    config: ConfigLike,
    *,
    frame_range: Optional[Tuple[int, int]] = None,
    n_workers: int = 1,
    progress: bool = True,
) -> CalibrationResult:
    """Run the calibration pipeline over a video.

    Parameters
    ----------
    video_path
        Anything imageio can open: MP4, AVI, multi-page TIFF, raw bin.
    config
        The parsed cal.yaml dict, or a path to load.
    frame_range
        Inclusive (start, stop). ``None`` -> all frames.
    n_workers
        Reserved. Frame I/O is single-threaded today; the parameter is
        here so future versions can add multithreaded reads without an
        API break.
    progress
        Show a tqdm bar.
    """
    del n_workers  # see docstring; tqdm parallelism not implemented
    cfg = _load_config(config)
    config_label = "<dict>" if isinstance(config, Mapping) else str(config)

    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta["fps"]) if "fps" in meta else 0.0

        first = reader.get_data(0)
        h, w = first.shape[:2]
        bit_depth = _bit_depth_from_dtype(first.dtype)

        try:
            sensor = SensorModel.from_config(cfg, frame_shape=(h, w))
        except ValueError:
            raise
        except Exception as exc:  # imageio / file-system failures wrap the cause
            raise SmokeSightCalibrationError(
                f"failed to build sensor model: {exc}"
            ) from exc
        sensor.validate_shape((h, w))

        try:
            atmos = make_atmos(cfg)
        except Exception as exc:
            raise SmokeSightCalibrationError(
                f"failed to build atmospheric model: {exc}"
            ) from exc

        n_total = _count_frames(reader)
        start, stop = _resolve_range(frame_range, n_total)
        n_frames = stop - start + 1

        cube = _calibrate_frames(
            reader=reader,
            start=start,
            stop=stop,
            sensor=sensor,
            atmos=atmos,
            progress=progress,
        )
    finally:
        reader.close()

    sigma_L = radiance_uncertainty(cube, sensor, atmos)

    return CalibrationResult(
        L=cube,
        sigma_L=sigma_L,
        metadata={
            "fps": fps,
            "n_frames": n_frames,
            "height": h,
            "width": w,
            "bit_depth": bit_depth,
            "video_path": str(video_path),
            "config_path": config_label,
            "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        sensor=sensor,
        atmos=atmos,
    )


def _load_config(config: ConfigLike) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    with open(Path(config), "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config file {config!s} did not parse to a mapping")
    return loaded


def _bit_depth_from_dtype(dtype: np.dtype[Any]) -> int:
    if dtype == np.uint8:
        return 8
    if dtype == np.uint16:
        # 14 / 16 are both stored in uint16 containers; lab gives the real number
        # via cal.yaml -- the metadata field here is the *container* width.
        return 16
    if dtype == np.uint32:
        return 32
    if dtype == np.float32:
        return 32
    return int(np.iinfo(dtype).bits) if np.issubdtype(dtype, np.integer) else 32


def _count_frames(reader: Any) -> Optional[int]:
    try:
        n = reader.count_frames()
    except (AttributeError, RuntimeError):
        try:
            n = reader.get_length()
        except (AttributeError, RuntimeError):
            return None
    if n in (None, float("inf")):
        return None
    return int(n)


def _resolve_range(
    frame_range: Optional[Tuple[int, int]], n_total: Optional[int]
) -> Tuple[int, int]:
    if frame_range is not None:
        start, stop = frame_range
        if start < 0 or stop < start:
            raise ValueError(
                f"frame_range must be (start <= stop, both >= 0); got {frame_range}"
            )
        return start, stop
    if n_total is None:
        raise SmokeSightCalibrationError(
            "video has unknown frame count; pass frame_range explicitly"
        )
    return 0, n_total - 1


def _calibrate_frames(
    *,
    reader: Any,
    start: int,
    stop: int,
    sensor: SensorModel,
    atmos: AtmosLike,
    progress: bool,
) -> FloatArray:
    n_frames = stop - start + 1
    src_indices: Iterable[int] = range(start, stop + 1)
    if progress:
        src_indices = tqdm(src_indices, desc="calibrate", unit="frame")

    cube: Optional[FloatArray] = None  # allocated lazily once we know N_lambda
    for out_idx, src_idx in enumerate(src_indices):
        frame = np.asarray(reader.get_data(src_idx), dtype=np.float32)
        if frame.ndim == 3:  # collapse colour channels -- treat as panchromatic
            frame = frame.mean(axis=-1)
        L_cal = _calibrate_one_frame(frame, sensor, atmos)
        if cube is None:
            cube = np.empty(
                (n_frames, *L_cal.shape), dtype=np.float32
            )  # (T, H, W, N_lambda)
        cube[out_idx] = L_cal

    assert cube is not None  # n_frames >= 1 by construction
    return cube


def _calibrate_one_frame(
    frame: FloatArray, sensor: SensorModel, atmos: AtmosLike
) -> FloatArray:
    # Order is load-bearing: subtract dark, then divide by flat, THEN gain.
    L_raw = (frame - sensor.dark) / sensor.flat
    L_abs = L_raw * sensor.gain
    # broadcast spectral response over a new wavelength axis
    L_cal = L_abs[..., np.newaxis] * sensor.spectral_response  # (H, W, N_lambda)
    L_surface = atmos.correct(L_cal)
    return np.asarray(L_surface, dtype=np.float32)
