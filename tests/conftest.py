"""Shared fixtures.

The synthetic video is the workhorse — every pipeline-level test feeds
it through one or more of the public functions and checks that the
recovered numbers match what we put in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import imageio.v2 as imageio
import numpy as np
import pytest

# Ground-truth plume parameters. Tests assert recovered values agree with
# these to a stated tolerance, so changing them changes the test bar.
PLUME_PEAK_TAU = 0.5
PLUME_SIGMA_PX = 5.0
PLUME_CENTER = (32, 32)  # (cx, cy) in a 64x64 frame
BACKGROUND_DN = 5000
N_FRAMES = 50
FRAME_HW: Tuple[int, int] = (64, 64)
BIT_DEPTH = 16


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """50-frame, 64x64, 16-bit TIFF stack with a known Gaussian plume."""
    h, w = FRAME_HW
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = PLUME_CENTER
    tau = PLUME_PEAK_TAU * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * PLUME_SIGMA_PX**2)
    )
    transmittance = np.exp(-tau)

    rng = np.random.default_rng(0)
    max_dn = 2**BIT_DEPTH - 1
    L_clean = (BACKGROUND_DN * transmittance).astype(np.float64)

    frames = []
    for _ in range(N_FRAMES):
        # Poisson shot noise via a normal approximation (large counts -> safe);
        # plus a flat 5-DN read-noise term.
        shot = rng.normal(0.0, np.sqrt(L_clean), size=L_clean.shape)
        read = rng.normal(0.0, 5.0, size=L_clean.shape)
        frame = np.clip(L_clean + shot + read, 0, max_dn).astype(np.uint16)
        frames.append(frame)

    out = tmp_path / "synthetic_plume.tif"
    imageio.mimwrite(out, frames)
    return out


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """The smallest config that calibrate() will accept."""
    return {"sensor": {"gain": 0.012, "bit_depth": BIT_DEPTH, "ner": 0.002}}


@pytest.fixture
def full_config(tmp_path: Path) -> Dict[str, Any]:
    """Config including flat-field and dark-current TIFFs in tmp_path."""
    h, w = FRAME_HW
    flat = np.ones((h, w), dtype=np.float32)
    dark = np.full((h, w), 100.0, dtype=np.float32)  # 100 DN baseline offset

    flat_path = tmp_path / "flat.tif"
    dark_path = tmp_path / "dark.tif"
    imageio.imwrite(flat_path, flat)
    imageio.imwrite(dark_path, dark)

    return {
        "sensor": {
            "gain": 0.012,
            "bit_depth": BIT_DEPTH,
            "ner": 0.002,
            "flat_field": str(flat_path),
            "dark_current": str(dark_path),
        }
    }
