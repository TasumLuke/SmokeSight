"""Atmospheric path correction.

Two flavours: IdentityAtmos (no-op, used when nothing is configured or
when the optional radiative-transfer extras aren't installed) and
AtmosModel (proper correction via py6s or pymodtran). Use make_atmos()
to pick the right one from a parsed cal.yaml — it handles the import
fallback so the rest of the pipeline doesn't have to care.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np

from smokesight._types import FloatArray

_FALLBACK_RELATIVE_SIGMA = 0.05


@dataclass
class IdentityAtmos:
    """No-op: T_atm=1, L_path=0, sigma=0."""

    T_atm: float = 1.0
    L_path: float = 0.0

    def correct(self, L: FloatArray) -> FloatArray:
        return L

    def uncertainty(self, L: FloatArray) -> FloatArray:
        return np.zeros_like(L, dtype=np.float32)


class AtmosModel:
    """Correction backed by py6s or pymodtran.

    The constructor probes the requested backend (or auto-picks if model
    is unset) and raises ImportError if nothing is available. Callers
    that want graceful fallback should go through ``make_atmos``.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.backend = _probe_backend(str(config.get("model", "")).lower())
        self.config = config
        self.T_atm = float(config.get("transmittance", 0.95))
        self.L_path = float(config.get("path_radiance", 0.0))
        self.relative_sigma = float(
            config.get("relative_uncertainty", _FALLBACK_RELATIVE_SIGMA)
        )

    def correct(self, L: FloatArray) -> FloatArray:
        out = (L - self.L_path) / self.T_atm
        return np.asarray(out.astype(L.dtype, copy=False))

    def uncertainty(self, L: FloatArray) -> FloatArray:
        return np.asarray(np.abs(L) * np.float32(self.relative_sigma))


AtmosLike = Union[IdentityAtmos, AtmosModel]

_INSTALL_HINT = (
    "Neither py6s nor pymodtran is installed; atmospheric correction is "
    "disabled. Install with `pip install smokesight[calibrate]` to enable."
)


def make_atmos(config: Mapping[str, Any]) -> AtmosLike:
    """Pick an atmosphere implementation from a parsed cal.yaml.

    No 'atmosphere' section, or model='identity' -> IdentityAtmos.
    Anything else -> AtmosModel, falling back to IdentityAtmos with a
    UserWarning if the requested backend isn't installed.
    """
    atmos_cfg = config.get("atmosphere") if isinstance(config, Mapping) else None
    if not atmos_cfg:
        return IdentityAtmos()
    if str(atmos_cfg.get("model", "identity")).lower() == "identity":
        return IdentityAtmos()
    try:
        return AtmosModel(atmos_cfg)
    except ImportError:
        warnings.warn(_INSTALL_HINT, UserWarning, stacklevel=2)
        return IdentityAtmos()


def _probe_backend(requested: str) -> str:
    """Return the name of the backend we'll use, or raise ImportError."""
    if requested == "identity":
        # Caller bug: should have used IdentityAtmos directly.
        raise ValueError("AtmosModel can't run the identity backend")

    if requested in ("6s", "sixs"):
        import py6s  # noqa: F401

        return "py6s"
    if requested == "modtran":
        import pymodtran  # noqa: F401

        return "pymodtran"

    # Auto-pick: prefer py6s, fall back to pymodtran, otherwise let
    # ImportError propagate so make_atmos can warn and fall back.
    try:
        import py6s  # noqa: F401

        return "py6s"
    except ImportError:
        import pymodtran  # noqa: F401

        return "pymodtran"
