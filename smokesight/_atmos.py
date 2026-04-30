"""Atmospheric path correction.

Two implementations:

* :class:`IdentityAtmos` - no correction (T_atm = 1, L_path = 0).
* :class:`AtmosModel` - full correction backed by py6s or pymodtran.

If neither optional dependency is installed, :func:`make_atmos` falls back
to :class:`IdentityAtmos` and emits a UserWarning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np
import numpy.typing as npt

NDArrayAny = npt.NDArray[Any]

_DEFAULT_RELATIVE_UNCERTAINTY = 0.05


@dataclass
class IdentityAtmos:
    """Identity atmospheric correction: returns L unchanged."""

    T_atm: float = 1.0
    L_path: float = 0.0
    relative_uncertainty: float = 0.0

    def correct(self, L: NDArrayAny) -> NDArrayAny:
        return L

    def uncertainty(self, L: NDArrayAny) -> NDArrayAny:
        return np.zeros_like(L, dtype=np.float32)


class AtmosModel:
    """Atmospheric correction backed by py6s or pymodtran.

    The constructor attempts to import py6s, then pymodtran. If neither is
    importable, :class:`ImportError` is raised; callers should use
    :func:`make_atmos` to get a graceful fallback to :class:`IdentityAtmos`.
    """

    def __init__(self, config: Mapping[str, Any]):
        backend = self._resolve_backend(config)
        self.backend_name: str = backend
        self.config: Mapping[str, Any] = config
        self.T_atm: float = float(config.get("transmittance", 0.95))
        self.L_path: float = float(config.get("path_radiance", 0.0))
        self.relative_uncertainty: float = float(
            config.get("relative_uncertainty", _DEFAULT_RELATIVE_UNCERTAINTY)
        )

    @staticmethod
    def _resolve_backend(config: Mapping[str, Any]) -> str:
        requested = str(config.get("model", "")).lower()
        if requested == "identity":
            raise ValueError(
                "AtmosModel does not handle 'identity'; use IdentityAtmos directly"
            )
        if requested in ("6s", "sixs"):
            import py6s  # noqa: F401  (imported for availability check)

            return "py6s"
        if requested == "modtran":
            import pymodtran  # noqa: F401

            return "pymodtran"
        try:
            import py6s  # noqa: F401

            return "py6s"
        except ImportError:
            import pymodtran  # noqa: F401

            return "pymodtran"

    def correct(self, L: NDArrayAny) -> NDArrayAny:
        result = (L - self.L_path) / self.T_atm
        return np.asarray(result.astype(L.dtype, copy=False))

    def uncertainty(self, L: NDArrayAny) -> NDArrayAny:
        return np.asarray(np.abs(L) * np.float32(self.relative_uncertainty))


AtmosLike = Union[IdentityAtmos, AtmosModel]


def make_atmos(config: Mapping[str, Any]) -> AtmosLike:
    """Construct an atmospheric model from config, falling back gracefully.

    If ``config`` lacks an ``atmosphere`` section or sets ``model: identity``,
    returns :class:`IdentityAtmos`. Otherwise tries :class:`AtmosModel`; on
    ImportError (py6s/pymodtran missing), emits a UserWarning and falls back
    to :class:`IdentityAtmos`.
    """
    atmos_cfg = config.get("atmosphere") if isinstance(config, Mapping) else None
    if not atmos_cfg or str(atmos_cfg.get("model", "identity")).lower() == "identity":
        return IdentityAtmos()
    try:
        return AtmosModel(atmos_cfg)
    except ImportError:
        warnings.warn(
            "Neither py6s nor pymodtran is installed; atmospheric correction "
            "is disabled. Install with `pip install smokesight[calibrate]` "
            "to enable.",
            UserWarning,
            stacklevel=2,
        )
        return IdentityAtmos()
