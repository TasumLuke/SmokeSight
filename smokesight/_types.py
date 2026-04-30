"""Shared type aliases.

Kept tiny on purpose: the only thing the private modules really agree
on is the numpy array shape. If we start adding more aliases here
they should probably live in a typed Protocol next to the consumer.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

FloatArray = npt.NDArray[Any]
