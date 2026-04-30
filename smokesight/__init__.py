"""SmokeSight: radiometric plume measurement from EO/IR video.

Public API (wired up once modules are implemented in Phase 4):

    from smokesight import calibrate, background, retrieve, dynamics, io

For now only ``__version__`` is exposed so ``pip install -e .`` and
``mypy smokesight`` pass against empty module stubs.
"""

__version__: str = "0.1.0"
__all__ = ["__version__"]
