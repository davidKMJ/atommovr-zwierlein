"""Compatibility shim — offline camera types live in ``camera.py``.

Prefer::

    from awg_controller.src.camera import GaussianCameraConfig, OfflineArrayCamera
"""

from awg_controller.src.camera import (  # noqa: F401
    GaussianCameraConfig,
    ImageGenerator,
    OfflineArrayCamera,
)

__all__ = ["GaussianCameraConfig", "ImageGenerator", "OfflineArrayCamera"]
