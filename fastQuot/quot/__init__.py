#!/usr/bin/env python
"""
__init__.py

Exposes the core tracking API.  All imports are guarded so that missing
optional dependencies (GUI toolkit, nd2reader/pims, etc.) do not prevent
the package from loading in a headless pipeline environment.
"""

# GUI dependencies — only needed for the interactive viewer.
try:
    import PySide6       # noqa: F401
    import pyqtgraph     # noqa: F401
except (ImportError, RuntimeError):
    pass

# Core tracking functions.  ChunkFilter pulls in nd2reader → pims; guard it.
try:
    from .core import (
        localize_file,
        track_file,
        track_directory,
        retrack_file,
        retrack_files,
    )
    from .read import ImageReader, read_config
    from .chunkFilter import ChunkFilter
except (ImportError, RuntimeError):
    pass

# Find spots (no heavy optional dependencies)
from .findSpots import detect

# Localize spots to subpixel resolution
from .subpixel import localize, localize_frame

# Reconnect spots into trajectories
from .track import track
