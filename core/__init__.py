"""
core/__init__.py

Bootstrap: insert the bundled fastQuot at the front of sys.path so that
all subsequent ``import quot`` calls resolve to the bundled copy rather
than any system-installed quot package.

fastQuot is bundled inside the repository:
    quot_saspt_workflow/
        fastQuot/
            quot/           ← the quot package lives here
        core/               ← this file lives here

GPU acceleration
----------------
The bundled quot automatically uses CUDA for detection and localisation when
PyTorch is installed and a CUDA device is present.  Set

    gpu:
      use_gpu: false

in ``settings_override.yaml`` to disable GPU and force CPU execution, or call
:func:`core.set_gpu_enabled` at runtime.
"""

import os
import sys

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../core/
_REPO_ROOT = os.path.dirname(_CORE_DIR)                          # .../quot_saspt_workflow/
_BUNDLED_FASTQUOT = os.path.join(_REPO_ROOT, "fastQuot")

# Prepend so the bundled copy always wins over any installed quot
if _BUNDLED_FASTQUOT not in sys.path:
    sys.path.insert(0, _BUNDLED_FASTQUOT)

# Fix nd2reader's hardcoded 16-bit pixel decoding so 8-bit .nd2 files
# (uiBpcInMemory == 8) don't crash the moment a frame is read. See
# core/nd2_bitdepth_patch.py for the full explanation. No-op if nd2reader
# isn't installed yet.
from .nd2_bitdepth_patch import apply as _apply_nd2_bitdepth_patch
_apply_nd2_bitdepth_patch()


def set_gpu_enabled(enabled: bool) -> None:
    """
    Enable or disable GPU acceleration for detection and localisation.

    Parameters
    ----------
    enabled : bool
        ``True`` (default) — use CUDA if a device is present.
        ``False`` — force CPU-only execution.
    """
    try:
        from quot import cuda_detect, cuda_localize
        cuda_detect._GPU_DETECT_AVAILABLE = enabled
        cuda_localize._GPU_LOCALIZE_AVAILABLE = enabled
    except Exception:
        pass  # modules not importable yet — harmless


def _apply_gpu_settings(settings: dict) -> None:
    """Called by run_local / Snakemake after settings are loaded."""
    use_gpu = settings.get("gpu", {}).get("use_gpu", True)
    if not use_gpu:
        set_gpu_enabled(False)


def _report_gpu_status() -> str:
    """Return a one-line string describing GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU enabled — {name}"
        return "GPU not available (no CUDA device found) — running on CPU"
    except ImportError:
        return "GPU not available (PyTorch not installed) — running on CPU"
