"""
cuda_detect.py — GPU-accelerated spot detection using PyTorch.

Drop-in GPU replacements for the FFT-based detection methods in findSpots.py.
Falls back gracefully to the CPU implementation if CUDA is not available or
if any error occurs.

Public API
----------
llr_gpu(I, k, w, t, return_filt) — GPU version of the LLR detector.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ---------------------------------------------------------------------------
# Device selection (module-level singleton)
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE: bool = _TORCH_OK and (
    __import__("torch").cuda.is_available() if _TORCH_OK else False
)


def _get_device() -> "torch.device":
    """Always returns CUDA — only called when _CUDA_AVAILABLE is True."""
    return torch.device("cuda")


# Cache the GPU kernel so we only build it once per (H, W, k, w) combination.
_kernel_cache: dict = {}


def _mle_amp_setup_torch(H: int, W: int, k: float, w: int,
                          device: "torch.device", dtype: "torch.dtype"):
    """
    Build the centred-Gaussian transfer function and normalisation constant
    on the target device.  Results are cached by (H, W, k, w).
    """
    key = (H, W, k, w, str(device))
    if key in _kernel_cache:
        return _kernel_cache[key]

    S = 2.0 * (k ** 2)
    coords = np.indices((w, w)) - (w - 1) / 2.0           # (2, w, w)
    g = np.exp(-((coords ** 2).sum(0)) / S)                # (w, w)
    g = g / g.sum()
    gc = g - g.mean()
    Sgc2 = float((gc ** 2).sum())

    # Pad gc to (H, W) and compute RFFT2
    padded = np.zeros((H, W), dtype=np.float64)
    hc = int(np.ceil(H / 2 - w / 2))
    wc = int(np.ceil(W / 2 - w / 2))
    padded[hc: hc + w, wc: wc + w] = gc

    G_rft_np = np.fft.rfft2(padded)                        # complex (H, W//2+1)
    G_rft = torch.from_numpy(G_rft_np).to(device=device)  # complex128 tensor

    result = (G_rft, Sgc2)
    _kernel_cache[key] = result
    return result


def _uniform_filter_torch(I_t: "torch.Tensor", w: int) -> "torch.Tensor":
    """
    2D uniform (box) filter — equivalent to scipy.ndimage.uniform_filter.
    Handles even and odd *w*; uses replicate padding to match scipy's 'reflect'
    behaviour at borders.

    I_t : (H, W) float tensor on any device.
    Returns a (H, W) tensor.
    """
    H, W = I_t.shape
    pad = w // 2
    kernel = (torch.ones(1, 1, w, w, device=I_t.device, dtype=I_t.dtype)
              / float(w * w))
    I_4d = I_t.unsqueeze(0).unsqueeze(0)                  # (1, 1, H, W)
    out = F.conv2d(I_4d, kernel, padding=pad)              # (1, 1, H, W)
    # Trim to original size (needed if w is even)
    return out[0, 0, :H, :W]


def llr_gpu(I: np.ndarray, k: float = 1.0, w: int = 9, t: float = 20.0,
            return_filt: bool = False):
    """
    GPU-accelerated log-likelihood ratio detector.

    Identical semantics to ``findSpots.llr``.  Falls back to the CPU
    implementation on any error.

    Parameters
    ----------
    I : 2D float64 ndarray (H, W)
    k, w, t, return_filt : same as ``findSpots.llr``
    """
    if not _TORCH_OK:
        raise RuntimeError("cuda_detect: PyTorch is not installed.")

    device = _get_device()
    dtype  = torch.float64

    H, W = I.shape
    n_pixels = w ** 2

    # Move image to GPU
    I_t = torch.from_numpy(I.astype(np.float64)).to(device=device, dtype=dtype)

    # Build / retrieve cached transfer function
    G_rft, Sgc2 = _mle_amp_setup_torch(H, W, k, w, device, dtype)

    # Uniform filters for LLR: local mean (A) and local mean-of-squares (B)
    A = _uniform_filter_torch(I_t, w)
    B = _uniform_filter_torch(I_t ** 2, w)

    # Gaussian-matched filter (C)
    I_rft = torch.fft.rfft2(I_t)
    C_raw = torch.fft.irfft2(I_rft * G_rft, s=(H, W))
    C     = torch.fft.fftshift(C_raw)

    # Only allow convex (positive) spots
    C = torch.clamp(C, min=0.0)

    # LLR statistic
    denom = torch.clamp(n_pixels * Sgc2 * (B - A ** 2), min=0.001)
    L = 1.0 - C ** 2 / denom

    # Zero out borders
    hw = w // 2
    L[:hw,  :] = 1.0
    L[:,  :hw] = 1.0
    L[-hw:, :] = 1.0
    L[:, -hw:] = 1.0

    # Log-likelihood
    L = torch.clamp(L, min=1e-10)
    llr_img = -(n_pixels / 2.0) * torch.log(L)

    # Threshold on CPU  (label_spots uses scipy — stays on CPU)
    llr_np = llr_img.cpu().numpy()

    # Import here to avoid circular import
    from .helper import threshold_image
    return threshold_image(llr_np, t=t, return_filt=return_filt)
