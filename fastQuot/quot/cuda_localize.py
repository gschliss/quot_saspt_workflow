"""
cuda_localize.py — GPU-accelerated subpixel localization using PyTorch.

Replaces the Python-loop-over-spots in subpixel.localize_frame with a fully
batched Levenberg-Marquardt fitter for the integrated-Gaussian (ls_int_gaussian)
PSF model.

All N spots in one frame are fit simultaneously as a single batched matrix
operation.  Runs on CUDA if available; also faster than the per-spot Python
loop even on CPU.

Public API
----------
localize_frame_gpu(img, positions, method, window_size, camera_bg,
                   camera_gain, **method_kwargs)
    Drop-in replacement for subpixel.localize_frame when
    method == 'ls_int_gaussian'.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> "torch.device":
    if _TORCH_OK and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Batched radial-symmetry initial guess
# ---------------------------------------------------------------------------

def _ring_mean_batch(patches: "torch.Tensor") -> "torch.Tensor":
    """
    Mean of the outer-ring pixels for each patch in the batch.

    Parameters
    ----------
    patches : (N, H, W) float tensor

    Returns
    -------
    (N,) float tensor
    """
    top    = patches[:, 0,  :-1].mean(dim=-1)
    right  = patches[:, :-1, -1].mean(dim=-1)
    bottom = patches[:, -1,  1:].mean(dim=-1)
    left   = patches[:,  1:,  0].mean(dim=-1)
    return (top + right + bottom + left) / 4.0


def _batched_radial_symmetry(patches: "torch.Tensor"):
    """
    Vectorised radial-symmetry localisation (Parthasarathy 2012) for a
    batch of PSF images.

    Parameters
    ----------
    patches : (N, H, W) float tensor

    Returns
    -------
    yc, xc : (N,) tensors, sub-pixel centre estimates in patch coordinates
    """
    N, H, W = patches.shape
    device  = patches.device
    dtype   = patches.dtype

    # Diagonal intensity gradients across each 2×2 corner block
    dI_du = patches[:, :H-1, 1:]  - patches[:, 1:, :W-1]  # (N, H-1, W-1)
    dI_dv = patches[:, :H-1, :W-1] - patches[:, 1:, 1:]    # (N, H-1, W-1)

    # Smooth gradients with a 3×3 uniform filter
    k_smooth = (torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9.0)
    du = F.conv2d(dI_du.reshape(N, 1, H-1, W-1), k_smooth, padding=1).reshape(N, H-1, W-1)
    dv = F.conv2d(dI_dv.reshape(N, 1, H-1, W-1), k_smooth, padding=1).reshape(N, H-1, W-1)

    dI2 = du ** 2 + dv ** 2  # (N, H-1, W-1)

    # Pixel-centre grid (relative to image centre)
    # Shape (1, H-1, 1) and (1, 1, W-1) so they broadcast with (N, H-1, W-1)
    H_half = H // 2
    W_half = W // 2
    ym = (torch.arange(H-1, device=device, dtype=dtype) - H_half + 0.5).view(1, H-1, 1)
    xm = (torch.arange(W-1, device=device, dtype=dtype) - W_half + 0.5).view(1, 1, W-1)

    # Gradient slopes  m = -(dv + du) / (du - dv)
    denom_m = du - dv
    denom_m = denom_m + (denom_m.abs() < 1e-10).float() * 1e-10  # avoid /0
    m = -(dv + du) / denom_m
    m = torch.clamp(m, -9e9, 9e9)
    b = ym - m * xm  # (N, H-1, W-1)

    # Intensity-weighted centroid for the weighting distance
    sdI2       = dI2.sum(dim=(-2, -1), keepdim=True).clamp(1e-10)
    ycentroid  = (dI2 * ym).sum(dim=(-2, -1), keepdim=True) / sdI2
    xcentroid  = (dI2 * xm).sum(dim=(-2, -1), keepdim=True) / sdI2

    dist = ((xm - xcentroid) ** 2 + (ym - ycentroid) ** 2).sqrt().clamp(1e-10)
    w    = dI2 / dist  # (N, H-1, W-1)

    # Nan mask from degenerate slopes
    nan_mask = torch.isnan(m)
    w = w.masked_fill(nan_mask, 0.0)
    b = b.masked_fill(nan_mask, 0.0)
    m = m.masked_fill(nan_mask, 0.0)

    wm2p1 = w / (m ** 2 + 1.0)

    sw   = wm2p1.sum(dim=(-2, -1))
    smmw = (m ** 2 * wm2p1).sum(dim=(-2, -1))
    smw  = (m * wm2p1).sum(dim=(-2, -1))
    smbw = (m * b * wm2p1).sum(dim=(-2, -1))
    sbw  = (b * wm2p1).sum(dim=(-2, -1))

    # Preserve sign of det; only guard against |det| < 1e-10
    det_raw = smw ** 2 - smmw * sw
    det = torch.where(
        det_raw.abs() < 1e-10,
        torch.full_like(det_raw, 1e-10),
        det_raw,
    )

    xc = (smbw * sw   - smw  * sbw) / det
    yc = (smbw * smw  - smmw * sbw) / det

    # Convert to image-frame coordinates (match CPU rs() output)
    yc = yc + (H + 1) / 2.0 - 1.0
    xc = xc + (W + 1) / 2.0 - 1.0

    # Clamp to valid window range
    yc = yc.clamp(0.0, float(H - 1))
    xc = xc.clamp(0.0, float(W - 1))

    return yc, xc


# ---------------------------------------------------------------------------
# Batched initial-intensity estimate
# ---------------------------------------------------------------------------

def _estimate_I0_batch(patches: "torch.Tensor", yc: "torch.Tensor",
                        xc: "torch.Tensor", bg: "torch.Tensor",
                        S: "torch.Tensor") -> "torch.Tensor":
    """
    Estimate I0 from the brightest pixel in each patch.

    Parameters
    ----------
    patches : (N, H, W)
    yc, xc  : (N,) — sub-pixel centres
    bg      : (N,) — background per pixel
    S       : scalar — sqrt(2) * sigma

    Returns
    -------
    I0 : (N,)
    """
    N, H, W = patches.shape

    # Index of brightest pixel
    flat_idx = patches.reshape(N, -1).argmax(dim=-1)   # (N,)
    ym = (flat_idx // W).float()                        # (N,)
    xm = (flat_idx %  W).float()                        # (N,)

    # Integrated Gaussian value at that pixel
    py = 0.5 * (torch.erf((ym - yc + 0.5) / S) - torch.erf((ym - yc - 0.5) / S))
    px = 0.5 * (torch.erf((xm - xc + 0.5) / S) - torch.erf((xm - xc - 0.5) / S))
    psf_val = (py * px).clamp(1e-6)

    obs_bright = patches[torch.arange(N, device=patches.device),
                         flat_idx // W, flat_idx % W]
    I0 = ((obs_bright - bg) / psf_val).clamp(0.0, 1e7)
    return I0


# ---------------------------------------------------------------------------
# Batched ring variance (for SNR)
# ---------------------------------------------------------------------------

def _ring_var_batch(patches: "torch.Tensor") -> "torch.Tensor":
    """
    Variance of the outer-ring pixels for each patch.

    Parameters
    ----------
    patches : (N, H, W)

    Returns
    -------
    (N,) float tensor
    """
    N, H, W = patches.shape
    rings = torch.cat([
        patches[:, 0,  1:],          # top edge  (left→right, skip corner)
        patches[:, 1:, -1],          # right edge (top→bottom)
        patches[:, -1, :-1],         # bottom edge (right→left)
        patches[:, :-1, 0],          # left edge  (bottom→top)
    ], dim=-1)                        # (N, perimeter)
    return rings.var(dim=-1, unbiased=True).clamp(1e-10)


# ---------------------------------------------------------------------------
# Core batched Levenberg-Marquardt fitter (integrated Gaussian PSF)
# ---------------------------------------------------------------------------

def _batch_lm_int_gaussian(patches: "torch.Tensor", sigma: float = 1.0,
                            ridge: float = 1e-4, max_iter: int = 10,
                            damp: float = 0.3, convergence: float = 1e-4):
    """
    Fit an integrated-Gaussian PSF to every patch simultaneously.

    Model:  f(y,x) = I0 * G_y(y) * G_x(x) + bg
    where   G_y(y) = 0.5 * (erf((y+0.5-yc)/S) - erf((y-0.5-yc)/S))
            S = sqrt(2)*sigma

    Parameters
    ----------
    patches  : (N, H, W) float64 tensor (background-subtracted, gain-corrected)
    sigma, ridge, max_iter, damp, convergence : same as fit_ls_int_gaussian

    Returns
    -------
    pars  : (N, 4)  — [y, x, I0, bg]
    err   : (N, 4)  — sqrt of -H_inv diagonal
    H_det : (N,)    — determinant of -H_inv
    rmse  : (N,)    — sqrt(mean squared residual)
    """
    N, H, W = patches.shape
    device   = patches.device
    dtype    = patches.dtype

    # Pre-compute sigma constants
    S   = torch.tensor(math.sqrt(2.0) * sigma, device=device, dtype=dtype)
    A   = torch.tensor(2.0 * sigma ** 2,        device=device, dtype=dtype)
    B_d = torch.tensor(math.sqrt(math.pi * 2.0 * sigma ** 2), device=device, dtype=dtype)

    # Pixel index vectors
    Y_idx = torch.arange(H, device=device, dtype=dtype)  # (H,)
    X_idx = torch.arange(W, device=device, dtype=dtype)  # (W,)

    # ---- Initial guess ----
    bg0       = _ring_mean_batch(patches)                             # (N,)
    yc0, xc0  = _batched_radial_symmetry(patches)                     # (N,)
    I0_0      = _estimate_I0_batch(patches, yc0, xc0, bg0, S)        # (N,)

    # pars: (N, 4)  columns = [y, x, I0, bg]
    pars = torch.stack([yc0, xc0, I0_0, bg0], dim=-1).clone()

    H_mat = torch.zeros(N, 4, 4, device=device, dtype=dtype)
    eye4  = torch.eye(4, device=device, dtype=dtype)

    # ---- LM iterations ----
    for _iter in range(max_iter):
        y  = pars[:, 0].unsqueeze(-1)   # (N, 1)
        x  = pars[:, 1].unsqueeze(-1)   # (N, 1)
        I0 = pars[:, 2]                 # (N,)
        bg = pars[:, 3]                 # (N,)

        # Integrated Gaussian projections: (N, H) and (N, W)
        proj_y = 0.5 * (torch.erf((Y_idx - y + 0.5) / S) -
                         torch.erf((Y_idx - y - 0.5) / S))   # (N, H)
        proj_x = 0.5 * (torch.erf((X_idx - x + 0.5) / S) -
                         torch.erf((X_idx - x - 0.5) / S))   # (N, W)

        # Derivatives wrt centre positions: (N, H) and (N, W)
        dproj_y = (torch.exp(-(Y_idx - y - 0.5) ** 2 / A) -
                   torch.exp(-(Y_idx - y + 0.5) ** 2 / A)) / B_d
        dproj_x = (torch.exp(-(X_idx - x - 0.5) ** 2 / A) -
                   torch.exp(-(X_idx - x + 0.5) ** 2 / A)) / B_d

        # Outer products for Jacobian columns — (N, H, W) each
        # J_y  = I0 * outer(dproj_y, proj_x)
        # J_x  = I0 * outer(proj_y,  dproj_x)
        # J_I0 = outer(proj_y, proj_x)
        # J_bg = ones
        J_I0  = torch.bmm(proj_y.unsqueeze(2), proj_x.unsqueeze(1))              # (N, H, W)
        J_y   = I0.view(N, 1, 1) * torch.bmm(dproj_y.unsqueeze(2), proj_x.unsqueeze(1))
        J_x   = I0.view(N, 1, 1) * torch.bmm(proj_y.unsqueeze(2),  dproj_x.unsqueeze(1))
        J_bg  = torch.ones(N, H, W, device=device, dtype=dtype)

        # PSF model
        model = I0.view(N, 1, 1) * J_I0 + bg.view(N, 1, 1)

        # Residuals and noise variance
        resid = patches - model                                        # (N, H, W)
        sig2  = (resid ** 2).mean(dim=(-2, -1)).clamp(1e-10)         # (N,)

        # Flatten spatial dims → (N, P, 4)
        P      = H * W
        J_flat = torch.stack([J_y.reshape(N, P),
                               J_x.reshape(N, P),
                               J_I0.reshape(N, P),
                               J_bg.reshape(N, P)], dim=-1)           # (N, P, 4)
        r_flat = resid.reshape(N, P)                                   # (N, P)

        # Gradient: (N, 4)
        sig2_inv = (1.0 / sig2).view(N, 1)
        grad = torch.bmm(J_flat.transpose(1, 2),
                         r_flat.unsqueeze(2)).squeeze(2) * sig2_inv   # (N, 4)

        # Hessian: (N, 4, 4)
        H_mat = -torch.bmm(J_flat.transpose(1, 2), J_flat) * sig2_inv.unsqueeze(2)

        # Ridge-regularised solve: H_reg @ step = -grad
        H_reg = H_mat - ridge * eye4.unsqueeze(0)                     # (N, 4, 4)
        try:
            step = torch.linalg.solve(H_reg, -grad.unsqueeze(-1)).squeeze(-1)  # (N, 4)
        except RuntimeError:
            # Increase ridge on singular batches and retry once
            H_reg2 = H_mat - (ridge * 100) * eye4.unsqueeze(0)
            try:
                step = torch.linalg.solve(H_reg2, -grad.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                break  # give up; keep current pars

        pars = pars + damp * step

        # Early exit if all spots have converged in (y, x)
        if (step[:, :2].abs().max() < convergence):
            break

    # ---- Error estimates from final Hessian ----
    H_final = H_mat - ridge * eye4.unsqueeze(0)
    try:
        H_inv = torch.linalg.inv(-H_final)                            # (N, 4, 4)
    except RuntimeError:
        H_inv = torch.zeros(N, 4, 4, device=device, dtype=dtype)

    diag = torch.diagonal(H_inv, dim1=-2, dim2=-1).clamp(0.0)        # (N, 4)
    err  = diag.sqrt()
    try:
        H_det = torch.linalg.det(H_inv)                               # (N,)
    except RuntimeError:
        H_det = torch.zeros(N, device=device, dtype=dtype)

    rmse  = sig2.sqrt()
    return pars, err, H_det, rmse


# ---------------------------------------------------------------------------
# Public interface — drop-in for subpixel.localize_frame
# ---------------------------------------------------------------------------

def localize_frame_gpu(img: np.ndarray, positions: np.ndarray,
                        method: str | None = None, window_size: int = 9,
                        camera_bg: float = 0.0, camera_gain: float = 1.0,
                        **method_kwargs) -> pd.DataFrame:
    """
    GPU-accelerated frame localisation for the ``ls_int_gaussian`` method.

    Falls back transparently to the CPU implementation for any other method
    or when CUDA is unavailable.

    Parameters
    ----------
    img          : 2D ndarray (H, W), raw camera frame
    positions    : (N, 2) int array of [y, x] detection positions
    method       : localisation method name
    window_size  : fitting window (must be odd; default 9)
    camera_bg    : camera background to subtract
    camera_gain  : camera gain (grey-values / photon)
    **method_kwargs : forwarded to the underlying fitter

    Returns
    -------
    pandas.DataFrame with the same columns as ``subpixel.localize_frame``
    """
    if not _TORCH_OK or method != "ls_int_gaussian":
        from .subpixel import localize_frame as _cpu
        return _cpu(img, positions, method=method, window_size=window_size,
                    camera_bg=camera_bg, camera_gain=camera_gain, **method_kwargs)

    if not isinstance(positions, np.ndarray) or positions.ndim != 2 or len(positions) == 0:
        return pd.DataFrame()

    hw = window_size // 2

    # Remove detections too close to the edge
    valid_mask = ((positions[:, 0] >= hw) &
                  (positions[:, 0] <  img.shape[0] - hw) &
                  (positions[:, 1] >= hw) &
                  (positions[:, 1] <  img.shape[1] - hw))
    positions = positions[valid_mask]
    if len(positions) == 0:
        return pd.DataFrame()

    # Background-subtract and gain-normalise the full frame (float64, on CPU)
    img_f = np.clip(img.astype(np.float64) - camera_bg, 0.0, np.inf) / camera_gain

    # Extract all PSF windows at once — shape (N, w, w)
    N  = len(positions)
    w  = window_size
    patches_np = np.lib.stride_tricks.as_strided(
        img_f,
        shape=(N, w, w),
        # We can't use stride tricks for non-contiguous windows; use loop
    ).copy() if False else np.stack([
        img_f[yd - hw: yd + hw + 1, xd - hw: xd + hw + 1]
        for yd, xd in positions
    ])                                                                  # (N, w, w)

    # Move to GPU
    device  = _get_device()
    patches = torch.from_numpy(patches_np).to(device=device, dtype=torch.float64)

    # Fitter kwargs
    sigma    = float(method_kwargs.get("sigma",    1.0))
    ridge    = float(method_kwargs.get("ridge",    1e-4))
    max_iter = int  (method_kwargs.get("max_iter", 10))
    damp     = float(method_kwargs.get("damp",     0.3))
    conv     = float(method_kwargs.get("convergence", 1e-4))

    # Run batched fitter
    pars, err, H_det, rmse = _batch_lm_int_gaussian(
        patches, sigma=sigma, ridge=ridge,
        max_iter=max_iter, damp=damp, convergence=conv,
    )

    # SNR requires ring variance per spot
    ring_var = _ring_var_batch(patches)                                # (N,)

    # Move results back to CPU numpy
    pars_np    = pars.cpu().numpy()
    err_np     = err.cpu().numpy()
    H_det_np   = H_det.cpu().numpy()
    rmse_np    = rmse.cpu().numpy()
    rv_np      = ring_var.cpu().numpy()

    amp_sq = (pars_np[:, 2] / (2.0 * math.pi * sigma ** 2)) ** 2     # I0→amplitude²
    snr    = np.where(rv_np > 0, amp_sq / rv_np, np.inf)

    # Build output DataFrame (same columns as CPU localize_frame)
    y_rel = pars_np[:, 0]                                             # in-window coords
    x_rel = pars_np[:, 1]
    y_detect = positions[:, 0].astype(float)
    x_detect = positions[:, 1].astype(float)
    y_global = y_rel + y_detect - hw
    x_global = x_rel + x_detect - hw

    I0 = pars_np[:, 2]
    bg = pars_np[:, 3]

    error_flag = ~(
        (y_rel >= 0) & (y_rel < w) &
        (x_rel >= 0) & (x_rel < w) &
        (I0 >= 0)    & (I0 <= 10000) &
        (bg >= 0)
    )

    df = pd.DataFrame({
        "y":          y_global,
        "x":          x_global,
        "I0":         I0,
        "bg":         bg,
        "y_err":      err_np[:, 0],
        "x_err":      err_np[:, 1],
        "I0_err":     err_np[:, 2],
        "bg_err":     err_np[:, 3],
        "H_det":      H_det_np,
        "error_flag": error_flag.astype(int),
        "snr":        snr,
        "rmse":       rmse_np,
        "n_iter":     max_iter,
        "y_detect":   y_detect,
        "x_detect":   x_detect,
    })
    return df
