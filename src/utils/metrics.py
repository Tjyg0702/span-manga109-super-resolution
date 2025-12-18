"""Image quality metrics: PSNR, SSIM."""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity

def calculate_psnr(
    img1: torch.Tensor | np.ndarray, 
    img2: torch.Tensor | np.ndarray,
    max_val: float = 1.0
) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical SR results: 28-35 dB.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Remove batch dim if present
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    if img1.shape[0] in [1, 3] and img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] in [1, 3] and img2.ndim == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(
    img1: torch.Tensor | np.ndarray,
    img2: torch.Tensor | np.ndarray,
    data_range: float = 1.0
) -> float:
    """
    Structural Similarity Index.
    Range: [-1, 1], higher is better. Good SR: >0.9.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Remove batch dim if present
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    # Convert from (C, H, W) to (H, W, C)
    if img1.shape[0] in [1, 3] and img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] in [1, 3] and img2.ndim == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Compute SSIM
    if img1.ndim == 3 and img1.shape[2] == 3:
        return structural_similarity(img1, img2, data_range=data_range, channel_axis=2)
    else:
        return structural_similarity(img1, img2, data_range=data_range)

def calculate_metrics_batch(
    sr_batch: torch.Tensor,
    hr_batch: torch.Tensor
) -> tuple[float, float]:
    """Calculate average PSNR and SSIM for a batch."""
    psnr_sum = 0.0
    ssim_sum = 0.0
    batch_size = sr_batch.shape[0]
    
    for i in range(batch_size):
        psnr_sum += calculate_psnr(sr_batch[i], hr_batch[i])
        ssim_sum += calculate_ssim(sr_batch[i], hr_batch[i])
    
    return psnr_sum / batch_size, ssim_sum / batch_size

