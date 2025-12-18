"""Visualization utilities for SR results."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) to numpy (H, W, C) in [0, 255]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr

def visualize_comparison(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    bicubic: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    title: str = "",
    metrics: Optional[dict] = None
):
    """
    Side-by-side comparison: LR | Bicubic | SR | HR
    """
    ncols = 4 if bicubic is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    
    lr_np = tensor_to_numpy(lr)
    sr_np = tensor_to_numpy(sr)
    hr_np = tensor_to_numpy(hr)
    
    axes[0].imshow(lr_np)
    axes[0].set_title(f"LR ({lr_np.shape[1]}x{lr_np.shape[0]})")
    axes[0].axis('off')
    
    idx = 1
    if bicubic is not None:
        bic_np = tensor_to_numpy(bicubic)
        axes[idx].imshow(bic_np)
        axes[idx].set_title("Bicubic")
        axes[idx].axis('off')
        idx += 1
    
    sr_title = "SR (Model)"
    if metrics and 'sr_psnr' in metrics:
        sr_title += f"\nPSNR: {metrics['sr_psnr']:.2f}"
    axes[idx].imshow(sr_np)
    axes[idx].set_title(sr_title)
    axes[idx].axis('off')
    idx += 1
    
    axes[idx].imshow(hr_np)
    axes[idx].set_title("HR (Ground Truth)")
    axes[idx].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(
    train_losses: list[float],
    val_psnrs: list[float],
    save_path: Optional[Path] = None
):
    """Plot training loss and validation PSNR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('L1 Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_psnrs)
    ax2.set_xlabel('Validation Step')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparison_grid(
    samples: list[dict],
    save_path: Path,
    title: str = "Model Comparison"
):
    """
    Create a grid showing multiple samples.
    Each sample dict: {'lr': tensor, 'sr': tensor, 'hr': tensor, 'name': str}
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        lr_np = tensor_to_numpy(sample['lr'])
        sr_np = tensor_to_numpy(sample['sr'])
        hr_np = tensor_to_numpy(sample['hr'])
        
        axes[i, 0].imshow(lr_np)
        axes[i, 0].set_title(f"LR" if i == 0 else "")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sr_np)
        axes[i, 1].set_title(f"SR (Ours)" if i == 0 else "")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(hr_np)
        axes[i, 2].set_title(f"HR (GT)" if i == 0 else "")
        axes[i, 2].axis('off')
        
        # Add sample name on the left
        axes[i, 0].set_ylabel(sample.get('name', f'Sample {i+1}'), fontsize=10)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

