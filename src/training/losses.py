"""Loss functions for super-resolution training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class L1Loss(nn.Module):
    """
    L1 (Mean Absolute Error) loss.
    Produces sharper outputs than L2/MSE, better for line art edges.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for enhanced texture quality.
    Optional addition to L1 for ablation study.
    """
    def __init__(self, layer_idx: int = 35, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        
        # Freeze VGG
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        
        sr_feat = self.features(sr_norm)
        hr_feat = self.features(hr_norm)
        
        return self.weight * F.l1_loss(sr_feat, hr_feat)


class CombinedLoss(nn.Module):
    """L1 + optional perceptual loss."""
    def __init__(self, use_perceptual: bool = False, perceptual_weight: float = 0.1):
        super().__init__()
        self.l1 = L1Loss()
        self.perceptual = PerceptualLoss(weight=perceptual_weight) if use_perceptual else None
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = self.l1(sr, hr)
        if self.perceptual:
            loss = loss + self.perceptual(sr, hr)
        return loss

