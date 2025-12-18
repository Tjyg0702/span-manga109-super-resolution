"""Training loop for SPAN super-resolution."""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..utils.metrics import calculate_psnr, calculate_ssim

class Trainer:
    """
    SPAN training with:
    - AdamW optimizer (wd=0.01)
    - Cosine annealing LR (2e-4 → 1e-6)
    - Early stopping on val PSNR plateau
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 2e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 0.01,
        max_iters: int = 5000,
        val_interval: int = 500,
        patience: int = 1000,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_iters = max_iters
        self.val_interval = val_interval
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.criterion = nn.L1Loss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_iters, eta_min=min_lr)
        
        # Tracking
        self.train_losses = []
        self.val_psnrs = []
        self.best_psnr = 0.0
        self.iters_no_improve = 0
    
    def train(self) -> dict:
        """Main training loop. Returns training history."""
        self.model.train()
        train_iter = iter(self.train_loader)
        
        pbar = tqdm(range(self.max_iters), desc="Training")
        start_time = time.time()
        
        for iteration in pbar:
            # Get batch (cycle through loader)
            try:
                lr_batch, hr_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                lr_batch, hr_batch = next(train_iter)
            
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            # Forward
            sr_batch = self.model(lr_batch)
            loss = self.criterion(sr_batch, hr_batch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.train_losses.append(loss.item())
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Validation
            if (iteration + 1) % self.val_interval == 0:
                val_psnr, val_ssim = self.validate()
                self.val_psnrs.append(val_psnr)
                
                print(f"\n[Iter {iteration+1}] Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")
                
                # Save best model
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.iters_no_improve = 0
                    self.save_checkpoint("best.pth")
                    print(f"  ✓ New best PSNR! Saved checkpoint.")
                else:
                    self.iters_no_improve += self.val_interval
                
                # Early stopping
                if self.iters_no_improve >= self.patience:
                    print(f"\n⚠ Early stopping: no improvement for {self.patience} iterations")
                    break
                
                self.model.train()
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        print(f"Best validation PSNR: {self.best_psnr:.2f} dB")
        
        # Save final checkpoint
        self.save_checkpoint("final.pth")
        
        return {
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs,
            'best_psnr': self.best_psnr,
            'elapsed_minutes': elapsed / 60
        }
    
    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Run validation, return avg PSNR and SSIM."""
        self.model.eval()
        
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0
        
        for lr_batch, hr_batch in self.val_loader:
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            sr_batch = self.model(lr_batch)
            
            for i in range(sr_batch.shape[0]):
                psnr_sum += calculate_psnr(sr_batch[i], hr_batch[i])
                ssim_sum += calculate_ssim(sr_batch[i], hr_batch[i])
                count += 1
        
        return psnr_sum / count, ssim_sum / count
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_psnrs = checkpoint.get('val_psnrs', [])

