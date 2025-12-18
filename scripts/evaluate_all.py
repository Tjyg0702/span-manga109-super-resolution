#!/usr/bin/env python3
"""Evaluate all experiment checkpoints and generate comparison."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.span import SPAN
from src.data.dataset import Manga109Dataset
from src.utils.metrics import calculate_psnr, calculate_ssim

PROJECT_ROOT = Path(__file__).parent.parent

def get_test_volumes(seed=42):
    import random
    images_dir = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
    volumes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    random.seed(seed)
    shuffled = volumes.copy()
    random.shuffle(shuffled)
    return sorted(shuffled[:19])

def evaluate(checkpoint_path, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SPAN(feature_channels=48, upscale=4)
    model.load_pretrained(str(checkpoint_path), strict=False)
    model = model.to(device)
    model.eval()
    
    test_dataset = Manga109Dataset(
        images_dir=PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images",
        volumes=get_test_volumes(),
        patches_per_epoch=500,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for lr, hr in tqdm(test_loader, desc=name, leave=False):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            for i in range(sr.shape[0]):
                psnr_list.append(calculate_psnr(sr[i], hr[i]))
                ssim_list.append(calculate_ssim(sr[i], hr[i]))
    
    return np.mean(psnr_list), np.std(psnr_list), np.mean(ssim_list), np.std(ssim_list)

def main():
    checkpoints = [
        ("checkpoints/SPAN_4x_pretrained.pth", "Pretrained (DF2K)"),
        ("checkpoints/exp1_baseline/best.pth", "Exp1: LR=2e-4"),
        ("checkpoints/exp2_lower_lr/best.pth", "Exp2: LR=5e-5"),
        ("checkpoints/exp3_perceptual/best.pth", "Exp3: L1+Perceptual"),
        ("checkpoints/exp4_longer/best.pth", "Exp4: 5000 iters"),
    ]
    
    print("="*70)
    print("EVALUATING ALL EXPERIMENTS ON MANGA109 TEST SET")
    print("="*70 + "\n")
    
    results = []
    for ckpt_path, name in checkpoints:
        full_path = PROJECT_ROOT / ckpt_path
        if full_path.exists():
            psnr_m, psnr_s, ssim_m, ssim_s = evaluate(full_path, name)
            results.append((name, psnr_m, psnr_s, ssim_m, ssim_s))
            print(f"{name}: PSNR={psnr_m:.2f}±{psnr_s:.2f}, SSIM={ssim_m:.4f}±{ssim_s:.4f}")
        else:
            print(f"⚠ Not found: {ckpt_path}")
    
    # Print table
    print("\n" + "="*70)
    print("RESULTS TABLE (for report)")
    print("="*70)
    print(f"{'Experiment':<25} {'PSNR (dB)':<18} {'SSIM':<18} {'Δ PSNR':<10}")
    print("-"*70)
    
    baseline_psnr = results[0][1] if results else 0
    for name, psnr_m, psnr_s, ssim_m, ssim_s in results:
        delta = psnr_m - baseline_psnr
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"{name:<25} {psnr_m:.2f} ± {psnr_s:.2f}      {ssim_m:.4f} ± {ssim_s:.4f}   {delta_str}")
    
    # Save to file
    with open(PROJECT_ROOT / "outputs/ablation_results.txt", 'w') as f:
        f.write("ABLATION STUDY: SPAN Fine-tuning on Manga109\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Experiment':<25} {'PSNR (dB)':<18} {'SSIM':<18} {'Δ PSNR':<10}\n")
        f.write("-"*70 + "\n")
        for name, psnr_m, psnr_s, ssim_m, ssim_s in results:
            delta = psnr_m - baseline_psnr
            delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            f.write(f"{name:<25} {psnr_m:.2f} ± {psnr_s:.2f}      {ssim_m:.4f} ± {ssim_s:.4f}   {delta_str}\n")
    
    print(f"\n✓ Results saved to outputs/ablation_results.txt")

if __name__ == "__main__":
    main()

