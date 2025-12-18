

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.span import SPAN
from src.models.swinir_official import SwinIR
from src.models.hat_official import HAT
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

def load_model_from_checkpoint(checkpoint_path, model_type='span', config_path=None):
    """Load model based on type"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'span':
        model = SPAN(feature_channels=48, upscale=4)
        model.load_pretrained(str(checkpoint_path), strict=False)
    
    elif model_type == 'swinir':
        model = SwinIR(
            upscale=4, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        )
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    
    elif model_type == 'hat':
        model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30,
            conv_scale=0.01, overlap_ratio=0.5,
            img_range=1., depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        )
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    return model

def evaluate(checkpoint_path, model_type, name):
    """Evaluate model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model_from_checkpoint(checkpoint_path, model_type)
    
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
    models_to_eval = [
        ("checkpoints/SPAN_4x_pretrained.pth", "span", "SPAN Pretrained"),
        ("checkpoints/exp1_baseline/best.pth", "span", "SPAN Exp1 (LR=2e-4)"),
        ("checkpoints/exp2_lower_lr/best.pth", "span", "SPAN Exp2 (LR=5e-5)"),
        ("checkpoints/exp3_perceptual/best.pth", "span", "SPAN Exp3 (Perceptual)"),
        ("checkpoints/exp4_longer/best.pth", "span", "SPAN Exp4 (5000 iters)"),
        
        # SwinIR comparisons
        ("checkpoints/swinir/best.pth", "swinir", "SwinIR from scratch"),
        ("checkpoints/swinir_finetuned/best.pth", "swinir", "SwinIR fine-tuned (10.15M)"),
        
        # HAT comparisons
        ("checkpoints/hat/best.pth", "hat", "HAT from scratch"),
        ("checkpoints/hat_finetuned/best.pth", "hat", "HAT fine-tuned (20.6M)"),
    ]
    
    print("="*80)
    print("EVALUATING ALL MODELS ON MANGA109 TEST SET")
    print("="*80 + "\n")
    
    results = []
    for ckpt_path, model_type, name in models_to_eval:
        full_path = PROJECT_ROOT / ckpt_path
        if full_path.exists():
            print(f"\nEvaluating: {name}")
            psnr_m, psnr_s, ssim_m, ssim_s = evaluate(full_path, model_type, name)
            results.append((name, model_type, psnr_m, psnr_s, ssim_m, ssim_s))
            print(f"  PSNR: {psnr_m:.2f}±{psnr_s:.2f} dB")
            print(f"  SSIM: {ssim_m:.4f}±{ssim_s:.4f}")
        else:
            print(f"⚠ Not found: {ckpt_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS TABLE - MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'PSNR (dB)':<18} {'SSIM':<18} {'Δ PSNR':<10}")
    print("-"*80)
    
    # Find SPAN baseline for comparison
    span_baseline_psnr = None
    for name, model_type, psnr_m, _, _, _ in results:
        if "Pretrained" in name and model_type == "span":
            span_baseline_psnr = psnr_m
            break
    
    if span_baseline_psnr is None and results:
        span_baseline_psnr = results[0][2]
    
    for name, model_type, psnr_m, psnr_s, ssim_m, ssim_s in results:
        delta = psnr_m - span_baseline_psnr if span_baseline_psnr else 0
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"{name:<30} {psnr_m:.2f} ± {psnr_s:.2f}      {ssim_m:.4f} ± {ssim_s:.4f}   {delta_str}")
    
    # Save comprehensive results
    output_file = PROJECT_ROOT / "outputs/model_comparison_results.txt"
    with open(output_file, 'w') as f:
        f.write("MODEL COMPARISON: SPAN vs SwinIR vs HAT on Manga109\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<30} {'PSNR (dB)':<18} {'SSIM':<18} {'Δ PSNR':<10}\n")
        f.write("-"*80 + "\n")
        for name, model_type, psnr_m, psnr_s, ssim_m, ssim_s in results:
            delta = psnr_m - span_baseline_psnr if span_baseline_psnr else 0
            delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            f.write(f"{name:<30} {psnr_m:.2f} ± {psnr_s:.2f}      {ssim_m:.4f} ± {ssim_s:.4f}   {delta_str}\n")
        
        f.write("\n\nMODEL SIZES:\n")
        f.write("-"*80 + "\n")
        f.write("SPAN:   1.84M parameters\n")
        f.write("SwinIR: 10.15M parameters (5.5x larger)\n")
        f.write("HAT:    22.56M parameters (12.3x larger)\n")
    
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()
