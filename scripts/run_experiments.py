#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent

EXPERIMENTS = [
    "configs/exp1_baseline_lr2e4.yaml",
    "configs/exp2_lower_lr.yaml",
    "configs/exp3_perceptual_loss.yaml",
    "configs/exp4_longer_training.yaml",
]

def run_experiment(config_path: str):
    """Run single training experiment."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {config_path}")
    print('='*60)
    
    cmd = [
        sys.executable, 
        "scripts/train.py", 
        "--config", config_path
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def evaluate_experiment(checkpoint_path: str, output_name: str):
    """Evaluate a trained model."""
    from src.models.span import SPAN
    from src.data.dataset import Manga109Dataset
    from src.utils.metrics import calculate_psnr, calculate_ssim
    from torch.utils.data import DataLoader
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SPAN(feature_channels=48, upscale=4)
    model.load_pretrained(checkpoint_path, strict=False)
    model = model.to(device)
    model.eval()
    
    # Get test volumes
    import random
    images_dir = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
    volumes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    random.seed(42)
    shuffled = volumes.copy()
    random.shuffle(shuffled)
    test_volumes = sorted(shuffled[:19])
    
    test_dataset = Manga109Dataset(
        images_dir=images_dir,
        volumes=test_volumes,
        patches_per_epoch=500,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for lr_batch, hr_batch in test_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            sr_batch = model(lr_batch)
            
            for i in range(sr_batch.shape[0]):
                psnr_list.append(calculate_psnr(sr_batch[i], hr_batch[i]))
                ssim_list.append(calculate_ssim(sr_batch[i], hr_batch[i]))
    
    return {
        'name': output_name,
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list)
    }

def main():
    print("="*60)
    print("ABLATION STUDY: SPAN Fine-tuning on Manga109")
    print("="*60)
    
    # Run all experiments
    for exp_config in EXPERIMENTS:
        success = run_experiment(exp_config)
        if not success:
            print(f"⚠ Experiment {exp_config} failed!")
    
    # Evaluate all experiments
    print("\n" + "="*60)
    print("EVALUATING ALL EXPERIMENTS")
    print("="*60)
    
    results = []
    
    # Baseline (pretrained)
    print("\nEvaluating: Pretrained baseline...")
    baseline = evaluate_experiment(
        str(PROJECT_ROOT / "checkpoints/SPAN_4x_pretrained.pth"),
        "Pretrained (DF2K)"
    )
    results.append(baseline)
    
    # Each experiment
    exp_names = [
        ("exp1_baseline", "Exp1: LR=2e-4 (baseline)"),
        ("exp2_lower_lr", "Exp2: LR=5e-5 (lower)"),
        ("exp3_perceptual", "Exp3: L1+Perceptual"),
        ("exp4_longer", "Exp4: Full 5000 iters"),
    ]
    
    for exp_dir, exp_name in exp_names:
        checkpoint = PROJECT_ROOT / f"checkpoints/{exp_dir}/best.pth"
        if checkpoint.exists():
            print(f"\nEvaluating: {exp_name}...")
            result = evaluate_experiment(str(checkpoint), exp_name)
            results.append(result)
        else:
            print(f"⚠ Checkpoint not found: {checkpoint}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Experiment':<30} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*60)
    
    for r in results:
        psnr_str = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
        ssim_str = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
        print(f"{r['name']:<30} {psnr_str:<15} {ssim_str:<15}")
    
    # Save results
    results_file = PROJECT_ROOT / "outputs/experiment_results.txt"
    with open(results_file, 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Experiment':<30} {'PSNR (dB)':<15} {'SSIM':<15}\n")
        f.write("-"*60 + "\n")
        for r in results:
            psnr_str = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
            ssim_str = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
            f.write(f"{r['name']:<30} {psnr_str:<15} {ssim_str:<15}\n")
    
    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()

