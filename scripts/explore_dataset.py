#!/usr/bin/env python3
"""Dataset exploration: statistics, samples, and structure analysis."""

import os
import sys
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === PATHS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def get_all_volumes():
    """Returns sorted list of all manga volume names."""
    return sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])

def get_volume_images(volume_name: str) -> list[Path]:
    """Returns list of image paths for a given volume."""
    vol_dir = IMAGES_DIR / volume_name
    return sorted(vol_dir.glob("*.jpg"))

def analyze_dataset():
    """Compute dataset statistics: volumes, pages, resolutions."""
    volumes = get_all_volumes()
    print(f"Total volumes: {len(volumes)}")
    
    stats = {
        'total_images': 0,
        'pages_per_volume': [],
        'widths': [],
        'heights': [],
        'aspects': [],
        'is_grayscale': 0,
        'is_rgb': 0
    }
    
    # Sample images from each volume for resolution stats
    for vol in volumes:
        images = get_volume_images(vol)
        stats['total_images'] += len(images)
        stats['pages_per_volume'].append(len(images))
        
        # Sample a few images per volume for resolution analysis
        for img_path in random.sample(images, min(3, len(images))):
            with Image.open(img_path) as img:
                w, h = img.size
                stats['widths'].append(w)
                stats['heights'].append(h)
                stats['aspects'].append(w / h)
                if img.mode == 'L':
                    stats['is_grayscale'] += 1
                else:
                    stats['is_rgb'] += 1
    
    print(f"\nTotal images: {stats['total_images']}")
    print(f"Pages per volume: min={min(stats['pages_per_volume'])}, max={max(stats['pages_per_volume'])}, avg={np.mean(stats['pages_per_volume']):.1f}")
    print(f"\nImage dimensions (sampled):")
    print(f"  Width:  min={min(stats['widths'])}, max={max(stats['widths'])}, avg={np.mean(stats['widths']):.0f}")
    print(f"  Height: min={min(stats['heights'])}, max={max(stats['heights'])}, avg={np.mean(stats['heights']):.0f}")
    print(f"  Aspect: min={min(stats['aspects']):.2f}, max={max(stats['aspects']):.2f}, avg={np.mean(stats['aspects']):.2f}")
    print(f"\nColor mode: RGB={stats['is_rgb']}, Grayscale={stats['is_grayscale']}")
    
    return stats

def create_train_val_test_split(volumes: list[str], seed: int = 42):
    """
    Split volumes into train/val/test as per proposal:
    - Train: 80 volumes
    - Val: 10 volumes  
    - Test: 19 volumes (standard Manga109 test split)
    """
    random.seed(seed)
    shuffled = volumes.copy()
    random.shuffle(shuffled)
    
    # Standard split from proposal
    test_volumes = shuffled[:19]
    val_volumes = shuffled[19:29]
    train_volumes = shuffled[29:]
    
    return {
        'train': sorted(train_volumes),
        'val': sorted(val_volumes),
        'test': sorted(test_volumes)
    }

def visualize_samples(num_samples: int = 6):
    """Show random sample pages from different volumes."""
    volumes = get_all_volumes()
    selected_vols = random.sample(volumes, num_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, vol in zip(axes, selected_vols):
        images = get_volume_images(vol)
        img_path = random.choice(images)
        img = Image.open(img_path)
        ax.imshow(np.array(img), cmap='gray' if img.mode == 'L' else None)
        ax.set_title(f"{vol}\n{img.size[0]}x{img.size[1]}", fontsize=9)
        ax.axis('off')
    
    plt.suptitle("Random Manga109 Samples", fontsize=14)
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "sample_pages.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample visualization to {OUTPUT_DIR / 'sample_pages.png'}")

def show_degradation_preview():
    """Show HR vs bicubic-downsampled LR comparison."""
    volumes = get_all_volumes()
    vol = random.choice(volumes)
    images = get_volume_images(vol)
    img_path = random.choice(images)
    
    hr_img = Image.open(img_path).convert('RGB')
    
    # Crop a 192x192 patch from center-ish area
    w, h = hr_img.size
    left = max(0, (w - 192) // 2)
    top = max(0, (h - 192) // 2)
    hr_patch = hr_img.crop((left, top, left + 192, top + 192))
    
    # Create LR (x4 downscale with bicubic)
    lr_patch = hr_patch.resize((48, 48), Image.BICUBIC)
    
    # Upscale LR back with bicubic for comparison
    bicubic_upscaled = lr_patch.resize((192, 192), Image.BICUBIC)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(np.array(hr_patch))
    axes[0].set_title("HR (192x192)\nGround Truth")
    axes[0].axis('off')
    
    axes[1].imshow(np.array(lr_patch))
    axes[1].set_title("LR (48x48)\nBicubic Downsampled")
    axes[1].axis('off')
    
    axes[2].imshow(np.array(bicubic_upscaled))
    axes[2].set_title("Bicubic Upscaled (192x192)\nBaseline")
    axes[2].axis('off')
    
    plt.suptitle(f"Degradation Preview - {vol}", fontsize=12)
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "degradation_preview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved degradation preview to {OUTPUT_DIR / 'degradation_preview.png'}")

if __name__ == "__main__":
    print("=" * 60)
    print("MANGA109 DATASET EXPLORATION")
    print("=" * 60)
    
    # Basic stats
    stats = analyze_dataset()
    
    # Volume split
    print("\n" + "=" * 60)
    print("TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    volumes = get_all_volumes()
    splits = create_train_val_test_split(volumes)
    print(f"Train: {len(splits['train'])} volumes")
    print(f"Val:   {len(splits['val'])} volumes")
    print(f"Test:  {len(splits['test'])} volumes")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    visualize_samples()
    show_degradation_preview()
    
    print("\n✓ Exploration complete!")

