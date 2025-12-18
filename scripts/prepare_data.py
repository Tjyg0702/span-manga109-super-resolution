#!/usr/bin/env python3
"""
Prepare Manga109 dataset: create train/val/test splits and extract patches.

Usage: python scripts/prepare_data.py [--patches-per-image 8]
"""

import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import prepare_dataset_splits

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def get_volume_split(seed: int = 42):
    """Split 109 volumes: 80 train, 10 val, 19 test."""
    volumes = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])
    
    random.seed(seed)
    shuffled = volumes.copy()
    random.shuffle(shuffled)
    
    return {
        'test': sorted(shuffled[:19]),
        'val': sorted(shuffled[19:29]),
        'train': sorted(shuffled[29:])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches-per-image', type=int, default=8,
                        help='Number of patches to extract per training image')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PREPARING MANGA109 DATASET")
    print("=" * 60)
    
    # Get split
    splits = get_volume_split(args.seed)
    print(f"\nVolume split:")
    print(f"  Train: {len(splits['train'])} volumes")
    print(f"  Val:   {len(splits['val'])} volumes")
    print(f"  Test:  {len(splits['test'])} volumes")
    
    # Save split for reproducibility
    split_file = OUTPUT_DIR / "volume_split.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'w') as f:
        for split_name, vols in splits.items():
            f.write(f"# {split_name}\n")
            for v in vols:
                f.write(f"{v}\n")
    print(f"\nSplit saved to {split_file}")
    
    # Extract patches
    print(f"\nExtracting patches ({args.patches_per_image} per train image)...")
    prepare_dataset_splits(
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        volumes_split=splits,
        patches_per_image=args.patches_per_image,
        seed=args.seed
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        lr_dir = OUTPUT_DIR / split / "lr"
        if lr_dir.exists():
            count = len(list(lr_dir.glob("*.png")))
            print(f"  {split}: {count} patches")

if __name__ == "__main__":
    main()

