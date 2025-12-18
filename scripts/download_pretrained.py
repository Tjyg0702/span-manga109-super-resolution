#!/usr/bin/env python3
"""
Download pretrained SPAN weights from official release.
Weights trained on DF2K (DIV2K + Flickr2K).
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Official SPAN x4 weights
PRETRAINED_URLS = {
    'span_x4': {
        'url': 'https://github.com/hongyuanyu/SPAN/releases/download/v1.0.0/SPAN_4x.pth',
        'filename': 'SPAN_4x_pretrained.pth',
    }
}

def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download file with progress indicator."""
    print(f"Downloading from {url}")
    print(f"Saving to {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size:
                    pct = downloaded / total_size * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end='')
    
    print(f"\n✓ Download complete: {dest}")

def main():
    print("=" * 60)
    print("DOWNLOADING PRETRAINED SPAN WEIGHTS")
    print("=" * 60)
    
    for name, info in PRETRAINED_URLS.items():
        dest = CHECKPOINT_DIR / info['filename']
        
        if dest.exists():
            print(f"\n{info['filename']} already exists, skipping.")
            continue
        
        try:
            download_file(info['url'], dest)
        except Exception as e:
            print(f"\n⚠ Failed to download {name}: {e}")
            print("You may need to download manually from:")
            print(f"  {info['url']}")
    
    print("\n✓ Pretrained weights ready!")

if __name__ == "__main__":
    main()

