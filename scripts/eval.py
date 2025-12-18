

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.span import SPAN, count_parameters
from src.data.dataset import Manga109Dataset
from src.data.preprocessing import image_to_tensor, tensor_to_image, create_lr_image
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.utils.visualization import visualize_comparison, create_comparison_grid

PROJECT_ROOT = Path(__file__).parent.parent

def get_test_volumes(seed: int = 42):
    """Get test volume names."""
    import random
    images_dir = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
    volumes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    
    random.seed(seed)
    shuffled = volumes.copy()
    random.shuffle(shuffled)
    return sorted(shuffled[:19])

def evaluate_model(model, test_loader, device):
    """Run evaluation, return metrics."""
    model.eval()
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(test_loader, desc="Evaluating"):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            sr_batch = model(lr_batch)
            
            for i in range(sr_batch.shape[0]):
                psnr_list.append(calculate_psnr(sr_batch[i], hr_batch[i]))
                ssim_list.append(calculate_ssim(sr_batch[i], hr_batch[i]))
    
    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list)
    }

def generate_samples(model, images_dir, volumes, device, output_dir, num_samples=6):
    """Generate visual comparison samples."""
    import random
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model.eval()
    samples = []
    
    # Pick random images from test volumes
    random.seed(123)
    selected = []
    for vol in random.sample(volumes, min(num_samples, len(volumes))):
        vol_dir = images_dir / vol
        img_paths = list(vol_dir.glob("*.jpg"))
        if img_paths:
            selected.append((vol, random.choice(img_paths)))
    
    for vol_name, img_path in selected:
        hr_img = Image.open(img_path).convert('RGB')
        
        # Crop a 192x192 patch
        w, h = hr_img.size
        left = (w - 192) // 2
        top = (h - 192) // 2
        hr_patch = hr_img.crop((left, top, left + 192, top + 192))
        
        # Create LR
        lr_patch = create_lr_image(hr_patch)
        
        # Bicubic upscale
        bicubic = lr_patch.resize((192, 192), Image.BICUBIC)
        
        # Model inference
        lr_tensor = image_to_tensor(lr_patch).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        hr_tensor = image_to_tensor(hr_patch)
        bic_tensor = image_to_tensor(bicubic)
        
        # Calculate metrics
        sr_psnr = calculate_psnr(sr_tensor, hr_tensor)
        bic_psnr = calculate_psnr(bic_tensor, hr_tensor)
        
        # Save individual comparison
        visualize_comparison(
            lr=lr_tensor,
            sr=sr_tensor.squeeze(0),
            hr=hr_tensor,
            bicubic=bic_tensor,
            save_path=output_dir / f"comparison_{vol_name}.png",
            title=f"{vol_name}",
            metrics={'sr_psnr': sr_psnr}
        )
        
        samples.append({
            'lr': lr_tensor,
            'sr': sr_tensor.squeeze(0),
            'hr': hr_tensor,
            'name': f"{vol_name} (PSNR: {sr_psnr:.2f})"
        })
    
    # Create grid
    if samples:
        create_comparison_grid(
            samples,
            save_path=output_dir / "comparison_grid.png",
            title="SPAN Super-Resolution Results"
        )
    
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true',
                        help='Mark as baseline (pretrained, not fine-tuned)')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of visual samples to generate')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPAN EVALUATION ON MANGA109")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = SPAN(feature_channels=48, upscale=4)
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_pretrained(str(checkpoint_path), strict=False)
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Data
    images_dir = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
    test_volumes = get_test_volumes()
    print(f"\nTest volumes: {len(test_volumes)}")
    
    test_dataset = Manga109Dataset(
        images_dir=images_dir,
        volumes=test_volumes,
        patches_per_epoch=500,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_model(model, test_loader, device)
    
    label = "Pretrained (Baseline)" if args.baseline else "Fine-tuned"
    print(f"\n{label} Results:")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} visual samples...")
    output_dir = PROJECT_ROOT / "outputs" / ("baseline" if args.baseline else "finetuned")
    generate_samples(model, images_dir, test_volumes, device, output_dir, args.num_samples)
    print(f"Samples saved to {output_dir}")
    
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()

