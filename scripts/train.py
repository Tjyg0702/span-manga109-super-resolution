

import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.span import SPAN
from src.models.swinir_official import SwinIR
from src.models.hat_official import HAT
from src.data.dataset import Manga109Dataset
from src.training.trainer import Trainer
from src.training.losses import L1Loss, CombinedLoss
from src.utils.visualization import plot_training_curves

PROJECT_ROOT = Path(__file__).parent.parent

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_volume_split(seed: int = 42):
    """Generate train/val/test split by volume."""
    import random
    images_dir = PROJECT_ROOT / "data" / "Manga109_released_2023_12_07" / "images"
    volumes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    
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
    parser.add_argument('--config', type=str, default='configs/span_manga.yaml')
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(PROJECT_ROOT / args.config)
    exp_name = config.get('name', 'default')
    
    print("=" * 60)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Model - support SPAN, SwinIR, HAT
    model_name = config['model'].get('name', 'span').lower()
    
    if model_name == 'span':
        model = SPAN(
            feature_channels=config['model']['num_features'],
            upscale=config['model']['upscale']
        )
        print(f"SPAN: {count_parameters(model):,} parameters")
        
        # Load pretrained for SPAN
        pretrained_path = args.pretrained or (PROJECT_ROOT / config.get('pretrained', {}).get('path', ''))
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained: {pretrained_path}")
            model.load_pretrained(str(pretrained_path), strict=False)
    
    elif model_name == 'swinir':
        model = SwinIR(
            upscale=config['model']['scale'],
            in_chans=3,
            img_size=config['data']['lr_patch_size'],
            window_size=config['model']['window_size'],
            img_range=config['model']['img_range'],
            depths=config['model']['depths'],
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            upsampler=config['model']['upsampler'],
            resi_connection='1conv'
        )
        print(f"SwinIR: {count_parameters(model):,} parameters ({count_parameters(model)/1e6:.2f}M)")
        
        # Load pretrained weights if specified
        pretrained_path = config.get('pretrained', {}).get('path', '')
        if pretrained_path and Path(PROJECT_ROOT / pretrained_path).exists():
            print(f"Loading pretrained SwinIR: {pretrained_path}")
            state_dict = torch.load(PROJECT_ROOT / pretrained_path, map_location='cpu', weights_only=False)
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded")
    
    elif model_name == 'hat':
        model = HAT(
            upscale=config['model']['scale'],
            in_chans=3,
            img_size=config['data']['lr_patch_size'],
            window_size=config['model']['window_size'],
            compress_ratio=config['model'].get('compress_ratio', 3),
            squeeze_factor=config['model'].get('squeeze_factor', 30),
            conv_scale=config['model'].get('conv_scale', 0.01),
            overlap_ratio=config['model'].get('overlap_ratio', 0.5),
            img_range=config['model']['img_range'],
            depths=config['model']['depths'],
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            upsampler=config['model']['upsampler'],
            resi_connection=config['model'].get('resi_connection', '1conv')
        )
        print(f"HAT: {count_parameters(model):,} parameters ({count_parameters(model)/1e6:.2f}M)")
        
        # Load pretrained weights if specified
        pretrained_path = config.get('pretrained', {}).get('path', '')
        if pretrained_path and Path(PROJECT_ROOT / pretrained_path).exists():
            print(f"Loading pretrained HAT: {pretrained_path}")
            state_dict = torch.load(PROJECT_ROOT / pretrained_path, map_location='cpu', weights_only=False)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k and 'relative_position_index' not in k}
            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Data
    splits = get_volume_split(config['data']['seed'])
    images_dir = PROJECT_ROOT / config['data']['manga109_dir']
    
    train_dataset = Manga109Dataset(
        images_dir=images_dir,
        volumes=splits['train'],
        patches_per_epoch=1600,
        augment=True
    )
    val_dataset = Manga109Dataset(
        images_dir=images_dir,
        volumes=splits['val'],
        patches_per_epoch=200,
        augment=False
    )
    
    print(f"Train: {len(train_dataset)} patches/epoch")
    print(f"Val: {len(val_dataset)} patches")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Loss function
    use_perceptual = config.get('loss', {}).get('use_perceptual', False)
    perceptual_weight = config.get('loss', {}).get('perceptual_weight', 0.1)
    
    if use_perceptual:
        print(f"Loss: L1 + Perceptual (weight={perceptual_weight})")
        criterion = CombinedLoss(use_perceptual=True, perceptual_weight=perceptual_weight)
    else:
        print("Loss: L1")
        criterion = L1Loss()
    
    # Training config
    print(f"\nTraining config:")
    print(f"  LR: {config['training']['lr']} → {config['training']['min_lr']}")
    print(f"  Batch: {config['training']['batch_size']}")
    print(f"  Max iters: {config['training']['max_iters']}")
    print(f"  Patience: {config['training']['patience']}")
    
    # Create output directories
    checkpoint_dir = PROJECT_ROOT / config['output']['checkpoint_dir']
    results_dir = PROJECT_ROOT / config['output']['results_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Trainer with custom loss
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config['training']['lr'],
        min_lr=config['training']['min_lr'],
        weight_decay=config['training']['weight_decay'],
        max_iters=config['training']['max_iters'],
        val_interval=config['training']['val_interval'],
        patience=config['training']['patience'],
        checkpoint_dir=checkpoint_dir,
    )
    
    if use_perceptual:
        trainer.criterion = criterion.to(device)
    
    print("\nStarting training...")
    history = trainer.train()
    
    # Save curves
    plot_training_curves(
        history['train_losses'],
        history['val_psnrs'],
        save_path=results_dir / 'training_curves.png'
    )
    
    # Save summary
    summary_file = results_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Best PSNR: {history['best_psnr']:.2f} dB\n")
        f.write(f"Training time: {history['elapsed_minutes']:.1f} min\n")
        f.write(f"Config: {args.config}\n")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {exp_name}")
    print(f"{'='*60}")
    print(f"Best PSNR: {history['best_psnr']:.2f} dB")
    print(f"Time: {history['elapsed_minutes']:.1f} min")
    print(f"Checkpoint: {checkpoint_dir / 'best.pth'}")

if __name__ == "__main__":
    main()
