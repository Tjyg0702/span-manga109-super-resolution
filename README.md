# Manga Super-Resolution

Fine-tuning SPAN, SwinIR, and HAT models on Manga109 dataset.

## Setup

```bash
# Clone and setup
git clone https://github.com/Tjyg0702/span-manga109-super-resolution
cd span-manga109-super-resolution
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1Bb66z3vIM4n4STi2okji6mASpToyqCmz" -O manga109.zip
unzip manga109.zip -d data/
```

## Download Pretrained Weights

```bash
cd checkpoints

# SPAN (DF2K pretrained)
wget https://github.com/Archaea0/SPAN/releases/download/v1.0/SPAN_4x.pth -O SPAN_4x_pretrained.pth

# SwinIR (DF2K pretrained)
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth -O SwinIR_4x_pretrained.pth

# HAT (ImageNet+DF2K pretrained)
wget "https://huggingface.co/Acly/hat/resolve/8403819bcbf5959d54c72383f0725f2525472d30/HAT_SRx4_ImageNet-pretrain.pth?download=true" -O HAT_4x_pretrained.pth
```

## Training

```bash
# SPAN
python scripts/train.py --config configs/exp1_baseline_lr2e4.yaml

# SwinIR  
python scripts/train.py --config configs/swinir_finetuned.yaml

# HAT
python scripts/train.py --config configs/hat_finetuned.yaml
```

## Evaluation

```bash
python scripts/evaluate_models.py
```

## Results

Best model: **HAT fine-tuned - 23.47 dB PSNR** on Manga109 test set.

See training curves and comparisons in `outputs/` directory.
