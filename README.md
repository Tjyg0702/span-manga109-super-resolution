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
