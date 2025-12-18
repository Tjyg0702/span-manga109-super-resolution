# Download Model Checkpoints

Trained model checkpoints are too large for GitHub (1.9GB total).

## Pretrained Weights (Required for Training)

```bash
cd checkpoints

# SPAN (18MB)
wget https://github.com/Archaea0/SPAN/releases/download/v1.0/SPAN_4x.pth -O SPAN_4x_pretrained.pth

# SwinIR (65MB)
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth -O SwinIR_4x_pretrained.pth

# HAT (82MB)
wget "https://huggingface.co/Acly/hat/resolve/8403819bcbf5959d54c72383f0725f2525472d30/HAT_SRx4_ImageNet-pretrain.pth?download=true" -O HAT_4x_pretrained.pth
```

## Trained Checkpoints (Optional)

Fine-tuned models from our experiments are available on request.

**Results:**
- HAT fine-tuned: 23.47 dB
- SwinIR fine-tuned: 22.66 dB  
- SPAN Exp1: 22.56 dB

You can reproduce these by training with the configs in `configs/` directory.
