# Manga Super-Resolution with SPAN (×4)

This repository contains the final project for a Deep Learning course, focusing on single-image super-resolution for manga images using the SPAN architecture. The project evaluates the impact of random initialization, pretraining, and domain-specific fine-tuning on the Manga109 dataset.

---

## Project Overview

Manga images contain strong line structures and high-frequency details that are difficult to reconstruct using traditional interpolation methods. In this project, we implement and evaluate the SPAN super-resolution model and compare its performance against bicubic interpolation under multiple training configurations.

The goal is not to achieve state-of-the-art performance, but to analyze how training strategies affect reconstruction quality in a controlled academic setting.

---

## Dataset

- **Manga109**
- 109 manga volumes
- Split into train / validation / test at the volume level
- Super-resolution scale: ×4
- Low-resolution images generated via bicubic downsampling

> Note: The Manga109 dataset is not included in this repository due to licensing restrictions.

---

## Methods Compared

- **Bicubic Interpolation**
- **SPAN (Random Initialization)**
- **SPAN (Pretrained)**
- **SPAN (Fine-tuned on Manga109)**

---

## Quantitative Results (Manga109, ×4)

| Method              | PSNR (dB) | SSIM  |
|---------------------|-----------|-------|
| SPAN (random)       | 2.43      | 0.0150 |
| SPAN (pretrained)   | 6.39      | 0.3312 |
| Bicubic             | 17.69     | 0.6519 |
| SPAN (fine-tuned)   | **18.26** | **0.6987** |

The fine-tuned SPAN model achieves the best performance, surpassing bicubic interpolation in both PSNR and SSIM.

---

## Qualitative Results

Visual comparisons demonstrate that:
- Bicubic interpolation produces smooth but blurry edges.
- Randomly initialized SPAN generates severe artifacts.
- Pretrained SPAN improves stability but lacks domain adaptation.
- Fine-tuned SPAN best reconstructs sharp manga strokes and contours.

---

## Implementation Details

- Framework: PyTorch
- Evaluation metrics: PSNR, SSIM
- Hardware: NVIDIA GPU (Google Colab)
- Training performed on cropped HR/LR patches

---

## How to Run

1. Open the notebook in `notebooks/`
2. Set up the environment (Google Colab recommended)
3. Download and prepare Manga109
4. Run cells sequentially for training and evaluation

---

## Limitations and Future Work

- Limited training time and compute resources
- No perceptual or adversarial losses
- Future work could explore larger-scale training, perceptual loss functions, or transformer-based SR models

---

## Acknowledgements

- SPAN architecture inspired by the original SPAN paper
- Manga109 dataset provided by the University of Tokyo

---

## Author

Ziyang Deng，   Shashank Patoju，  Bharath Kumar Kakumani
NYU – Deep Learning Final Project
