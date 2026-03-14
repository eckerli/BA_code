# Generative Adversarial Networks (GANs) in finance: an overview

This code is referenced in the paper https://arxiv.org/abs/2106.06364.

It explores the use of Generative Adversarial Networks (GANs) for synthetic financial time series generation.

## Overview

The notebook (`gans_in_finance.ipynb`) implements and compares 10 GAN variants applied to real financial data (loaded via `yfinance`):

- **DCGAN** — standard deep convolutional GAN
- **LSGAN** — least squares loss variant
- **SAGAN** — self-attention GAN
- **WGAN** — Wasserstein GAN
- **WGAN-GP** — Wasserstein GAN with gradient penalty
- **DRAGAN** — deep regret analytic GAN
- **RaGAN / RaLSGAN** — relativistic loss improvements
- **YLgan** — further loss variant
- **BigGAN / BigGAN-deep** — large-scale GAN architectures

Each model is trained on 1D financial time series windows and evaluated on its ability to generate realistic synthetic data. Originally developed for Google Colab, adapted to run locally with TensorFlow.

## Requirements

- Python 3.12
- A virtual environment (recommended)

## Setup

### 1. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install \
  "tensorflow==2.16.2" \
  "numpy<2" \
  "yfinance==1.2.0" \
  "scipy==1.17.1" \
  "pandas==3.0.1" \
  "matplotlib==3.10.8" \
  "ipykernel==7.2.0"
```

> **Important:** NumPy must be pinned to `<2`. TensorFlow 2.16.2 was compiled against NumPy 1.x and will fail to import with NumPy 2.x (`numpy.core` was removed in NumPy 2.0).

### 3. Register the kernel with Jupyter

```bash
python -m ipykernel install --user --name=ba_code --display-name "BA Code (.venv)"
```

### 4. Launch the notebook

```bash
jupyter notebook gans_in_finance.ipynb
# or
jupyter lab gans_in_finance.ipynb
```

Select the **"BA Code (.venv)"** kernel when prompted.

## Known Issues

- **NumPy 2.x incompatibility**: TensorFlow 2.16.2 requires `numpy<2`. If you see `ImportError: numpy.core.umath failed to import`, run `pip install "numpy<2"` to downgrade.
- **yfinance API changes**: Data download uses `yfinance`. If the Yahoo Finance API changes, the `download_data()` function in the notebook may need adjustment.
