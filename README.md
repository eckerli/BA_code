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

## Setup

```bash
source venv/bin/activate
pip install tensorflow yfinance matplotlib numpy
```