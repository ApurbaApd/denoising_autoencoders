# Medical MRI Denoising with Deep Learning

A comprehensive comparative analysis of Autoencoder architectures for denoising Medical Images (MRI). This project benchmarks **CAE**, **U-Net**, **VAE**, and **ResNet-AE** on a dataset of 7,000 samples.

## 📊 Models Implemented
| Model | Type | Best For |
| :--- | :--- | :--- |
| **CAE** | Standard Conv AE | Baseline performance |
| **U-Net** | Skip Connections | High-frequency detail (edges) |
| **VAE** | Probabilistic | Regularization & Latent space analysis |
| **ResNet-AE** | Residual Blocks | Deep feature extraction |


## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt