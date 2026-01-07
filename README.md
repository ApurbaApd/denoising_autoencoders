# Medical MRI Denoising with Deep Learning

A comprehensive comparative analysis of Autoencoder architectures for denoising Medical Images (MRI). This project benchmarks **CAE**, **U-Net**, **VAE**, and **ResNet-AE** on a dataset of 7,000 samples.

## 📊 Models Implemented
| Model | Type | Best For |
| :--- | :--- | :--- |
| **CAE** | Standard Conv AE | Baseline performance |
| **U-Net** | Skip Connections | High-frequency detail (edges) |
| **VAE** | Probabilistic | Regularization & Latent space analysis |
| **ResNet-AE** | Residual Blocks | Deep feature extraction |


\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}

\begin{document}

\section{Mathematical Formulation of Autoencoders}

\subsection{Problem Statement}
Given a clean medical image $x \in \mathbb{R}^{H \times W}$ and an additive noise distribution $\eta \sim \mathcal{N}(0, \sigma^2)$, the observed noisy image is defined as $\tilde{x} = x + \eta$. The objective of the autoencoder $f_\theta$ is to reconstruct the clean image $\hat{x}$ such that the Mean Squared Error (MSE) is minimized:

\begin{equation}
    \mathcal{L}_{MSE}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \| x_i - f_\theta(\tilde{x}_i) \|_2^2
\end{equation}

\subsection{Convolutional Autoencoder (CAE)}
The CAE learns spatial features using sequential convolution operations ($*$) followed by a non-linear activation function $\sigma$ (typically ReLU).

\noindent \textbf{Encoder:}
\begin{equation}
    h = \sigma(W_e * \tilde{x} + b_e)
\end{equation}

\noindent \textbf{Decoder:}
The decoder utilizes transposed convolutions (denoted as $*^T$) to upsample the latent representation back to the original spatial dimensions:
\begin{equation}
    \hat{x} = \sigma'(W_d *^T h + b_d)
\end{equation}
where $W$ and $b$ represent the learnable weights and biases, respectively.

\subsection{U-Net (Skip Connections)}
The U-Net architecture addresses the loss of high-frequency spatial details during downsampling. Let $E_l$ denote the feature map at the $l$-th encoder layer and $D_l$ denote the feature map at the corresponding decoder layer. The U-Net utilizes skip connections via channel-wise concatenation $[\cdot, \cdot]$:

\begin{equation}
    D_l = \text{Conv}\left( \left[ \text{UpSample}(D_{l+1}), E_l \right] \right)
\end{equation}
This allows gradients and spatial information to flow directly from the encoder to the decoder.

\subsection{Variational Autoencoder (VAE)}
The VAE models the data generation process probabilistically. The encoder approximates the posterior $q_\phi(z|\tilde{x})$ and the decoder models the likelihood $p_\theta(x|z)$.

\noindent \textbf{Reparameterization Trick:}
To enable backpropagation through the stochastic latent layer, the variable $z$ is sampled as:
\begin{equation}
    z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
\end{equation}

\noindent \textbf{Optimization Objective (ELBO):}
The loss function minimizes the reconstruction error while regularizing the latent space using the Kullback-Leibler (KL) divergence:
\begin{equation}
    \mathcal{L}_{VAE} = \| x - \hat{x} \|^2 + \beta \cdot D_{KL}\Big( \mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, 1) \Big)
\end{equation}
where the KL divergence term is expanded as:
\begin{equation}
    D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)
\end{equation}

\subsection{Residual Autoencoder (ResNet-AE)}
To mitigate the vanishing gradient problem in deep networks, this architecture employs Residual Blocks. Instead of learning the direct mapping $H(x)$, the layers approximate the residual function $F(x) := H(x) - x$.

\noindent \textbf{Residual Block Formulation:}
\begin{equation}
    y_l = \sigma \Big( \mathcal{F}(x_l, \{W_l\}) + x_l \Big)
\end{equation}
Here, $x_l$ is the identity shortcut connection, and $\mathcal{F}$ represents the stack of convolutional layers within the block.

\end{document}

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt