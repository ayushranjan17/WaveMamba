# WaveMamba: Hyperspectral Image Classification using CNN-Enhanced Multi-Level Wavelet Features Fusion-based Residual Mamba Network

## Overview
This project introduces **WaveMamba**, a novel approach for Hyperspectral Image Classification (HSIC). The key innovation lies in combining **wavelet transforms** with **CNNs** and the **Mamba state space model** to improve both computational efficiency and accuracy in HSIC tasks. The model is designed to capture both spatial and spectral dependencies in hyperspectral data by utilizing **multi-level wavelet decomposition** and CNNs for spectral-spatial feature extraction.

## Key Features
- **Daubechies Wavelet CNN**: This component extracts spectral and spatial features using a four-level discrete wavelet transform (DWT) and CNN layers.
- **Hyperspectral Residual Mamba (HRM) Block**: Incorporates Scalable Self-Attention Mechanisms (SSMs) to handle long-range dependencies and enhance classification performance.
- **Memory-Efficient**: The model is lightweight and efficient, requiring less data and computational resources.
- **High Accuracy**: Outperforms traditional CNN and Transformer-based methods on popular hyperspectral datasets.

## Datasets Used
1. **Pavia University**: A 610 × 340 pixel dataset with 103 spectral bands, featuring nine land-cover classes.
2. **Salinas**: A 512 × 217 pixel dataset with 224 spectral bands, containing 16 classes of agricultural land-cover types.
3. **Houston**: A dataset with 144 spectral bands and 15 land-cover classes, captured around the University of Houston.

## Model Architecture
The WaveMamba model architecture is composed of the following:
- **Factor Analysis (FA)** for dimensionality reduction.
- **Daubechies Wavelet CNN** for spectral-spatial feature extraction.
- **HRM Block** for improving classification accuracy through attention-based mechanisms and residual connections.
- **Softmax Layer** for final classification output.

## Evaluation Metrics
- **Overall Accuracy (OA)**: Measures the percentage of correctly classified samples.
- **Average Accuracy (AA)**: Average of per-class accuracies.
- **Kappa Coefficient (K)**: A measure of classification accuracy relative to random chance.

## Experimental Results
The model was evaluated using the Pavia University, Salinas, and Houston datasets. Key findings include:
- Optimal learning rate: **0.01**
- Optimal patch size: **12**
- The model outperformed state-of-the-art methods like 3DCNN, CS2DT, and ViTs.

## Ablation Study
The ablation study showed that removing any of the core modules—Wavelet Transform, CNN, or HRM Block—resulted in decreased performance, demonstrating the importance of each component in the WaveMamba architecture.

[Download the dataset](https://drive.google.com/drive/folders/1n1vyY9RoiwI6Be4NyGUx2HflwX_gm9zh?usp=sharing)

