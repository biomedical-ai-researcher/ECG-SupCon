# A Hardware-Aware 1D-MobileViT Framework for Efficient Edge-Based ECG Arrhythmia Detection: A Simulation-Based Feasibility Study

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hOVnU72gPdJAdD8ccw0TriAKUNc7EKwv) 
[![Status](https://img.shields.io/badge/Status-Under_Editorial_and_Peer_Review-blue.svg)]()

This repository contains the official PyTorch implementation and hardware profiling code for the manuscript **"A Hardware-Aware 1D-MobileViT Framework for Efficient Edge-Based ECG Arrhythmia Detection: A Simulation-Based Feasibility Study"**, submitted to *Engineering_Science_and_Technology,_an_International_Journal*.

## 📌 Overview
Continuous ECG monitoring in ambulatory settings requires models that achieve high diagnostic performance while remaining computationally efficient for deployment on resource-constrained edge devices. We propose a lightweight 1D-MobileViT architecture that processes raw 1D ECG signals directly, eliminating the need for computationally expensive 2D time-frequency transformations (e.g., Continuous Wavelet Transform, CWT). The network combines convolutional layers for local feature extraction with lightweight self-attention mechanisms to capture long-range temporal dependencies.

To improve feature representation under clinically imbalanced data distributions, the framework incorporates Supervised Contrastive Learning (SupCon). In addition, hardware deployability is assessed through a simulation-based edge profiling methodology. While the acquisition pipeline is conceptually designed around a continuous Raspberry Pi Pico H microcontroller setup, inference performance is benchmarked using ONNX Runtime with a single-threaded ARM CPU configuration to emulate realistic mobile edge environments.

### ✨ Key Features

- **Lightweight Edge-Oriented Architecture**
  - Only **254.16K parameters (~0.25M)** and approximately **4.9M MACs**.

- **Direct 1D ECG Processing**
  - Operates directly on raw ECG signals without requiring computationally expensive 2D time-frequency representations (e.g., CWT).

- **Efficient Local-Global Feature Modeling**
  - Combines convolutional feature extraction with lightweight self-attention to capture both local morphology and long-range temporal dependencies.

- **Supervised Contrastive Learning**
  - Incorporates **SupCon** to improve feature representation and mitigate the effects of clinical class imbalance.

- **Strong Patient-Level Generalization**
  - Achieves a **mean overall accuracy of 87.80% ± 1.64** and a **mean weighted F1-score of 88.00% ± 1.00** on the **MIT-BIH Arrhythmia Database** under the strict unseen patient-specific **DS1/DS2** evaluation protocol (5 independent training runs).
  - Demonstrates strong sensitivity for detecting **Ventricular Ectopic Beats (VEBs)** in previously unseen patients.

- **Cross-Dataset Robustness**
  - Without retraining, achieves **88.00% accuracy** and a **weighted F1-score of 89.00%** on the **INCART** dataset.

- **Real-Time Edge Inference**
  - ONNX Runtime (Opset 18, single-threaded ARM CPU) profiling shows an average inference latency of approximately **1.5 ms per heartbeat**, supporting low-latency mobile edge deployment.

- **Open-Source & Reproducible**
  - Provides a complete end-to-end pipeline for dataset preparation, model training, evaluation, ONNX export, and hardware-aware edge profiling, enabling reproducible cardiovascular AI research.
    
## 🏗️ Architecture

The proposed architecture integrates local feature extraction via efficient 1D convolutions with global dependency modeling using lightweight attention mechanisms. 

<p align="center">
  <img src="proposed_pipeline.png" width="95%" alt="Pipeline Schematic">
  <br><em>Fig 1: Schematic overview of the proposed 1D-MobileViT-based framework, including signal acquisition concept and joint optimization via SupCon.</em>
</p>

## 🛠️ Project Structure & Data
To ensure a seamless reviewer experience and immediate reproducibility, both benchmark datasets (MIT-BIH & INCART) have been pre-processed according to the AAMI EC57 standard and securely cached. 

The provided "Google Colab Notebook" contains an automated smart data pipeline that downloads these arrays directly, avoiding the extensive processing time associated with raw PhysioNet extraction.

## 🚀 How to Run (1-Click Execution)
> **⚠️ Note for Reviewers:** 
> When executing the first cell (Module 1: Setup), Google Colab may display transient warnings or version conflict messages related to its pre-installed backend packages. Please **ignore these warnings**. They do not affect the execution of our 1D-MobileViT architecture. The code is designed to run end-to-end flawlessly.
The entire experimental pipeline—from data loading to training, external validation, and hardware profiling—is encapsulated in a single, well-documented Jupyter Notebook.

1. Click the **"Open In Colab"** badge at the top of this repository.
2. Run the cells sequentially.
3. The notebook will automatically:
   - Download the pre-processed ECG datasets.
   - Initialize the compact 1D-MobileViT backbone.
   - Train using the dual-loss (Focal + SupCon) strategy.
   - Generate publication-ready visualizations (Learning Curves, Confusion Matrix, t-SNE).
   - Perform strict cross-dataset evaluation on INCART.
   - Simulate Edge Hardware Latency using ONNX Runtime.

## 🖥️ Conceptual Hardware & Profiling
The clinical acquisition pipeline is conceptually designed around a continuous **Raspberry Pi Pico H** microcontroller interfaced with an **AD8232** sensor. 

To validate the inference feasibility of this setup prior to full-scale clinical hardware deployment, the codebase includes a rigorous hardware profiling module. The trained model is exported to **ONNX format (Opset 18)** and executed using strict single-threaded CPU execution to realistically emulate a modern smartphone edge node connected via USB OTG.

## 📝 Citation
If you find this code or methodology useful for your research, please cite our paper:
```bibtex
@article{1DMobileViT_JESTECH,
  title={A Hardware-Aware 1D-MobileViT Framework for Efficient Edge-Based ECG Arrhythmia Detection},
  author={Anonymous for Review},
  journal={Engineering Science and Technology, an International Journal},
  year={2026}
}
