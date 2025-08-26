# Thesis-Superpatcher
Research and Development of Superpatcher Transformers for Image and Whole Slide Image (WSI) Analysis.  
This repository contains the experimental pipelines, training code, and model iterations developed as part of my thesis at ECU. The framework builds on superpixel-based tokenization for transformers, supporting both standard image datasets and large-scale WSI pipelines.  

## Overview
Superpatcher leverages **SLIC superpixel segmentation** to convert images into structured sets of tokens. Each superpixel is vectorized through a CNN backbone, aggregated, and then passed into a Transformer encoder for classification. This approach provides a middle ground between the **local feature focus of CNNs** and the **global context modeling of Transformers**.  

Two primary pipelines are developed:
1. **Standard-Superpatcher** – designed for benchmark datasets like Oxford Pets, ImageNet subsets, and custom high-resolution datasets.  
2. **WSI-Superpatcher** – tailored for histopathology data (e.g., Camelyon16/17), capable of handling extremely large slides by segmenting tissue regions into superpixels, extracting patches, and encoding them as transformer inputs.  

## Major Goals
- Develop a scalable **superpixel-to-token pipeline** for transformer-based image classification.  
- Compare **baseline CNNs, Vision Transformers, and Superpatcher models** across multiple datasets.  
- Investigate the role of **positional encodings** (centroid-based) and **covariance embeddings** in improving superpixel transformer performance.  
- Demonstrate that superpixel-driven tokenization provides **computationally efficient** yet **semantically meaningful** representations for both natural and medical images.  

## Setup

This project requires **CUDA 12.4** and PyTorch with GPU acceleration.  
You can download and install CUDA 12.4 here:  
[CUDA 12.4 (Windows x86_64)](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)  

## Installation


### 1. Create and activate a virtual environment (example with conda)
conda create -n sptransformer python=3.10
conda activate sptransformer

### 2. Install PyTorch and TorchVision compiled for CUDA 12.4
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124

### 3. Install remaining dependencies
pip install -r requirements.txt


## Methodology

### Superpixel Generation
- **SLIC (Simple Linear Iterative Clustering):** Used across all datasets for consistent and interpretable superpixel segmentation.  
- **WSI Preprocessing:** Includes tissue extraction, Otsu-thresholding at thumbnail resolution, coordinate mapping to high-resolution space, and superpixel segmentation at relevant pyramid levels.  

### Feature Extraction
- Images are passed through **CNN backbones** (ResNet18/50/101) to obtain feature maps.  
- Superpixel maps guide aggregation: features corresponding to each superpixel are averaged or upsampled into a fixed-length vector.  

### Transformer Encoding
- Superpixel vectors are treated as tokens.  
- Optional **positional encodings** (based on superpixel centroids) and **covariance embeddings** are added.  
- Tokens are processed through Transformer encoder layers for classification or regression tasks.  

## Benchmarking & Evaluation
- **Datasets:** Oxford Pets, ImageNet subsets, Custom High-Res Bird-Cat-Dog dataset, Camelyon-16/17 WSIs.  
- **Baselines:** Pre-trained ResNet variants and Vision Transformers at multiple input resolutions (224, 512, 1024).  
- **Superpatcher Variants:**  
  - Without positional/covariance embeddings  
  - With positional/covariance embeddings  

Evaluation is performed using classification accuracy (for natural image datasets) and regression/classification scores (for WSIs).  

## Repository Layout (current)
- `Data/` – Directory for different datasets. GitHub only contains Oxford Pets You will have to get the other datasets and provide the correct paths in the config to run the models.   
- `Models/` – CNN backbones, Superpixel transformers, embedding modules, and their Results.  
- `misc/` – A series of scripts that were used to generate visualizations and test ides.

## Status
The repository is under active development and will continue to evolve as experiments are finalized for thesis defense. Future work may include:  
- Expanding beyond classification to segmentation tasks.  
- Exploring alternative superpixel algorithms and hybrid graph-based representations.  
- Optimizing for multi-GPU distributed training on WSI-scale data.  
