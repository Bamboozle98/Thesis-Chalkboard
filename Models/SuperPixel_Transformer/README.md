# SuperPixel_Transformer

This directory contains the main source code for the Superpatcher Transformer project.  
It is organized into modular components including dataloaders, model pipelines, CNN backbones, superpixel algorithms, and transformer definitions.  

## Directory Overview

### DataLoaders/
Implements dataset loaders for both natural image datasets and histopathology whole slide images (WSIs).  
- **Camelyon_16/** – Custom dataloaders for the Camelyon16 WSI dataset.  
- **NaturalDataLoaders/** – General dataloaders for natural image datasets such as Oxford Pets, ImageNet subsets, and High-Res datasets. PE denotes the use of Positional and Covariance encodings. 

### Model_Pipelines/
High-level training and evaluation pipelines.  
- **Large_Scale_Imagery/** – Pipelines for large WSI datasets, including Camelyon and Gerhardt. Handles preprocessing, patch extraction, and superpixel mapping.  
- **StandardDatasets/** – PyTorch Lightning training modules (`Lightning_SPFormer`, `Lightning_SPFormerPE`, etc.) for standard benchmark datasets. PE denotes the use of Positional and Covariance encodings.  

### PNP_CNNs/
Backbone CNNs and lightweight convolutional networks used for feature extraction from superpixels.  
- **Resnet18.py / Resnet50.py / Resnet101.py** – ResNet variants adapted for superpixel feature extraction.  
- **customResnet.py / MiniCNN.py** – Custom CNN architectures for layer testing.  
- **Camelyon_Resnet_18.py** – ResNet specifically adapted for Camelyon WSI tasks.  

### Superpixel_Algorithms/
Implementations of superpixel segmentation methods.  
- **SLIC.py** – Standard SLIC superpixel generation for natural images.  
- **WSI_SLIC.py** – WSI-specific adaptation of SLIC for large-scale histopathology slides.  

### Transformer/
Transformer encoder implementations.  
- **Transformer.py** – Core transformer model used to process superpixel feature vectors as tokens for classification or regression.  

## Notes
- All modules are designed to work together in the Superpatcher pipeline.  
- PyTorch Lightning is used for training management in `Model_Pipelines/`.  
- CNN backbones in `PNP_CNNs/` provide feature extraction for both natural and WSI datasets.  
- Superpixel generation (`Superpixel_Algorithms/`) only uses SLIC.  
