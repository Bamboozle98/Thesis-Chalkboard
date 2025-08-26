# Comparison Models

This directory contains baseline and reference models used for comparison against the Superpatcher Transformer.  
Each subfolder corresponds to a dataset (High-Res, ImageNet, OxfordPets) and includes implementations of standard CNNs and Vision Transformers for benchmarking.

## Structure

### High_Res
- **Data_Loader_HighRes.py** – Dataloader for the custom High-Resolution Bird-Cat-Dog dataset.  
- **Resnet18_HighRes.py / Resnet50_HighRes.py / Resnet101_HighRes.py** – Standard ResNet architectures (pretrained on ImageNet, fine-tuned on the High-Res dataset).  
- **16x16Transformer_HighRes.py** – Vision Transformer baseline using 16×16 patches as tokens.  

### ImageNet
- **Data_Loader_ImageNet.py** – Dataloader for the ImageNet subset used in experiments.  
- **Resnet18_ImageNet.py / Resnet50_ImageNet.py / Resnet101_ImageNet.py** – ResNet baselines evaluated on ImageNet.  
- **16x16Transformer_ImageNet.py** – Vision Transformer baseline for ImageNet classification.  

### OxfordPets
- **Data_Loader_OxfordPets.py** – Dataloader for the Oxford-IIIT Pets dataset.  
- **Resnet18_OxfordPets.py / Resnet50_OxfordPets.py / Resnet101_OxfordPets.py** – ResNet baselines for OxfordPets classification(fine-tuned).  
- **16x16Transformer_OxfordPets.py** – Vision Transformer baseline for OxfordPets classification.  

## Purpose
These comparison models serve as **benchmarks** to evaluate the performance of the proposed Superpatcher Transformer. By training and testing standard CNNs (ResNet18/50/101) and a conventional Vision Transformer across multiple datasets, we establish performance baselines for accuracy, scalability, and training efficiency.  

The results from these models are used to highlight the strengths and weaknesses of the Superpatcher methodology relative to conventional architectures.
