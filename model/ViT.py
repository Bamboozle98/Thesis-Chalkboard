import torch
import torchvision.models as models
import torch.nn as nn
from Data_Loader import data_process


def transformer_model(class_names):
    # Load a pre-trained Vision Transformer model from torchvision
    model = models.vit_b_16(weights='DEFAULT')  # Load pre-trained ViT

    # Modify the classifier head to fit your number of classes (for Oxford Pets)
    num_classes = len(class_names)  # Number of classes in your dataset
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
