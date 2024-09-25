import torch
import torchvision.models as models
import torch.nn as nn


def transformer_model(class_names, num_superpixel_features):
    """
    Load a pre-trained Vision Transformer model from torchvision and modify the classifier head.

    Parameters:
    - class_names (list): List of class names in your dataset.
    - num_superpixel_features (int): The number of features per superpixel.

    Returns:
    - model: The modified Vision Transformer model.
    """
    # Load a pre-trained Vision Transformer model from torchvision
    model = models.vit_b_16(weights='DEFAULT')  # Load pre-trained ViT

    # Modify the classifier head to fit your number of classes (for Oxford Pets)
    num_classes = len(class_names)  # Number of classes in your dataset
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Modify the input layer to accept superpixel feature vectors
    model.embeddings.patch_embeddings.proj = nn.Linear(num_superpixel_features,
                                                       model.embeddings.patch_embeddings.proj.out_features)

    return model

