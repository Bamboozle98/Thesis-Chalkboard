import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from Data_Loader_ImageNet import train_loader, val_loader
from tqdm import tqdm
import os


def main():
    # Load Vision Transformer (ViT-B/16) with pretrained weights
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.eval()  # Set Model to evaluation mode

    # Move the Model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function for evaluation
    criterion = nn.CrossEntropyLoss()

    # Validation phase
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validating Pretrained ViT-B/16")

        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print("\nValidation Results:")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
