import torch
from torchvision.models import resnet18, ResNet18_Weights
from Data_Loader_IN import val_loader

# Load pretrained ResNet-18
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test accuracy
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for faster inference
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predictions

        total += labels.size(0)  # Total number of samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

accuracy = 100 * correct / total
print(f"Accuracy on the validation set: {accuracy:.2f}%")

