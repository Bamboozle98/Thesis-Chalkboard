import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Models.SuperPixel_Transformer.config import imagenet_val_dir, imagenet_train_dir

train_root_dir = imagenet_train_dir
val_root_dir = imagenet_val_dir

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

train_data = torchvision.datasets.ImageFolder(train_root_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)

val_data = torchvision.datasets.ImageFolder(val_root_dir, transform=transform)
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)


