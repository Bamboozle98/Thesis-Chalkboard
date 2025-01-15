import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, ground_truth_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".JPEG")
        ])  # Sort to ensure the order matches the ground truth file

        if ground_truth_file:
            with open(ground_truth_file, "r") as f:
                self.labels = [int(line.strip()) for line in f.readlines()]

        else:
            raise ValueError("Ground truth file is required for validation dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# Usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Initialize validation dataset
val_dataset = ImageNetDataset(
    root_dir=r"E:\ImageNet\unpacked\validate",
    ground_truth_file=r"E:\ImageNet\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt",
    transform=transform,
)

# DataLoader
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Debugging
for images, labels in val_loader:
    print("Image Batch Shape:", images.shape)
    print("Labels in Batch:", labels)
    break
