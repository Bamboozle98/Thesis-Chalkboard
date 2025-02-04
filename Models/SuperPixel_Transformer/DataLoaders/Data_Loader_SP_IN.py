import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from Models.SuperPixel_Transformer.Superpixel_Algorithms.SLIC import create_superpixel_image


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", val_labels_file=None, num_segments=50):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.n_segments = num_segments

        if split == "train":
            # Get image paths and extract labels from filenames
            self.image_paths = [
                os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".JPEG")
            ]
            self.labels = [f.split("_")[0] for f in os.listdir(root_dir) if f.endswith(".JPEG")]
        elif split == "val":
            # Read validation ground truth labels
            if not val_labels_file:
                raise ValueError("val_labels_file must be provided for validation split.")
            with open(val_labels_file, "r") as f:
                self.labels = [int(line.strip()) for line in f.readlines()]

            # Sort validation images to match ground truth order
            self.image_paths = sorted(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".JPEG")]
            )

        # Map WNIDs to numeric labels
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(set(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.split == "train":
            wnid = self.labels[idx]
            label = self.wnid_to_idx[wnid]
        elif self.split == "val":
            label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Create the superpixel map using SLIC
        superpixel_map = create_superpixel_image(image, n_segments=self.n_segments)

        # Convert the label to a Tensor
        label = torch.tensor(label, dtype=torch.long)

        return superpixel_map, image, label


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths
train_root_dir = r"E:\ImageNet\unpacked\train"
val_root_dir = r"E:\ImageNet\unpacked\validate"
val_labels_file = r"E:\ImageNet\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"

# Create datasets
train_dataset = ImageNetDataset(root_dir=train_root_dir, transform=transform, split="train")
val_dataset = ImageNetDataset(root_dir=val_root_dir, transform=transform, split="val", val_labels_file=val_labels_file)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
