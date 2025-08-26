import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import ImageNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from Models.SuperPixel_Transformer.Superpixel_Algorithms.SLIC import create_superpixel_image
from Models.CONFIG.config import (oxford_dataset_dir, batch_size, imagenet_train_dir, imagenet_val_dir,
                                  dataset_option, high_res_dir, match_size, imagenet_root)


class SuperpixelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, n_segments=50, cache_superpixels=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.n_segments = n_segments
        self.cache_superpixels = cache_superpixels
        self.superpixel_cache = {}

        if self.cache_superpixels:
            print("Precomputing superpixel maps for efficiency...")
            for img_path in self.image_paths:
                image = Image.open(img_path).convert("RGB")
                tensor_image = transforms.ToTensor()(image)
                self.superpixel_cache[img_path] = create_superpixel_image(tensor_image, n_segments=n_segments)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)

        superpixel_map = (
            self.superpixel_cache[img_path]
            if self.cache_superpixels
            else create_superpixel_image(transformed_image, n_segments=self.n_segments)
        )

        label = self.labels[idx]
        return superpixel_map, transformed_image, label


def load_dataset(dataset_name=dataset_option, cache_superpixels=False, ox_dir=oxford_dataset_dir,
                 in_train=imagenet_train_dir, in_val=imagenet_val_dir):
    """
    Loads the specified dataset (Oxford Pets or ImageNetDataset) with superpixel processing.

    Args:
        dataset_name (str): 'oxford_pets' or 'imagenet'
        cache_superpixels (bool): Whether to cache superpixel maps for efficiency.
        ox_dir (str): Directory where the dataset is stored.
        in_train (str): Directory where the training set for ImageNetDataset is stored.
        in_val (str): Directory where the validation set for ImageNetDataset is stored.

    Returns:
        train_loader, val_loader, class_names
    """
    IMG_SIZE = match_size
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == 'image_net':
        train_data = ImageNet(root=imagenet_root, split='train', transform=transform)
        val_data = ImageNet(root=imagenet_root, split='val', transform=transform)

        class_names = train_data.classes
        n_classes = len(class_names)

        train_dataset = SuperpixelDataset(
            [p for p, _ in train_data.samples],
            [l for _, l in train_data.samples],
            transform=transform,
            cache_superpixels=cache_superpixels
        )
        val_dataset = SuperpixelDataset(
            [p for p, _ in val_data.samples],
            [l for _, l in val_data.samples],
            transform=transform,
            cache_superpixels=cache_superpixels
        )


    elif dataset_name == 'high_res':
        image_paths = []
        labels = []

        for class_name in os.listdir(high_res_dir):
            class_dir = os.path.join(high_res_dir, class_name)

            if os.path.isdir(class_dir):
                for img_file in glob.glob(os.path.join(class_dir, '*.jpg')):
                    image_paths.append(img_file)

                    labels.append(class_name)

        label_encoder = LabelEncoder()

        encoded_labels = label_encoder.fit_transform(labels)

        class_names = label_encoder.classes_
        n_classes = len(class_names)

        train_images, val_images, train_labels, val_labels = train_test_split(image_paths, encoded_labels,
                                                                              test_size=0.2, random_state=42)

        train_dataset = SuperpixelDataset(train_images, train_labels, transform=transform,
                                          cache_superpixels=cache_superpixels)

        val_dataset = SuperpixelDataset(val_images, val_labels, transform=transform,
                                        cache_superpixels=cache_superpixels)



    elif dataset_name == 'oxford_pets':
        image_files = glob.glob(os.path.join(ox_dir, '*.jpg'))

        def extract_label(file_name):
            return '_'.join(os.path.basename(file_name).split('_')[:-1])

        image_paths = []
        labels = []

        for img_file in image_files:
            label = extract_label(img_file)
            image_paths.append(img_file)
            labels.append(label)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        class_names = label_encoder.classes_
        n_classes = len(class_names)

        train_images, val_images, train_labels, val_labels = train_test_split(
            image_paths, encoded_labels, test_size=0.2, random_state=42
        )

        train_dataset = SuperpixelDataset(train_images, train_labels, transform=transform, cache_superpixels=cache_superpixels)
        val_dataset = SuperpixelDataset(val_images, val_labels, transform=transform, cache_superpixels=cache_superpixels)


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'oxford_pets' or 'imagenet'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names, n_classes
