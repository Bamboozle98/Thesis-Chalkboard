import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from SLIC import create_superpixel_image


def permute_image(x):
    return x.permute(1, 2, 0)


class OxfordPetsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, n_segments=50):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.n_segments = n_segments

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations (if any) to the original image for CNN input
        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)

        # Create the superpixel map using SLIC
        superpixel_map = create_superpixel_image(transformed_image, n_segments=self.n_segments)
        # print("Superpixel map")
        # print(superpixel_map)
        # Get the label
        label = self.labels[idx]

        return superpixel_map, transformed_image, label


# Function to create DataLoaders
def data_process_SP(dataset_dir = 'C:/Users/cbran/PycharmProjects/Thesis-Chalkboard/Data/test'):

    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

    def extract_label(file_name):
        label = file_name.split('_')[:-1]  # Extract class name from filename
        return '_'.join(label)

    image_paths = []
    labels = []

    for img_file in image_files:
        img_path = os.path.join(dataset_dir, img_file)
        label = extract_label(img_file)

        image_paths.append(img_path)
        labels.append(label)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_

    # Split into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, encoded_labels, test_size=0.2, random_state=42
    )

    # Define image transformations
    IMG_SIZE = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(permute_image)
    ])
    train_dataset = OxfordPetsDataset(train_images, train_labels, transform=transform)
    val_dataset = OxfordPetsDataset(val_images, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names
