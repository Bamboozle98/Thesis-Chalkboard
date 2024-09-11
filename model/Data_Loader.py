import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from SLIC import create_superpixel_image


# Define the custom Dataset class at the module level
class OxfordPetsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, n_segments=100):
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

        # This is where the images are processed via superpixels.
        superpixel_image = create_superpixel_image(image, n_segments=self.n_segments)

        # Apply transformations
        if self.transform:
            superpixel_image = self.transform(superpixel_image)

        # Get the label
        label = self.labels[idx]

        return superpixel_image, label

def data_process():
    # Define the path to the dataset
    dataset_dir = 'C:/Users/mccutcheonc18/PycharmProjects/Thesis-Chalkboard/OxfordPets/images'

    # List all image files
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

    # Extract class labels from filenames
    def extract_label(file_name):
        label = file_name.split('_')[:-1]  # All parts before the last underscore form the class name
        return '_'.join(label)

    # Create lists to store file paths and corresponding labels
    image_paths = []
    labels = []

    for img_file in image_files:
        img_path = os.path.join(dataset_dir, img_file)
        label = extract_label(img_file)

        image_paths.append(img_path)
        labels.append(label)

    # Encode labels into integers using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Get class names
    class_names = label_encoder.classes_  # This will give you an array of class names

    print(f"Class names: {class_names}")

    # Split into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, encoded_labels, test_size=0.2, random_state=42
    )

    # Define image transformations
    IMG_SIZE = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # This is ImageNet normalization, certainly not necessary
    ])

    # Create train and validation datasets
    train_dataset = OxfordPetsDataset(train_images, train_labels, transform=transform)
    val_dataset = OxfordPetsDataset(val_images, val_labels, transform=transform)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names




