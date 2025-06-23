import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from Models.SuperPixel_Transformer.config import high_res_dir, batch_size
import glob


class High_Res_Dataset(Dataset):
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

        # Get the label
        label = self.labels[idx]

        return transformed_image, label


# Function to create DataLoaders
def data_process(dataset_dir = high_res_dir):
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

    # Define image transformations
    IMG_SIZE = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = High_Res_Dataset(train_images, train_labels, transform=transform)
    val_dataset = High_Res_Dataset(val_images, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names
