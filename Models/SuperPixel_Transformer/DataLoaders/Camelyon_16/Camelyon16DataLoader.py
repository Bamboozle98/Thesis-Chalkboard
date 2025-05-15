import os
import csv
from torch.utils.data import Dataset, DataLoader, random_split
import openslide
from torchvision import transforms
from Models.SuperPixel_Transformer.Superpixel_Algorithms.SLIC import create_superpixel_image
from Models.SuperPixel_Transformer.config import wsi_dir


class Camelyon16WSIDataset(Dataset):
    def __init__(self, wsi_dir, transform=None, thumbnail_size=(2048, 2048),
                 num_segments=50, cache_superpixels=False, csv_path=None):
        """
        Args:
            wsi_dir (str): Directory containing WSIs (e.g., train or test folder).
            transform (callable, optional): Transformation to apply.
            thumbnail_size (tuple): Size for the downsampled image.
            num_segments (int): Number of segments for superpixel segmentation.
            cache_superpixels (bool): If True, precompute and cache the superpixel maps.
            csv_path (str, optional): Path to CSV file with validation labels.
        """
        self.wsi_dir = wsi_dir
        self.wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith('.tif')]
        self.transform = transform
        self.thumbnail_size = thumbnail_size
        self.num_segments = num_segments
        self.cache_superpixels = cache_superpixels
        self.csv_path = csv_path
        self.superpixel_cache = {}

        # Load CSV if provided
        if self.csv_path is not None:
            self.label_mapping = {}
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 2:
                        continue
                    if row[0].lower() == "filename":
                        continue
                    self.label_mapping[row[0]] = row[1]
        else:
            self.label_mapping = None

        # Optionally precompute superpixel maps on the thumbnail
        if self.cache_superpixels:
            print("Precomputing superpixel maps for efficiency...")
            for file_name in self.wsi_files:
                wsi_path = os.path.join(self.wsi_dir, file_name)
                slide = openslide.OpenSlide(wsi_path)
                full_width, full_height = slide.level_dimensions[4]
                thumbnail = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
                scale_x = full_width / thumbnail.width
                scale_y = full_height / thumbnail.height
                if self.transform:
                    thumbnail_trans = self.transform(thumbnail)
                else:
                    thumbnail_trans = transforms.ToTensor()(thumbnail)
                self.superpixel_cache[wsi_path] = create_superpixel_image(thumbnail_trans, n_segments=self.num_segments)
                slide.close()

    def __len__(self):
        return len(self.wsi_files)

    def __getitem__(self, idx):
        wsi_file = self.wsi_files[idx]
        wsi_path = os.path.join(self.wsi_dir, wsi_file)
        slide = openslide.OpenSlide(wsi_path)
        full_width, full_height = slide.level_dimensions[4]
        thumbnail = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        scale_x = full_width / thumbnail.width
        scale_y = full_height / thumbnail.height
        if self.transform:
            thumbnail_trans = self.transform(thumbnail)
        else:
            thumbnail_trans = transforms.ToTensor()(thumbnail)
        if self.cache_superpixels:
            superpixel_map = self.superpixel_cache[wsi_path]
        else:
            superpixel_map = create_superpixel_image(thumbnail_trans, n_segments=self.num_segments)
        # Label extraction:
        if self.label_mapping is not None:
            file_stem = os.path.splitext(wsi_file)[0]
            label_str = self.label_mapping.get(file_stem, "Unknown")
            label_map = {"Tumor": 1, "Normal": 0}
            label = label_map.get(label_str, -1)
        else:
            label_str = wsi_file.split('_')[0].lower()
            label_map = {"tumor": 1, "normal": 0}
            label = label_map.get(label_str, -1)
        slide.close()
        # Return the superpixel map, thumbnail, label, scaling factors, full-resolution dims, and the slide path.
        return superpixel_map, thumbnail_trans, label, (scale_x, scale_y), (full_width, full_height), wsi_path

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_camelyon(wsi_dir, transform, thumbnail_size=(2048, 2048),
                               num_segments=50, cache_superpixels=False,
                               csv_path=None, val_split=0.2):
    # Create the dataset using your existing Camelyon16WSIDataset


    dataset = Camelyon16WSIDataset(wsi_dir=wsi_dir, transform=transform,
                                   thumbnail_size=thumbnail_size, num_segments=num_segments,
                                   cache_superpixels=cache_superpixels, csv_path=csv_path)

    # Compute split lengths
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # Random split into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader

