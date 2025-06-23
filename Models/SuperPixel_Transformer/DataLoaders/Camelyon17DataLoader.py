import os
import csv
from torch.utils.data import Dataset, DataLoader
import openslide
from torchvision import transforms
from Models.SuperPixel_Transformer.Superpixel_Algorithms.SLIC import create_superpixel_image


class Camelyon17WSIDataset(Dataset):
    def __init__(self, wsi_dir, csv_path, transform=None, thumbnail_size=(2048, 2048),
                 num_segments=50, cache_superpixels=False, csv_delimiter=','):
        """
        Args:
            wsi_dir (str): Directory containing WSIs (e.g., train or test folder).
            csv_path (str): Path to CSV file with slide-level labels.
            transform (callable, optional): Transformation to apply.
            thumbnail_size (tuple): Size for the downsampled image.
            num_segments (int): Number of segments for superpixel segmentation.
            cache_superpixels (bool): If True, precompute and cache the superpixel maps.
            csv_delimiter (str): Delimiter used in the CSV file.
        """
        self.wsi_dir = wsi_dir
        self.wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith('.tif')]
        self.csv_path = csv_path
        self.transform = transform
        self.thumbnail_size = thumbnail_size
        self.num_segments = num_segments
        self.cache_superpixels = cache_superpixels
        self.superpixel_cache = {}

        # Read the CSV into a mapping: slide name (without extension) -> label
        self.label_mapping = {}
        with open(self.csv_path, 'r') as f:
            # Adjust fieldnames or use DictReader if the CSV has a header row.
            reader = csv.DictReader(f, delimiter=csv_delimiter)
            for row in reader:
                # Adjust the keys to match your CSV file. For example, assume columns "Slide" and "Label"
                slide_name = row.get('Slide') or row.get('filename')
                label_val = row.get('Label') or row.get('Diagnosis')
                if slide_name is not None and label_val is not None:
                    # Store without extension to facilitate matching
                    self.label_mapping[os.path.splitext(slide_name)[0]] = label_val

        # Optionally precompute superpixel maps on the thumbnail for speed
        if self.cache_superpixels:
            print("Precomputing superpixel maps for efficiency...")
            for file_name in self.wsi_files:
                wsi_path = os.path.join(self.wsi_dir, file_name)
                slide = openslide.OpenSlide(wsi_path)
                full_width, full_height = slide.level_dimensions[0]
                thumbnail = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
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
        full_width, full_height = slide.level_dimensions[0]
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

        # Label extraction: look up the slide (using filename without extension)
        file_stem = os.path.splitext(wsi_file)[0]
        label_str = self.label_mapping.get(file_stem, "Unknown")
        # Adjust mapping as needed – here we assume "Tumor" and "Normal"
        label_map = {"Tumor": 1, "Normal": 0, "tumor": 1, "normal": 0}
        label = label_map.get(label_str, -1)
        slide.close()

        # Return the superpixel map, thumbnail, label, scaling factors, full-resolution dims, and the slide path.
        return superpixel_map, thumbnail_trans, label, (scale_x, scale_y), (full_width, full_height), wsi_path


def load_camelyon17(wsi_train_path, wsi_test_path, csv_train_path, csv_test_path, cache_superpixels=False):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Camelyon17WSIDataset(
        wsi_dir=wsi_train_path,
        csv_path=csv_train_path,
        transform=transform,
        thumbnail_size=(2048, 2048),
        num_segments=50,
        cache_superpixels=cache_superpixels,
        csv_delimiter=','  # change if your CSV uses a different delimiter
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataset = Camelyon17WSIDataset(
        wsi_dir=wsi_test_path,
        csv_path=csv_test_path,
        transform=transform,
        thumbnail_size=(2048, 2048),
        num_segments=50,
        cache_superpixels=cache_superpixels,
        csv_delimiter=','
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader
