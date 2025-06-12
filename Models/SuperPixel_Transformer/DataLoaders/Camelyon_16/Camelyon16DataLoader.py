import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
import openslide
from torchvision import transforms
from shapely.geometry import Polygon

from Models.SuperPixel_Transformer.Superpixel_Algorithms.SLIC import create_superpixel_image


def camelyon_collate(batch):
    """
    Custom collate: unwrap single-sample batches so downstream _step sees one tuple,
    avoiding manual slicing.
    """
    if len(batch) == 1:
        # return the single sample tuple directly
        return batch[0]

    # otherwise, batch multiple samples (if you ever use batch_size>1)
    sp_maps  = default_collate([b[0] for b in batch])
    patch_ts = default_collate([b[1] for b in batch])
    labels   = default_collate([b[2] for b in batch])
    paths    = [b[3] for b in batch]
    levels   = default_collate([b[4] for b in batch])
    x0s      = default_collate([b[5] for b in batch])
    y0s      = default_collate([b[6] for b in batch])
    polys    = [b[7] for b in batch]
    return sp_maps, patch_ts, labels, paths, levels, x0s, y0s, polys

class Camelyon16WSIDataset(Dataset):
    """
    Every item returns:
      - superpixel map (1, H, W)
      - normalized thumbnail (3, H, W)
      - slide-level binary label (0/1)
      - slide_path, level, x0, y0 (thumbnail coords)
      - list of tumor Polygons (shapely) in thumb coords
    """
    def __init__(self,
                 root_dir: str,
                 annotation_dir: str = None,
                 thumbnail_size=(2048, 2048),
                 num_segments: int = 50,
                 transform = None):
        self.image_dir   = os.path.join(root_dir, "images")
        self.ann_dir     = annotation_dir or os.path.join(root_dir, "annotations")
        self.files       = sorted(f for f in os.listdir(self.image_dir)
                                  if f.endswith(".tif"))
        self.thumb_sz    = thumbnail_size
        self.n_seg       = num_segments
        self.transform   = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn          = self.files[idx]
        slide_path  = os.path.join(self.image_dir, fn)
        xml_path    = os.path.join(self.ann_dir, fn.replace('.tif', '.xml'))

        slide       = openslide.OpenSlide(slide_path)
        full_w, full_h = slide.level_dimensions[0]
        thumb       = slide.get_thumbnail(self.thumb_sz).convert('RGB')
        slide.close()
        thumb_w, thumb_h = thumb.size

        img_t       = self.transform(thumb)
        sp_np       = create_superpixel_image(np.array(thumb), n_segments=self.n_seg)
        sp_map      = torch.from_numpy(sp_np).long().unsqueeze(0)

        polys = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for ann in root.findall(".//Annotation[@PartOfGroup='Tumor']"):
                coords = [(float(c.get('X')), float(c.get('Y'))) for c in ann.find('Coordinates').findall('Coordinate')]
                scaled = [(x*thumb_w/full_w, y*thumb_h/full_h) for x,y in coords]
                polys.append(Polygon(scaled))

        slide_label = 1.0 if polys else 0.0
        return (sp_map, img_t, torch.tensor(slide_label), slide_path, 0, 0, 0, polys)


def load_camelyon(
    root_dir: str,
    annotation_dir: str = None,
    thumbnail_size=(2048,2048),
    num_segments: int = 50,
    transform = None,
    batch_size: int = 1,
    val_split: float = 0.2,
    num_workers: int = 4
):
    ds    = Camelyon16WSIDataset(root_dir, annotation_dir, thumbnail_size, num_segments, transform)
    n_val = int(len(ds) * val_split)
    n_tr  = len(ds) - n_val
    train, val = random_split(ds, [n_tr, n_val])
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=camelyon_collate),
            DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=camelyon_collate))