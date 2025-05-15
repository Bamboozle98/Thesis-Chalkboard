import os
import csv
import numpy as np
import torch
import openslide
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
from Models.SuperPixel_Transformer.Superpixel_Algorithms.WSI_SLIC import create_superpixel_image

class Camelyon16WSIDataset(Dataset):
    def __init__(
        self,
        wsi_dir,
        thumbnail_size=(2048, 2048),
        num_segments=50,
        csv_path=None,
        patch_level=6   # slide level to read patches from
    ):
        self.wsi_dir        = wsi_dir
        self.thumbnail_size = thumbnail_size
        self.num_segments   = num_segments
        self.patch_level    = patch_level

        # list slide files
        all_files = [
            f for f in os.listdir(wsi_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        ]
        self.wsi_files = sorted(all_files, key=lambda f: int(os.path.splitext(f)[0]))

        # load RS scores
        if csv_path is None:
            raise ValueError("csv_path must be provided")
        self.label_mapping = {}
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw, rs in csv.reader(f, delimiter='\t'):
                if raw.lower() == "case#":
                    continue
                try:
                    norm = str(int(raw))
                    self.label_mapping[norm] = float(rs)
                except:
                    pass

        # filter out un‐annotated
        self.wsi_files = [
            fn for fn in self.wsi_files
            if str(int(os.path.splitext(fn)[0])) in self.label_mapping
        ]

    def __len__(self):
        return len(self.wsi_files)

    def __getitem__(self, idx):
        fn   = self.wsi_files[idx]
        path = os.path.join(self.wsi_dir, fn)

        slide = openslide.OpenSlide(path)
        # dims at patch_level (level-N) and at level-0
        fullN_w, fullN_h = slide.level_dimensions[self.patch_level]
        full0_w, full0_h = slide.level_dimensions[0]

        thumb = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        slide.close()

        # tissue crop on thumbnail
        tn      = np.array(thumb)
        gray    = rgb2gray(tn)
        mask    = gray < threshold_otsu(gray)
        ys, xs  = np.where(mask)
        y0, y1  = ys.min(), ys.max() + 1
        x0, x1  = xs.min(), xs.max() + 1
        thumb_crop = thumb.crop((x0, y0, x1, y1))

        # superpixel map on the **thumbnail crop**
        sp_thumb = create_superpixel_image(
            thumb_crop,
            (fullN_w, fullN_h),
            self.num_segments
        )

        # compute scales
        scale0_x = full0_w / thumb.width
        scale0_y = full0_h / thumb.height
        scaleN_x = fullN_w / thumb.width
        scaleN_y = fullN_h / thumb.height

        # lookup label
        norm = str(int(os.path.splitext(fn)[0]))
        rs   = float(self.label_mapping[norm])

        # to tensors
        x = torch.from_numpy(sp_thumb).long().unsqueeze(0)   # (1, T_h, T_w)
        y = torch.tensor(rs, dtype=torch.float32)

        # return everything: map, label, path, scales, level, crop offsets
        return (
            x, y, path,
            scale0_x, scale0_y,
            scaleN_x, scaleN_y,
            self.patch_level,
            x0, y0
        )

def load_camelyon(wsi_dir, num_segments=50, csv_path=None,
                  val_split=0.2, random_state=42):
    dataset = Camelyon16WSIDataset(
        wsi_dir=wsi_dir,
        num_segments=num_segments,
        csv_path=csv_path
    )
    idxs      = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idxs, test_size=val_split,
                                           random_state=random_state)

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    # Windows-friendly single-worker loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch[0]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch[0]
    )
    return train_loader, val_loader