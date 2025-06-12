# Models/SuperPixel_Transformer/Model_Pipelines/Large_Scale_Imagery/High_Res_DataL.py

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from skimage.color        import rgb2gray
from skimage.filters      import threshold_otsu
from skimage.segmentation import slic
from skimage.measure      import regionprops
import openslide
from torchvision import transforms
from sklearn.model_selection import train_test_split

class HighResSPDataset(Dataset):
    def __init__(self, wsi_dir, csv_path, patch_level=5, n_segments=150):
        self.wsi_dir     = wsi_dir
        self.patch_level = patch_level
        self.n_segments  = n_segments

        # load RS scores
        self.mapping = {}
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            for raw, rs in reader:
                if raw.lower().startswith("case"):
                    continue
                try:
                    self.mapping[str(int(raw))] = float(rs)
                except ValueError:
                    pass

        # gather & sort WSI files
        all_files = [
            fn for fn in os.listdir(wsi_dir)
            if fn.lower().endswith(('.tif', '.tiff'))
        ]
        all_files = sorted(all_files, key=lambda fn: int(os.path.splitext(fn)[0]))

        # keep only annotated slides
        self.files = [
            fn for fn in all_files
            if str(int(os.path.splitext(fn)[0])) in self.mapping
        ]

        # build patch_transform once
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # normalization constants for RS
        self.sids   = sorted(self.mapping)
        self.raw_rs = np.array([self.mapping[s] for s in self.sids], dtype=np.float32)
        self.rs_mean = float(self.raw_rs.mean())
        self.rs_std  = float(self.raw_rs.std())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn  = self.files[idx]
        sid = str(int(os.path.splitext(fn)[0]))
        raw = float(self.mapping[sid])
        norm = (raw - self.rs_mean) / (self.rs_std + 1e-8)
        rs = torch.tensor([norm], dtype=torch.float32)
        path = os.path.join(self.wsi_dir, fn)

        # 1) read full at patch_level
        slide = openslide.OpenSlide(path)
        W, H  = slide.level_dimensions[self.patch_level]
        full  = slide.read_region((0,0), self.patch_level, (W,H)).convert("RGB")
        slide.close()

        # 2) tissue‐crop
        arr  = np.array(full)
        gray = rgb2gray(arr)
        thr  = threshold_otsu(gray)
        mask = gray < thr
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        crop_img = full.crop((x0, y0, x1, y1))

        crop_np = np.array(crop_img)
        sp_map = slic(
            crop_np,
            n_segments=self.n_segments,
            compactness=10,
            start_label=0
        ).astype(np.int32)

        # wrap into a (1, Hc, Wc) LongTensor
        sp_map_tensor = torch.from_numpy(sp_map).long().unsqueeze(0)

        # 4) normalize the cropped patch for the CNN
        patch_tensor = self.patch_transform(crop_img)

        return (
            sp_map_tensor,  # now shape = (1, Hc, Wc)
            patch_tensor,  # (3, Hc, Wc)
            rs,  # [1]
            path,  # str
            self.patch_level,  # int
            x0,  # int
            y0  # int
        )


def highres_loader(
    wsi_dir,
    csv_path,
    patch_level=5,
    n_segments=50,
    val_split=0.2,
    random_state=42
):
    ds = HighResSPDataset(wsi_dir, csv_path, patch_level, n_segments)
    idxs = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        idxs, test_size=val_split, random_state=random_state
    )
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    # Windows‐safe, unpack single sample directly
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=lambda b: b[0]
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda b: b[0]
    )
    return train_loader, val_loader
