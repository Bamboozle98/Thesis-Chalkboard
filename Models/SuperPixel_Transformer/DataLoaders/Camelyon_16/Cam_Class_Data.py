import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from skimage.color        import rgb2gray
from skimage.filters      import threshold_otsu
from skimage.segmentation import slic
import openslide
from torchvision import transforms
from sklearn.model_selection import train_test_split

class Cam16Dataset(Dataset):
    def __init__(self, wsi_dir, patch_level=6, n_segments=150):
        self.wsi_dir     = wsi_dir
        self.patch_level = patch_level
        self.n_segments  = n_segments

        # 1) list & sort all *.tif/.tiff by their numeric ID
        all_files = [
            fn for fn in os.listdir(wsi_dir)
            if fn.lower().endswith(('.tif', '.tiff'))
        ]
        all_files = sorted(
            all_files,
            key=lambda fn: int(re.search(r'_(\d+)', fn).group(1))
        )

        # 2) keep only normal_* or tumor_*
        self.files = [
            fn for fn in all_files
            if fn.lower().startswith(("normal_", "tumor_"))
        ]

        # 3) precompute labels: normal → 0, tumor → 1
        self.labels = [
            1 if fn.lower().startswith("tumor_") else 0
            for fn in self.files
        ]

        # 4) one transform for your CNN patches
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn    = self.files[idx]
        label = self.labels[idx]

        # build full path & open
        path = os.path.join(self.wsi_dir, fn)

        # debug
        print(f"[Cam16Dataset] opening: {path!r}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            slide = openslide.OpenSlide(path)
        except openslide.OpenSlideUnsupportedFormatError:
            # fallback: it’s not a supported WSI, so raise with extra info
            raise RuntimeError(
                f"{path} is not a supported OpenSlide format. "
                f"Maybe it’s a plain TIFF? size={os.path.getsize(path)} bytes"
            )

        # read the whole level‐`patch_level` at once
        W, H  = slide.level_dimensions[self.patch_level]
        full  = slide.read_region((0, 0), self.patch_level, (W, H)).convert("RGB")
        slide.close()

        # tissue‐only crop
        arr  = np.array(full)
        gray = rgb2gray(arr)
        thr  = threshold_otsu(gray)
        mask = gray < thr
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop_img = full.crop((x0, y0, x1, y1))

        # superpixel map
        crop_np = np.array(crop_img)
        sp_map  = slic(
            crop_np,
            n_segments = self.n_segments,
            compactness = 10,
            start_label = 0
        ).astype(np.int32)
        sp_map_tensor = torch.from_numpy(sp_map).long().unsqueeze(0)

        # normalized patch for CNN
        patch_tensor = self.patch_transform(crop_img)

        return {
            "sp_map":   sp_map_tensor,
            "patch":    patch_tensor,
            "label":    torch.tensor(label, dtype=torch.long),
            "meta": {
                "filename": fn,
                "level":    self.patch_level,
                "x0":       x0,
                "y0":       y0
            }
        }

def cam_loader(
    wsi_dir,
    patch_level=6,
    n_segments=50,
    val_split=0.2,
    random_state=42
):
    ds = Cam16Dataset(wsi_dir, patch_level=patch_level, n_segments=n_segments)
    idxs = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        idxs, test_size=val_split, random_state=random_state
    )

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=lambda batch: batch[0]
    )
    val_loader   = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda batch: batch[0]
    )
    return train_loader, val_loader
