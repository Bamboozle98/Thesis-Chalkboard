import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import torch.nn.functional as F
from Models.CONFIG.config import (
    num_epochs, learning_rate, cnn_option,
    use_checkpoint, g_ann, wsi_dir
)
from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50       import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN       import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet  import CustomResNet20
from Models.SuperPixel_Transformer.Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon_16.ExperimentalDC16 import load_camelyon

torch.set_float32_matmul_precision('high')

def cnn_selection(option):
    if option=="ResNet18":    return ResNet18()
    if option=="ResNet50":    return ResNet50()
    if option=="CustomResNet":return CustomResNet20()
    if option=="SuperpixelCNN":
        return SuperpixelCNN(in_channels=3,out_channels=512)
    raise ValueError(option)

def extract_levelN_patch(slide_path, bbox, level):
    slide = openslide.OpenSlide(slide_path)
    x,y,w,h = bbox
    try:    patch = slide.read_region((x,y), level, (w,h)).convert("RGB")
    finally:slide.close()
    return patch

class LitNetwork(pl.LightningModule):
    def __init__(self, cnn_option, feature_dim=512):
        super().__init__()
        self.cnn         = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(feature_dim, 1)
        self.loss_func   = torch.nn.MSELoss()
        self.pos_proj    = torch.nn.Sequential(
            torch.nn.Linear(2, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.train_mae   = torchmetrics.MeanAbsoluteError()
        self.val_mae     = torchmetrics.MeanAbsoluteError()
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def forward(self, tokens):
        return self.transformer(tokens)

    def _step(self,
              sp_thumb, rs, path,
              scale0_x, scale0_y,
              scaleN_x, scaleN_y,
              patch_level,
              x0_crop, y0_crop):
        y_true = rs.to(self.device).view(-1)

        _, H_t, W_t = sp_thumb.shape
        thumb_map = sp_thumb.squeeze(0).cpu().numpy().astype(int)
        regions   = regionprops(thumb_map)

        all_tokens = []
        for r in regions:
            minr,minc,maxr,maxc = r.bbox
            orig_minr = minr + y0_crop
            orig_minc = minc + x0_crop
            orig_maxr = maxr + y0_crop
            orig_maxc = maxc + x0_crop

            x0_lvl0 = int(round(orig_minc * scale0_x))
            y0_lvl0 = int(round(orig_minr * scale0_y))
            wN = int(round((orig_maxc-orig_minc)*scaleN_x))
            hN = int(round((orig_maxr-orig_minr)*scaleN_y))

            patch = extract_levelN_patch(path, (x0_lvl0,y0_lvl0,wN,hN), patch_level)
            t     = self.patch_transform(patch).to(self.device).unsqueeze(0)

            feat_map = self.cnn(t)                       # [1,C,h',w']
            P = 4
            fmap_ds = F.adaptive_avg_pool2d(feat_map,(P,P))  # [1,C,P,P]
            C,pH,pW  = fmap_ds.shape[1:]
            tokens   = fmap_ds.view(1,C,pH*pW).permute(0,2,1) # [1,P*P,C]

            cy,cx = r.centroid
            norm  = torch.tensor([cx/W_t, cy/H_t],dtype=torch.float32,device=self.device).unsqueeze(0)
            pos   = self.pos_proj(norm)                    # [1,C]
            tokens = tokens + pos.unsqueeze(1)             # [1,P*P,C]

            all_tokens.append(tokens)

        if not all_tokens:
            return None

        seq   = torch.cat(all_tokens, dim=1)              # [1,total,C]
        preds = self(seq).squeeze(-1)
        return preds, y_true

    def training_step(self, batch, batch_idx):
        out = self._step(*batch)
        if out is None: return None
        preds, y_true = out
        loss = self.loss_func(preds, y_true)
        self.log("train_loss",loss,on_epoch=True)
        self.log("train_mae", self.train_mae(preds,y_true),on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self._step(*batch)
        if out is None: return None
        preds, y_true = out
        loss = self.loss_func(preds, y_true)
        self.log("val_loss",loss,on_epoch=True,prog_bar=True)
        self.log("val_mae", self.val_mae(preds,y_true),on_epoch=True,prog_bar=True)

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(),lr=learning_rate)
        sched = torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.1)
        return {"optimizer":opt,
                "lr_scheduler":{"scheduler":sched,"monitor":"val_mae"}}

if __name__=="__main__":
    import os, math
    import matplotlib.pyplot as plt
    import numpy as np
    import openslide
    from skimage.color        import rgb2gray
    from skimage.filters      import threshold_otsu
    from skimage.segmentation import mark_boundaries, slic
    from skimage.measure      import regionprops

    # 1) Grab one batch from your loader (uses same crop & scale logic as __getitem__)
    model       = LitNetwork(cnn_option)
    train_loader, val_loader = load_camelyon(wsi_dir=wsi_dir, csv_path=g_ann)
    sp_thumb, rs, path, scale0_x, scale0_y, scaleN_x, scaleN_y, lvl, x0_crop, y0_crop = next(iter(train_loader))

    # 2) Re-open slide & recreate the exact same thumbnail + crop
    slide    = openslide.OpenSlide(path)
    thumb    = slide.get_thumbnail((2048,2048)).convert("RGB")
    tn_np    = np.array(thumb)
    gray     = rgb2gray(tn_np)
    t_otsu   = threshold_otsu(gray)
    mask     = gray < t_otsu
    ys, xs   = np.where(mask)
    yc0,yc1  = ys.min(), ys.max()+1
    xc0,xc1  = xs.min(), xs.max()+1

    thumb_crop = thumb.crop((xc0, yc0, xc1, yc1))
    tc_np      = np.array(thumb_crop)

    # 3) Run SLIC **on that crop** so we get a map the same size as thumb_crop:
    sp_crop = slic(tc_np, n_segments=50, compactness=10, start_label=0)

    # 4) Overlay boundaries on the cropped thumbnail:
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(mark_boundaries(tc_np, sp_crop))
    ax.set_title(f"Sample: {os.path.basename(path)}")
    ax.axis('off')
    plt.show()

    # 5) Plot **all** superpixel patches in a grid, using the correct scales:

    regions = regionprops(sp_crop.astype(int))
    N       = len(regions)
    cols    = 8
    rows    = math.ceil(N/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for idx, (r, ax) in enumerate(zip(regions, axes)):
        minr, minc, maxr, maxc = r.bbox

        # 5a) map bbox in cropped‐thumb coords back into full‐thumb coords
        orig_minr = minr + yc0
        orig_minc = minc + xc0
        orig_maxr = maxr + yc0
        orig_maxc = maxc + xc0

        # 5b) **ORIGIN** at level-0 must use scale0_x / scale0_y:
        x0_level0 = int(round(orig_minc * scale0_x))
        y0_level0 = int(round(orig_minr * scale0_y))

        # 5c) **SIZE** at patch_level uses scaleN_x / scaleN_y:
        wN = int(round((orig_maxc - orig_minc) * scaleN_x))
        hN = int(round((orig_maxr - orig_minr) * scaleN_y))

        patch = slide.read_region(
            (x0_level0, y0_level0),
            lvl,
            (wN, hN)
        ).convert("RGB")

        ax.imshow(patch)
        ax.set_title(f"SP{idx}", fontsize=6)
        ax.axis('off')

    # turn off any unused axes
    for ax in axes[N:]:
        ax.axis('off')

    plt.suptitle("All superpixel high-res patches", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    slide.close()

    model = LitNetwork(cnn_option)

    # 6) now proceed with training
    logger = pl_loggers.TensorBoardLogger("../logs/rs_regression")
    cbs    = []
    if use_checkpoint:
        cbs.append(pl.callbacks.ModelCheckpoint(
            monitor="val_mae", mode="min", save_top_k=1
        ))
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=cbs,
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)


