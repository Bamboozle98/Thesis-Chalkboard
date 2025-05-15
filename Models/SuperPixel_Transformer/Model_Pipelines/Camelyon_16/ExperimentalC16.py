# Models/SuperPixel_Transformer/Model_Pipelines/Lightning_SPFormer_Camelyon16.py

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import numpy as np
from skimage.measure import label as sklabel, regionprops
from PIL import Image
import openslide
import torch.nn.functional as F

from Models.SuperPixel_Transformer.config import (
    num_epochs, learning_rate, dataset_option, cnn_option,
    use_checkpoint, wsi_dir, g_dir, g_ann
)
from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet import CustomResNet20
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon_16.ExperimentalDC16 import load_camelyon


# CUDA optimization
torch.set_float32_matmul_precision('high')

def cnn_selection(option):
    if option == "ResNet18":
        return ResNet18()
    elif option == "ResNet50":
        return ResNet50()
    elif option == "CustomResNet":
        return CustomResNet20()
    elif option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    else:
        raise ValueError(f"Invalid CNN selection: {option}")

def extract_level1_patch(path, bbox, level=2):
    slide = openslide.OpenSlide(path)
    x,y,w,h = bbox
    try:
        patch = slide.read_region((x,y), level, (w,h)).convert("RGB")
    finally:
        slide.close()
    return patch


class LitNetwork(pl.LightningModule):
    def __init__(self, cnn_option, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.cnn         = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(feature_dim, 1)
        self.loss_func   = torch.nn.MSELoss()
        # a small MLP to project 2D → D for positional encoding
        self.pos_proj    = torch.nn.Sequential(
            torch.nn.Linear(2, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.train_mae   = torchmetrics.MeanAbsoluteError()
        self.val_mae     = torchmetrics.MeanAbsoluteError()
        # only tensor & normalize the *high-res* patches
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
        # optional: learn a 2D → D position projection for centroids
        self.pos_proj = torch.nn.Linear(2, feature_dim)

    def forward(self, tokens):
        return self.transformer(tokens)

    def _step(self,
              sp_thumb, rs, path,
              scale0_x, scale0_y,
              scaleN_x, scaleN_y,
              patch_level,
              x0_crop, y0_crop):
        # 1) ground truth
        y_true = rs.to(self.device).view(-1)

        # 2) regionprops on the thumbnail‐crop map
        _, H_t, W_t = sp_thumb.shape
        thumb_map = sp_thumb.squeeze(0).cpu().numpy().astype(int)
        regions = regionprops(thumb_map)

        all_tokens = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox

            # 3) re‐anchor into full‐thumbnail coords
            orig_minr = minr + y0_crop
            orig_minc = minc + x0_crop
            orig_maxr = maxr + y0_crop
            orig_maxc = maxc + x0_crop

            # 4) map to level-0 origin & compute level-N size
            x0_lvl0 = int(round(orig_minc * scale0_x))
            y0_lvl0 = int(round(orig_minr * scale0_y))
            wN = int(round((orig_maxc - orig_minc) * scaleN_x))
            hN = int(round((orig_maxr - orig_minr) * scaleN_y))

            # 5) extract high-res patch & preprocess
            patch = extract_level1_patch(path,
                                         (x0_lvl0, y0_lvl0, wN, hN),
                                         level=patch_level)
            t = self.patch_transform(patch).to(self.device).unsqueeze(0)  # [1,3,H,W]

            # 6) CNN → feature map
            feat_map = self.cnn(t)  # [1, C, h', w']

            # 7) downsample to a small grid of tokens (4×4 here)
            P = 4
            fmap_ds = F.adaptive_avg_pool2d(feat_map, (P, P))  # [1, C, P, P]
            C, pH, pW = fmap_ds.shape[1:]
            tokens = fmap_ds.view(1, C, pH * pW).permute(0, 2, 1)  # [1, P*P, C]

            # 8) positional encoding from normalized centroid
            cy, cx = r.centroid
            norm = torch.tensor([cx / W_t, cy / H_t],
                                dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # [1,2]
            pos = self.pos_proj(norm)  # [1, C]

            # 9) add pos to each token
            tokens = tokens + pos.unsqueeze(1)  # [1, P*P, C]
            all_tokens.append(tokens)

        # 10) if no regions, bail out
        if not all_tokens:
            return None

        # 11) concatenate and run transformer
        seq = torch.cat(all_tokens, dim=1)  # [1, total_tokens, C]
        preds = self.transformer(seq).squeeze(-1)  # [1]

        return preds, y_true

    def training_step(self, batch, batch_idx):
        (sp_thumb, rs, path,
         scale0_x, scale0_y,
         scaleN_x, scaleN_y,
         patch_level,
         x0_crop, y0_crop) = batch

        out = self._step(
            sp_thumb, rs, path,
            scale0_x, scale0_y,
            scaleN_x, scaleN_y,
            patch_level,
            x0_crop, y0_crop
        )
        if out is None:
            return None
        preds, y_true = out
        loss = self.loss_func(preds, y_true)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mae",  self.train_mae(preds, y_true), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self._step(*batch)
        if out is None:
            return  # still skip if no tokens
        preds, y_true = out
        loss = self.loss_func(preds, y_true)

        # now these logs will actually run once per batch
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae(preds, y_true),
                 on_epoch=True, prog_bar=True)



    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "monitor":   "val_mae"}}



if __name__ == "__main__":
    model  = LitNetwork(cnn_option)
    device = "gpu"

    # pass your RS TSV path via config.csv_path
    train_loader, val_loader = load_camelyon(
        wsi_dir=g_dir,
        csv_path=g_ann
    )
    logger = pl_loggers.TensorBoardLogger(
        save_dir=(f"../../../../logs/rs_regression")
    )

    callbacks = []
    if use_checkpoint:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            monitor="val_mae",
            mode="min",
            save_top_k=1
        ))

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
