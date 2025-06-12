import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from docutils.nodes import classifier
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.functional.classification import binary_accuracy
from torchvision import transforms
from skimage.segmentation import mark_boundaries
from skimage.measure    import regionprops
import matplotlib.pyplot as plt

from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50       import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN       import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet  import CustomResNet20
from Models.SuperPixel_Transformer.Transformer            import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon_16.Cam_Class_Data import cam_loader
from Models.SuperPixel_Transformer.Model_Pipelines.Large_Scale_Imagery.Camelyon.Cam_Debug import CamDebugCallback
from Models.SuperPixel_Transformer.config import (
    num_epochs, learning_rate, cnn_option,
    use_checkpoint, wsi_dir, ann_dir
)

torch.set_float32_matmul_precision('high')

def cnn_selection(option):
    if option == "ResNet18":
        return ResNet18()
    if option == "ResNet50":
        return ResNet50()
    if option == "CustomResNet":
        return CustomResNet20()
    if option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    raise ValueError(f"Unknown CNN option {option}")

class LitNetwork(pl.LightningModule):
    def __init__(self, cnn_option, feature_dim=512):
        super().__init__()
        # same CNN + transformer encoder
        self.cnn         = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(feature_dim, 1)
        self.loss_func  = torch.nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.val_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
        # ────────────────────────────────────────────────────────────────

        # positional MLP stays the same
        self.pos_proj   = torch.nn.Sequential(
            torch.nn.Linear(2, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim)
        )

        # keep the un‐normalization buffers
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, tokens):
        return self.transformer(tokens)

    def _step(self, sp_map, patch_tensor, label, *args):
        # label is 0 or 1
        y_true = label.to(self.device).view(-1).float()
        debug = {}

        # build superpixel tokens (unchanged)
        label_map = sp_map.squeeze(0).cpu().numpy()
        regions = regionprops(label_map)
        all_tokens = []
        for i, r in enumerate(regions):
            minr, minc, maxr, maxc = r.bbox
            t = patch_tensor[:, minr:maxr, minc:maxc].unsqueeze(0).to(self.device)
            mask_np = (label_map[minr:maxr, minc:maxc] == r.label).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)
            t = t * mask_t.unsqueeze(1)
            if t.mean().item() < 0.05:
                continue

            # CNN → pooled feature map → tokens
            feat_map = self.cnn(t)
            fmap_ds = F.adaptive_avg_pool2d(feat_map, (8, 8))
            tokens = fmap_ds.view(1, fmap_ds.size(1), -1).permute(0, 2, 1)

            # add positional encoding
            cy, cx = r.centroid
            norm = torch.tensor(
                [cx / label_map.shape[1], cy / label_map.shape[0]],
                dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            pos = self.pos_proj(norm)
            tokens = tokens + pos.unsqueeze(1)

            all_tokens.append(tokens)

        if not all_tokens:
            return None

        # concatenate all SP tokens → [1, n_sp, feat]
        seq = torch.cat(all_tokens, dim=1)

        # at the end, run through the transformer’s built-in head
        enc = self.transformer(seq)
        # enc could be shape [B, n_sp, 1] or [B, n_sp], so:
        #  - mean over the superpixel dim → [B, 1] or [B]
        #  - then flatten to [B]
        logits = enc.mean(dim=1).view(-1)
        return logits, y_true, debug

    def training_step(self, batch, batch_idx):
        sp_map, patch_tensor, label = batch["sp_map"], batch["patch"], batch["label"]
        out = self._step(sp_map, patch_tensor, label)
        if out is None:
            return None
        logits, y_true, _ = out

        loss = self.loss_func(logits, y_true)
        probs = torch.sigmoid(logits)
        acc = self.train_acc(probs, y_true.int())

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sp_map, patch_tensor, label = batch["sp_map"], batch["patch"], batch["label"]
        out = self._step(sp_map, patch_tensor, label)
        if out is None:
            return None
        logits, y_true, _ = out

        loss = self.loss_func(logits, y_true)
        probs = torch.sigmoid(logits)
        acc = self.val_acc(probs, y_true.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_acc"}}


if __name__ == "__main__":
    # 1) swap in your binary‐loader
    train_loader, val_loader = cam_loader(
        wsi_dir   = wsi_dir,
        patch_level = 6,
        n_segments  = 100,
        val_split   = 0.2
    )

    # 2) instantiate the classifier
    model = LitNetwork(cnn_option)

    # 3) (optional) debug as before…

    # 4) train with binary metrics
    callbacks = []
    if use_checkpoint:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
        )
    logger = pl_loggers.TensorBoardLogger("../../logs/binary_classification")
    debug_cb = CamDebugCallback(n_segments=100, every_n_epochs=1)

    trainer = pl.Trainer(
        max_epochs   = num_epochs,
        accelerator  = "gpu",
        callbacks    = [*callbacks, debug_cb],
        logger       = logger
    )
    trainer.fit(model, train_loader, val_loader)
