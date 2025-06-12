# Models/SuperPixel_Transformer/Model_Pipelines/High_Res_Exp.py

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from skimage.segmentation import mark_boundaries
from skimage.measure    import regionprops
import matplotlib.pyplot as plt

from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50       import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN       import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet  import CustomResNet20
from Models.SuperPixel_Transformer.Transformer            import TransformerEncoder
from Models.SuperPixel_Transformer.Model_Pipelines.Large_Scale_Imagery.Gerardt.High_Res_DataL import highres_loader
from Models.SuperPixel_Transformer.Model_Pipelines.Large_Scale_Imagery.Gerardt.DebugCallback import DebugCallback
from Models.SuperPixel_Transformer.config import (
    num_epochs, learning_rate, cnn_option,
    use_checkpoint, g_dir, g_ann
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
        self.cnn         = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(feature_dim, 1)
        self.loss_func   = torch.nn.MSELoss()
        # positional MLP
        self.pos_proj    = torch.nn.Sequential(
            torch.nn.Linear(2, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.train_mae   = torchmetrics.MeanAbsoluteError()
        self.val_mae     = torchmetrics.MeanAbsoluteError()
        # buffers for unnormalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, tokens):
        return self.transformer(tokens)

    def _step(self, sp_map, patch_tensor, rs, *args):
        y_true = rs.to(self.device).view(-1)
        debug = {}

        label_map = sp_map.squeeze(0).cpu().numpy()
        regions = regionprops(label_map)

        all_tokens = []
        for i, r in enumerate(regions):
            minr, minc, maxr, maxc = r.bbox
            t = patch_tensor[:, minr:maxr, minc:maxc].unsqueeze(0).to(self.device)
            # build a binary mask tensor of exactly the same H×W
            mask_np = (label_map[minr:maxr, minc:maxc] == r.label).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)  # [1,H,W]
            # expand to channels and multiply
            t = t * mask_t.unsqueeze(1)  # [1,3,H,W], zeros outside SP

            # skip blank patches
            if t.mean().item() < 0.05:
                continue

            # prepare numpy mask for overlay
            mask_np = label_map[minr:maxr, minc:maxc] == r.label

            # Debug first SP occasionally
            if i == 0 and self.global_step % 50 == 0:
                from torchvision.utils import save_image
                vis = (t * self.std + self.mean).clamp(0,1)[0].cpu()
                save_image(vis, f"debug/SP{r.label}_patch.png")
                overlay = mark_boundaries(vis.permute(1,2,0).numpy(), mask_np.astype(int), color=(1,1,0))
                plt.imsave(f"debug/SP{r.label}_overlay.png", overlay)
                debug['cnn_input'] = vis.unsqueeze(0)

            # CNN forward
            feat_map = self.cnn(t)
            if i == 0 and 'cnn_input' in debug:
                debug['cnn_output'] = feat_map.detach().cpu()

            # build tokens
            fmap_ds = F.adaptive_avg_pool2d(feat_map, (8,8))  # [1,C,8,8]
            tokens = fmap_ds.view(1, fmap_ds.size(1), -1).permute(0,2,1)
            if i == 0:
                debug['tokens'] = tokens[0].detach().cpu()

            # positional encoding
            cy, cx = r.centroid
            norm = torch.tensor(
                    [cx / label_map.shape[1], cy / label_map.shape[0]],
                    dtype = torch.float32,
                    device = self.device
                                                  ).unsqueeze(0)
            pos = self.pos_proj(norm)
            if i == 0:
                debug['pos'] = pos[0].detach().cpu()

            tokens = tokens + pos.unsqueeze(1)
            all_tokens.append(tokens)

        if not all_tokens:
            return None

        seq = torch.cat(all_tokens, dim=1)
        preds = self.transformer(seq).squeeze(-1)
        return preds, y_true, debug

    def training_step(self, batch, batch_idx):
        out = self._step(*batch)
        if out is None:
            return None
        preds, y_true, debug = out
        loss = self.loss_func(preds, y_true)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mae", self.train_mae(preds, y_true), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self._step(*batch)
        if out is None:
            return None
        preds, y_true, debug = out
        loss = self.loss_func(preds, y_true)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae(preds, y_true), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_mae"}}


if __name__ == "__main__":
    # 1) Load data
    train_loader, val_loader = highres_loader(g_dir, g_ann, patch_level=5, n_segments=50)

    # 2) Instantiate
    model = LitNetwork(cnn_option)

    # 3) One-off debug
    batch = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        _, _, debug = model._step(*batch)
        if debug:
            from torchvision.utils import make_grid, save_image
            # log first 64 CNN channels
            fm = debug['cnn_output'][0]
            grid = make_grid(fm[:64].unsqueeze(1), nrow=8, normalize=True, scale_each=True)
            save_image(grid, 'cnn_output_64ch.png')

    # 4) Train
    callbacks = []
    if use_checkpoint:
        callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val_mae", mode="min", save_top_k=1))
    logger = pl_loggers.TensorBoardLogger("../../logs/rs_regression")
    debug_cb = DebugCallback(n_segments=50, every_n_epochs=1)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        callbacks=[*callbacks, debug_cb],
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)


