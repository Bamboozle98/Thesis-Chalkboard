import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
from torchvision.utils import save_image
from skimage.segmentation import mark_boundaries
from skimage.measure    import regionprops

class CamDebugCallback(pl.Callback):
    def __init__(self, n_segments, every_n_epochs=1):
        super().__init__()
        self.n_segments     = n_segments
        self.every_n_epochs = every_n_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # only debug on first batch of each epoch
        if batch_idx != 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return

        # unpack your batch‐dict
        sp_map       = batch["sp_map"].squeeze(0)   # [Hc,Wc]
        patch_tensor = batch["patch"]              # [3,Hc,Wc]
        label        = batch["label"]              # tensor([0]) or [1]
        meta         = batch.get("meta", {})
        slide_id     = meta.get("filename", "unknown").replace(".tif", "")
        level        = meta.get("level", None)
        x0, y0       = meta.get("x0", 0), meta.get("y0", 0)

        # set up debug folder
        epoch   = trainer.current_epoch
        out_dir = f"debug_epoch_{epoch:02d}_{slide_id}"
        os.makedirs(out_dir, exist_ok=True)

        # 1) Save full superpixel overlay
        mean = pl_module.mean
        std  = pl_module.std
        vis_full = (patch_tensor * std + mean).clamp(0,1)[0].cpu().permute(1,2,0).numpy()
        overlay_full = mark_boundaries(vis_full, sp_map.cpu().numpy(), color=(1,1,0))
        plt.imsave(f"{out_dir}/full_sp_map_overlay.png", overlay_full)

        # 2) Run one _step pass to get preds & debug patches
        pl_module.eval()
        with torch.no_grad():
            # note: _step expects sp_map batch‐dim, so unsqueeze
            out = pl_module._step(sp_map.unsqueeze(0), patch_tensor, label)
        if out is None:
            return
        preds, truth, debug = out

        # 3) Save each SP patch and overlay
        Hc, Wc = sp_map.shape
        label_map = sp_map.cpu().numpy()
        props     = regionprops(label_map)

        for r in props:
            minr, minc, maxr, maxc = r.bbox
            sp_id = r.label

            # extract & unnormalize patch
            t = patch_tensor[:, minr:maxr, minc:maxc].unsqueeze(0).to(pl_module.device)
            if t.mean().item() < 0.05:
                continue
            vis_patch = (t * std + mean).clamp(0,1)[0].cpu()

            # save the raw patch
            save_image(vis_patch, f"{out_dir}/SP{sp_id:02d}_patch.png")

            # overlay boundaries
            np_patch = vis_patch.permute(1,2,0).numpy()
            mask = (label_map[minr:maxr, minc:maxc] == sp_id)
            overlay = mark_boundaries(np_patch, mask.astype(int), color=(1,1,0))
            plt.imsave(f"{out_dir}/SP{sp_id:02d}_overlay.png", overlay)

        # 4) log prediction vs truth as two simple scalars
        writer = trainer.logger.experiment
        gs = trainer.global_step
        writer.add_scalar("debug_pred_vs_true/pred", preds.item(), gs)
        writer.add_scalar("debug_pred_vs_true/true", truth.item(), gs)

        pl_module.train()
