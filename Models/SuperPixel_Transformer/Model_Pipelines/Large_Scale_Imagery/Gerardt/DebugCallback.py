import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from skimage.segmentation import mark_boundaries
from skimage.measure    import regionprops

class DebugCallback(pl.Callback):
    def __init__(self, n_segments, every_n_epochs=1):
        super().__init__()
        self.n_segments     = n_segments
        self.every_n_epochs = every_n_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx != 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return

        sp_map, patch_tensor, rs, path, lvl, x0_c, y0_c = batch

        sp_map = sp_map.squeeze(0)
        label_map = sp_map.cpu().numpy()
        regions = list(torch.tensor(np.unique(label_map)).int().tolist())

        # Unnormalize prep
        mean = torch.tensor([0.485, 0.456, 0.406], device=pl_module.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=pl_module.device).view(1, 3, 1, 1)

        # Set up debug directory
        slide_id = os.path.basename(path).replace('.tiff', '')
        epoch = trainer.current_epoch
        out_dir = f"debug/epoch_{epoch:02d}_{slide_id}"
        os.makedirs(out_dir, exist_ok=True)

        # Un-normalize the entire crop:
        vis_full = (patch_tensor * pl_module.std + pl_module.mean).clamp(0, 1)[0].cpu()  # [3,Hc,Wc]
        # NumPy versions:
        np_full = vis_full.permute(1, 2, 0).numpy()  # [Hc,Wc,3]
        sp_full = sp_map.cpu().numpy()  # [Hc,Wc] superpixel IDs

        # Overlay boundaries:
        overlay_full = mark_boundaries(np_full, sp_full, color=(1, 1, 0))

        # Save to disk
        os.makedirs(out_dir, exist_ok=True)
        plt.imsave(f"{out_dir}/full_sp_map_overlay.png", overlay_full)

        # Step through model once
        pl_module.eval()
        with torch.no_grad():
            out = pl_module._step(sp_map.unsqueeze(0), patch_tensor, rs, path, lvl, x0_c, y0_c)
        if out is None:
            return
        preds, truth, debug = out

        # CNN input patches
        Hc, Wc = sp_map.shape
        label_map = sp_map.cpu().numpy()
        props = list(regionprops(label_map))

        for r in props:
            minr, minc, maxr, maxc = r.bbox
            sp_id = r.label
            cy, cx = map(int, r.centroid)

            t = patch_tensor[:, minr:maxr, minc:maxc].unsqueeze(0).to(pl_module.device)
            if t.mean().item() < 0.05:
                continue

            # Unnormalize
            vis_patch = (t * std + mean).clamp(0, 1)[0].detach().cpu()

            # Save input patch
            patch_path = os.path.join(out_dir, f"SP{sp_id:02d}_patch.png")
            save_image(vis_patch, patch_path)

            # Overlay mask boundary
            mask = (label_map[minr:maxr, minc:maxc] == sp_id)
            np_patch = vis_patch.permute(1, 2, 0).numpy()
            overlay = mark_boundaries(np_patch, mask.astype(int), color=(1,1,0))
            fig, ax = plt.subplots()
            ax.imshow(overlay)
            ax.axis("off")
            ax.set_title(f"SP#{sp_id}")
            plt.tight_layout()
            overlay_path = os.path.join(out_dir, f"SP{sp_id:02d}_overlay.png")
            plt.savefig(overlay_path)
            plt.close()

        # Log prediction vs ground truth
        trainer.logger.experiment.add_scalars(
            "debug/pred_vs_true",
            {"pred": preds.item(), "true": truth.item()},
            global_step=trainer.global_step
        )
