import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from Models.SuperPixel_Transformer.config import num_epochs, learning_rate
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet101 import ResNet101
from Models.SuperPixel_Transformer.DataLoaders.Data_Loader import load_dataset
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.config import dataset_option, cnn_option, use_checkpoint, num_superpixels
from skimage.measure import regionprops

# CUDA optimization
torch.set_float32_matmul_precision('high')


def cnn_selection(option):
    if option == "ResNet18":
        return ResNet18()
    elif option == "ResNet50":
        return ResNet50()
    elif option == "ResNet101":
        return ResNet101()
    elif option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    else:
        raise ValueError(f"Invalid CNN selection: {option}")


def get_centroids(sp_map, num_segments):
    centroids = np.zeros((num_segments, 2), dtype=np.float32)
    # precompute row & col indices
    rows, cols = np.indices(sp_map.shape)
    for label in range(num_segments):
        mask = (sp_map == label)
        if mask.any():
            # mean row and mean col of all pixels with this label
            centroids[label, 0] = rows[mask].mean()
            centroids[label, 1] = cols[mask].mean()
        else:
            # if slic never produced this label, leave at (0,0)
            # this is due to SLIC not generating a fixed number of superpixels, and it's tendency ot make 'junk'
            # superpixels occasionally
            pass

    return torch.from_numpy(centroids)  # → shape: [num_segments, 2]

def covariance(x_mean: float, y_mean: float, coords: torch.Tensor) -> torch.Tensor:
    """
    coords: Tensor of shape [N,2], each row = (row_idx, col_idx) of one pixel
    x_mean, y_mean: scalar centroid coordinates (same units as coords)
    Returns: Tensor[3] = [cov_xx, cov_xy, cov_yy]
    """
    # center the coords
    diffs = coords - torch.tensor([x_mean, y_mean], device=coords.device)  # [N,2]

    # unbiased covariance estimates
    # using (N-1) in denominator
    N = coords.shape[0]
    if N <= 1:
        # degenerate superpixel (rare)—just return zeros
        return torch.zeros(3, device=coords.device)

    cov_xx = (diffs[:, 0].pow(2).sum()     ) / (N - 1)
    cov_xy = (diffs[:, 0] * diffs[:, 1]).sum() / (N - 1)
    cov_yy = (diffs[:, 1].pow(2).sum()     ) / (N - 1)

    return torch.stack([cov_xx, cov_xy, cov_yy])  # [3]


class LitNetwork(pl.LightningModule):
    def __init__(self, n_classes):
        super(LitNetwork, self).__init__()
        self.num_classes = n_classes  # Adjust number of classes as necessary
        out_channels = 512
        self.cnn = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(out_channels, n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')
        self.num_segments = num_superpixels
        self.pos_proj = torch.nn.Linear(2, out_channels)
        self.cov_proj = torch.nn.Linear(3, out_channels)
        self.cov_gate = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, superpixel_map):
        out_image = self.cnn(x)                 # [b, d, r, c]
        b, d, r, c = out_image.shape
        dev = out_image.device

        # 1) scatter to superpix vectors
        exp_map = superpixel_map.unsqueeze(1).expand_as(out_image)
        buf = torch.zeros((b, d, self.num_segments), device=dev)
        sp = torch.scatter_reduce(
            buf, 2,
            exp_map.reshape(b, d, r*c),
            out_image.reshape(b, d, r*c),
            reduce='mean'
        ).permute(0, 2, 1)  # [b, num_pix, d]

        # Pre-allocate lists
        batch_cents = []
        batch_cov   = []

        for i in range(b):
            sp_map_i = superpixel_map[i].cpu().numpy()    # [r, c]
            # compute centroids once
            cents_px = get_centroids(sp_map_i, self.num_segments)  # [num_pix, 2]
            cents_px = cents_px.to(dev)

            # positional (normalized)
            cents_norm = cents_px / torch.tensor([r, c], device=dev)
            batch_cents.append(cents_norm)

            # covariance
            covs = []
            for lbl in range(self.num_segments):
                ys, xs = torch.nonzero(superpixel_map[i]==lbl, as_tuple=True)
                coords  = torch.stack([ys, xs], dim=1).float()
                covs.append(covariance(
                    cents_px[lbl,0].item(),
                    cents_px[lbl,1].item(),
                    coords
                ))
            batch_cov.append(torch.stack(covs, dim=0))  # [num_pix,3]

        # stack & project
        batch_cents = torch.stack(batch_cents, dim=0)    # [b, num_pix, 2]
        pos_enc     = self.pos_proj(batch_cents)         # [b, num_pix, d]

        cov_feat    = torch.stack(batch_cov, dim=0)      # [b, num_pix, 3]
        cov_enc     = self.cov_proj(cov_feat)            # [b, num_pix, d]
        cov_enc     = torch.tanh(self.cov_gate) * cov_enc

        # fuse & forward
        superpix_vectors = sp + pos_enc + cov_enc
        return self.transformer(superpix_vectors)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",  # Monitor validation accuracy
                "interval": "epoch",  # Adjust learning rate every epoch
                "frequency": 1  # Frequency of learning rate adjustment
            }
        }

    # self.transformer optimizer to isolate the Resnet18 weights
    def training_step(self, data, batch_idx):
        superpix_map, im, label = data[0], data[1], data[2]
        outs = self.forward(im, superpix_map)
        loss = self.loss_func(outs, label)
        self.log("train_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, val_data, batch_idx):
        superpix_map, im, label = val_data[0], val_data[1], val_data[2]
        outs = self.forward(im, superpix_map)
        self.val_acc(outs, label)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


if __name__ == "__main__":
    device = "gpu"

    if dataset_option in ['oxford_pets', 'image_net', 'high_res']:
        train_loader, val_loader, class_names, n_classes = load_dataset(dataset_name=dataset_option)
        logger = pl_loggers.TensorBoardLogger(save_dir="../../../my_logs")
    else:
        raise ValueError("Error with dataset loader selection.")

    model = LitNetwork(n_classes=n_classes)

    if use_checkpoint:
        checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, callbacks=[checkpoint], logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        print("Training Model from Default weights.")
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
