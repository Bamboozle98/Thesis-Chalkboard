import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from Models.SuperPixel_Transformer.PNP_CNNs.Resnet101 import ResNet101
from Models.SuperPixel_Transformer.config import num_epochs, learning_rate
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet101 import ResNet101
from Models.SuperPixel_Transformer.DataLoaders.Data_Loader import load_dataset
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.config import dataset_option, cnn_option, use_checkpoint

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


def get_centroids(superpixel_map, image, num_superpixels=50):
    """
    Computes centroids from a superpixel map.
    Assumes:
      - superpixel_map is shape (b, 1, rows, cols) or (b, rows, cols)
      - image is shape (b, d, rows, cols)
    The function returns centroids of shape (b, num_superpixels, 2).
    """
    with torch.no_grad():
        # Ensure superpixel_map has 4 dimensions.
        if superpixel_map.dim() == 3:
            superpixel_map = superpixel_map.unsqueeze(1)

        # Isolate Dimension Values
        b, _, rows, cols = superpixel_map.shape
        b_img, d, r, c = image.shape

        # Assertion for debugging
        assert b == b_img, f"Batch size mismatch: {b} vs {b_img}"
        assert rows == r and cols == c, "Spatial dimensions must match between superpixel_map and image."

        # Flatten superpixel_map to (b, 1, rows*cols)
        superpixel_map = superpixel_map.view(b, 1, rows * cols)

        # Shift boundaries so that a value of 0 maps to bucket 0 (instead of -1)
        # Flattened Superpixel dimensions were represented as a column of all pixel values in an image (224x224).
        # Must be organized into 50 superpixel assignments so that buffer and Superpixel map have matching dimensions
        # for scatter.
        boundaries = torch.linspace(-1e-6, superpixel_map.max().float(), num_superpixels + 1,
                                    device=superpixel_map.device)
        superpixel_map = (torch.bucketize(superpixel_map.float(), boundaries) - 1).long()

        # Convert R_vals and C_vals to appropriate datatype.
        # Create row and column indices with shape (1, rows*cols)
        r_vals = torch.arange(rows, device=superpixel_map.device, dtype=torch.float32)
        c_vals = torch.arange(cols, device=superpixel_map.device, dtype=torch.float32)
        r_vals, c_vals = torch.meshgrid(r_vals, c_vals, indexing='ij')
        r_vals = r_vals.reshape(1, -1)
        c_vals = c_vals.reshape(1, -1)

        # Expand indices to shape (b, 1, rows*cols) to match Superpixel map and Buffer.
        r_vals = r_vals.expand(b, 1, -1)
        c_vals = c_vals.expand(b, 1, -1)

        # Create a positional buffer with shape (b, 1, num_superpixels)
        pos_buffer = torch.zeros(b, 1, num_superpixels, device=superpixel_map.device, dtype=torch.float32)

        # Compute the mean row and column indices for each superpixel via scatter_reduce
        center_rs = torch.scatter_reduce(pos_buffer, 2, superpixel_map, r_vals, reduce='mean')
        center_cs = torch.scatter_reduce(pos_buffer, 2, superpixel_map, c_vals, reduce='mean')

        center_rs_maps = superpixel_map[center_rs]

        top = pos_buffer[:,:,0] - center_rs_maps
        cov = torch.scatter_reduce(top, 2, superpixel_map, center_cs, reduce='mean')

        # Concatenate row and column centroids: shape (b, 2, num_superpixels)
        centroids = torch.cat((center_rs, center_cs), dim=1)
        # Permute to shape (b, num_superpixels, 2)
        centroids = centroids.permute(0, 2, 1)

    return centroids


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()
        n_classes = 37  # Adjust number of classes as necessary
        out_channels = 512
        self.num_dimensions = out_channels
        self.cnn = SuperpixelCNN(in_channels=3, out_channels=out_channels)
        self.res_cnn = ResNet18()
        self.transformer = TransformerEncoder(out_channels, n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')

        self.projection = torch.nn.Linear(2, out_channels)

    def forward(self, x, superpixel_map):
        out_image = self.res_cnn(x)  # shape: (b, 512, rows, cols)

        # For feature aggregation, expand the superpixel map as needed.
        superpixel_map_expanded = superpixel_map.unsqueeze(1).expand_as(out_image)

        b, d, r, c = out_image.shape
        num_pix = 50  # number of superpixels

        buffer = torch.zeros((b, d, num_pix), device=out_image.device)

        # Clone the result to break any in-place linkage.
        superpix_vectors = torch.scatter_reduce(
            buffer, 2, superpixel_map_expanded.view(b, d, r * c),
            out_image.view(b, d, r * c), reduce='mean'
        ).clone()

        superpix_vectors = superpix_vectors.permute(0, 2, 1)  # shape: (b, num_superpixels, d)

        # Compute centroids (already no_grad inside get_centroids)
        centroids = get_centroids(superpixel_map, out_image, num_pix)  # shape: (b, num_superpixels, 2)
        pos_encs = self.projection(centroids)  # shape: (b, num_superpixels, out_channels)

        superpix_vectors += pos_encs

        predictions = self.transformer(superpix_vectors)
        return predictions

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
    train_loader, val_loader, class_names = load_dataset(dataset_name=dataset_option)
    model = LitNetwork()
    logger = pl_loggers.TensorBoardLogger(save_dir="../../PE_oxford_logs")

    device = "gpu"

    if use_checkpoint:
        checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, callbacks=[checkpoint], logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elif not use_checkpoint:
        print("Training Model from Default weights.")
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        print("Error with determining Checkpoint usage.")
