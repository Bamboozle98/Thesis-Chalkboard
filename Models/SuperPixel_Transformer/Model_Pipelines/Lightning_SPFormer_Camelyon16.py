import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from Models.SuperPixel_Transformer.config import num_epochs, learning_rate, dataset_option, cnn_option, use_checkpoint, \
    wsi_train, wsi_test, wsi_test_truth
from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon16DataLoader import load_camelyon
from Models.SuperPixel_Transformer.DataLoaders.Data_Loader import load_dataset
import numpy as np
from skimage.measure import label as sklabel, regionprops
import openslide
from PIL import Image

# CUDA optimization
torch.set_float32_matmul_precision('high')


def cnn_selection(option):
    if option == "ResNet18":
        return ResNet18()
    elif option == "ResNet50":
        return ResNet50()
    elif option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    else:
        raise ValueError(f"Invalid CNN selection: {option}")


def map_bbox_to_full_resolution(bbox, scale_factors):
    """
    Maps a bounding box from thumbnail coordinates to full-resolution coordinates.

    Args:
        bbox (tuple): (x, y, w, h) in thumbnail coordinates.
        scale_factors (tuple): (scale_x, scale_y) factors.

    Returns:
        tuple: (x_full, y_full, w_full, h_full)
    """
    x, y, w, h = bbox
    scale_x, scale_y = scale_factors
    return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))


def extract_high_res_patch(wsi_path, bbox_full, max_size=(1024, 1024)):
    """
    Extract a high-resolution patch from a slide given full-resolution bounding box coordinates.
    If the patch is larger than max_size, it will be resized.

    Args:
        wsi_path (str): Path to the slide.
        bbox_full (tuple): (x, y, w, h) in full-resolution coordinates.
        max_size (tuple): Maximum size (width, height) for the patch.

    Returns:
        PIL Image patch.
    """
    slide = openslide.OpenSlide(wsi_path)
    x, y, w, h = bbox_full
    patch = slide.read_region((x, y), 0, (w, h)).convert("RGB")
    slide.close()

    # If the patch is larger than max_size in either dimension, downsample it:
    if patch.width > max_size[0] or patch.height > max_size[1]:
        # Preserve aspect ratio
        patch.thumbnail(max_size, Image.LANCZOS)
    return patch


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()
        n_classes = 37  # Adjust as necessary
        self.feature_dim = 512  # Assumed CNN output channels (after global pooling)
        self.cnn = cnn_selection(cnn_option)  # CNN for processing high-res patches
        self.transformer = TransformerEncoder(self.feature_dim, n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')
        # Define a transform for high-resolution patches (you can add normalization if needed)
        self.full_res_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def forward(self, tokens):
        """
        tokens: tensor of shape [batch, num_tokens, feature_dim]
        """
        predictions = self.transformer(tokens)
        return predictions

    def training_step(self, data, batch_idx):
        # Unpack items returned by the dataloader:
        # global_superpix_map: computed on the thumbnail (shape: [1, H_thumb, W_thumb])
        # thumbnail: the downsampled image used for global segmentation (transformed)
        # label: the slide-level label
        # scale_factors: (scale_x, scale_y) mapping thumbnail->full-res
        # full_dims: (full_width, full_height)
        # wsi_path: file path to the full-resolution slide
        global_superpix_map, thumbnail, label, scale_factors, full_dims, wsi_path = data

        # Compute regions from the superpixel map (assumed to be a 2D label map)
        sp_map_np = global_superpix_map.squeeze().cpu().numpy().astype(np.int32)
        regions = regionprops(sklabel(sp_map_np))

        patch_features = []
        for region in regions:
            # Get bounding box in thumbnail coordinates (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = region.bbox
            # Convert to (x, y, width, height)
            bbox_thumb = (minc, minr, maxc - minc, maxr - minr)
            # Map thumbnail bounding box to full resolution:
            bbox_full = map_bbox_to_full_resolution(bbox_thumb, scale_factors)
            # Extract high-resolution patch:
            patch = extract_high_res_patch(wsi_path, bbox_full, max_size=(1024,1024))
            # Apply full-resolution transform:
            patch_tensor = self.full_res_transform(patch).to(next(self.cnn.parameters()).device)  # shape: [C, H_patch, W_patch]
            # Process patch with CNN; add batch dimension:
            cnn_out = self.cnn(patch_tensor.unsqueeze(0))  # e.g. shape: [1, feature_dim, H_out, W_out]
            # Globally pool to get a feature vector:
            feat = torch.nn.functional.adaptive_avg_pool2d(cnn_out, (1, 1)).squeeze()  # shape: [feature_dim]
            patch_features.append(feat)

        if len(patch_features) == 0:
            # Fallback if no regions were detected
            tokens = torch.zeros((1, 1, self.feature_dim), device=self.device)
        else:
            # Stack features into a token tensor: shape [num_regions, feature_dim]
            tokens = torch.stack(patch_features)
            # Add a batch dimension: shape [1, num_regions, feature_dim]
            tokens = tokens.unsqueeze(0)

        # Pass tokens through transformer
        outs = self.forward(tokens)
        loss = self.loss_func(outs, label)
        self.log("train_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        global_superpix_map, thumbnail, label, scale_factors, full_dims, wsi_path = data
        wsi_path = wsi_path[0]  # Extract the string from the batch (assuming batch_size=1)
        sp_map_np = global_superpix_map.squeeze().cpu().numpy().astype(np.int32)
        regions = regionprops(sklabel(sp_map_np))
        patch_features = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox  # (min_row, min_col, max_row, max_col)
            bbox_thumb = (minc, minr, maxc - minc, maxr - minr)  # Correct conversion to (x, y, w, h)
            bbox_full = map_bbox_to_full_resolution(bbox_thumb, scale_factors)
            patch = extract_high_res_patch(wsi_path, bbox_full, max_size=(1024,1024))
            patch_tensor = self.full_res_transform(patch).to(next(self.cnn.parameters()).device)
            cnn_out = self.cnn(patch_tensor.unsqueeze(0))
            feat = torch.nn.functional.adaptive_avg_pool2d(cnn_out, (1, 1)).squeeze()
            patch_features.append(feat)

        if len(patch_features) == 0:
            tokens = torch.zeros((1, 1, self.feature_dim), device=self.device)
        else:
            tokens = torch.stack(patch_features).unsqueeze(0)
        outs = self.forward(tokens)
        self.val_acc(outs, label)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }


if __name__ == "__main__":
    model = LitNetwork()
    device = "gpu"

    if dataset_option in ['oxford_pets', 'image_net']:
        train_loader, val_loader, class_names = load_dataset(dataset_name=dataset_option)
        logger = pl_loggers.TensorBoardLogger(save_dir="../../my_logs")
    elif dataset_option == 'camelyon_16':
        train_loader, val_loader = load_camelyon()
        logger = pl_loggers.TensorBoardLogger(save_dir="../../camelyon_16_logs")
    else:
        raise ValueError("Error with dataset loader selection.")

    if use_checkpoint:
        checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, callbacks=[checkpoint], logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        print("Training Model from Default weights.")
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
