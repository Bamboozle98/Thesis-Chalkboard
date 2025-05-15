import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from Models.SuperPixel_Transformer.config import num_epochs, learning_rate, dataset_option, cnn_option, use_checkpoint, \
    wsi_dir
from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet import CustomResNet20
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon_16.Camelyon16DataLoader import load_camelyon
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
    elif option == "CustomResNet":
        return CustomResNet20()
    elif option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    else:
        raise ValueError(f"Invalid CNN selection: {option}")

def on_validation_epoch_end(self):
    self.val_acc.reset()


def map_bbox_to_full_resolution(bbox, scale_factors, full_dims):
    """
    Maps a bounding box from thumbnail to full-resolution coordinates,
    ensuring the values are cast to Python numbers.

    Args:
        bbox (tuple): (x, y, w, h) in thumbnail coordinates (x, y, etc. can be tensors or numbers).
        scale_factors (tuple): (scale_x, scale_y), factors to map thumbnail to full-res.
        full_dims (tuple): (full_width, full_height) of the slide.

    Returns:
        tuple: (x_full, y_full, w_full, h_full) clamped into valid full-res coordinates.
    """
    x, y, w, h = bbox
    scale_x, scale_y = scale_factors
    full_width, full_height = full_dims

    # Convert inputs to floats if they're tensors or not plain numbers.
    x = float(x) if not isinstance(x, (float, int)) else x
    y = float(y) if not isinstance(y, (float, int)) else y
    w = float(w) if not isinstance(w, (float, int)) else w
    h = float(h) if not isinstance(h, (float, int)) else h
    scale_x = float(scale_x) if not isinstance(scale_x, (float, int)) else scale_x
    scale_y = float(scale_y) if not isinstance(scale_y, (float, int)) else scale_y

    # Compute full resolution coordinates.
    x_full = max(0, int(round(x * scale_x)))
    y_full = max(0, int(round(y * scale_y)))
    w_full = max(1, int(round(w * scale_x)))
    h_full = max(1, int(round(h * scale_y)))

    # Clamp so the patch does not exceed the slide dimensions.
    x_full = min(x_full, full_width - 1)
    y_full = min(y_full, full_height - 1)
    w_full = min(w_full, full_width - x_full)
    h_full = min(h_full, full_height - y_full)

    return x_full, y_full, w_full, h_full



def extract_high_res_patch(wsi_path, bbox_full, max_size=(224, 224)):
    """
    Extract a high-resolution patch from a slide given full-resolution bounding box coordinates.
    Returns a downsampled PIL Image if the patch is larger than max_size.

    Args:
        wsi_path (str): Path to the slide.
        bbox_full (tuple): (x, y, w, h) in full-resolution coordinates.
        max_size (tuple): Maximum size (width, height) for the patch.

    Returns:
        PIL Image: Extracted patch (or None if extraction fails).
    """
    slide = openslide.OpenSlide(wsi_path)
    x, y, w, h = bbox_full

    # If patch dimensions are invalid, exit early.
    if w <= 0 or h <= 0:
        print(f"Warning: invalid region dimensions: {bbox_full}")
        slide.close()
        return None

    try:
        patch = slide.read_region((x, y), 0, (w, h)).convert("RGB")
    except Exception as e:
        print(f"Error reading region at {(x, y, w, h)}: {e}")
        slide.close()
        return None

    slide.close()

    # If the patch is larger than max_size, downsample it preserving aspect ratio.
    if patch.width > max_size[0] or patch.height > max_size[1]:
        patch.thumbnail(max_size, Image.LANCZOS)
    return patch


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()
        n_classes = 2  # Adjust as necessary
        self.feature_dim = 512  # CNN output channels 
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
        global_superpix_map, thumbnail, label, scale_factors, full_dims, wsi_path = data
        wsi_path = wsi_path[0]

        sp_map_np = global_superpix_map.squeeze().cpu().numpy().astype(np.int32)
        regions = regionprops(sklabel(sp_map_np))

        patch_features = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bbox_thumb = (minc, minr, maxc - minc, maxr - minr)
            bbox_full = map_bbox_to_full_resolution(bbox_thumb, scale_factors, full_dims)
            # Extract high-resolution patch; may return None if the region is invalid.
            patch = extract_high_res_patch(wsi_path, bbox_full, max_size=(1024, 1024))
            if patch is None:
                # Skip this region if extraction failed
                continue
            # Apply full-resolution transform:
            patch_tensor = self.full_res_transform(patch).to(next(self.cnn.parameters()).device)
            # Process patch with CNN and globally pool features:
            cnn_out = self.cnn(patch_tensor.unsqueeze(0))
            feat = torch.nn.functional.adaptive_avg_pool2d(cnn_out, (1, 1)).squeeze()
            patch_features.append(feat)

        if len(patch_features) == 0:
            # Fallback if no valid regions were detected
            tokens = torch.zeros((1, 1, self.feature_dim), device=self.device)
        else:
            tokens = torch.stack(patch_features).unsqueeze(0)

        outs = self.forward(tokens)
        loss = self.loss_func(outs, label)
        self.log("train_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        global_superpix_map, thumbnail, label, scale_factors, full_dims, wsi_path = data
        wsi_path = wsi_path[0]
        sp_map_np = global_superpix_map.squeeze().cpu().numpy().astype(np.int32)
        regions = regionprops(sklabel(sp_map_np))

        patch_features = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bbox_thumb = (minc, minr, maxc - minc, maxr - minr)
            bbox_full = map_bbox_to_full_resolution(bbox_thumb, scale_factors, full_dims)
            patch = extract_high_res_patch(wsi_path, bbox_full, max_size=(1024, 1024))
            if patch is None:
                # Skip invalid patches
                continue
            patch_tensor = self.full_res_transform(patch).to(next(self.cnn.parameters()).device)
            cnn_out = self.cnn(patch_tensor.unsqueeze(0))
            feat = torch.nn.functional.adaptive_avg_pool2d(cnn_out, (1, 1)).squeeze()
            patch_features.append(feat)

        if len(patch_features) == 0:
            tokens = torch.zeros((1, 1, self.feature_dim), device=self.device)
        else:
            tokens = torch.stack(patch_features).unsqueeze(0)

        outs = self.forward(tokens)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor([label], device=self.device)
        elif label.ndim == 0:
            label = label.unsqueeze(0).to(self.device)
        else:
            label = label.to(self.device)

        acc = self.val_acc(outs, label)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    model = LitNetwork()
    device = "gpu"

    if dataset_option in ['oxford_pets', 'image_net']:
        train_loader, val_loader, class_names = load_dataset(dataset_name=dataset_option)
        logger = pl_loggers.TensorBoardLogger(save_dir="../../../my_logs")
    elif dataset_option == 'camelyon_16':
        train_loader, val_loader = load_camelyon(wsi_dir, transform=transform)
        logger = pl_loggers.TensorBoardLogger(save_dir="../../../camelyon_16_logs")
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
