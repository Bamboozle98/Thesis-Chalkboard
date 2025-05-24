import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image
from Models.SuperPixel_Transformer.config import num_superpixels
import torch

def permute_image(x):
    return x.permute(1, 2, 0)

def create_superpixel_image(image, n_segments):
    """
    Args:
        image (torch.Tensor or np.ndarray): If torch.Tensor, assumed to be [C, H, W]
    Returns:
        segments (torch.Tensor): [H, W] integer mask of superpixel regions
    """
    # If tensor, convert to channel-last numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # [H, W, C]

    # Apply SLIC algorithm
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)

    return torch.from_numpy(segments).long()  # Ensure PyTorch tensor output


