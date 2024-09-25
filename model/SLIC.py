import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image


def create_superpixel_image(image, n_segments):
    """
    Applies SLIC to an image and returns both the superpixel map and a visualization of the superpixel image.

    Parameters:
    - image (PIL.Image): Input image.
    - n_segments (int): Number of superpixels to generate.

    Returns:
    - segments (np.ndarray): Superpixel map where each pixel is assigned a superpixel label.
    - PIL.Image: Image with superpixels for visualization (optional, can be removed if not needed).
    """
    # Convert image to numpy array
    image_np = np.array(image)

    # Apply SLIC algorithm to generate superpixels
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=0)  # Start labels at 0

    return segments,

