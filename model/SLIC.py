import numpy as np
from skimage.segmentation import slic
from PIL import Image


def create_superpixel_image(image, n_segments):
    """
    Applies SLIC to an image and returns a new image. This is for testing the computational time difference of using
    SLIC in preprocessing vs training time. Ideally, we will convolve each super pixel into a vector of standardized
    dimensions.

    Parameters:
    - image (PIL.Image): Input image.
    - n_segments (int): Number of superpixels to generate.

    Returns:
    - PIL.Image: Image with superpixels.
    """
    image_np = np.array(image)

    # Apply SLIC
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=1)

    unique_segments = np.unique(segments)
    superpixel_image = np.zeros_like(image_np)

    return Image.fromarray(superpixel_image.astype('uint8'))
