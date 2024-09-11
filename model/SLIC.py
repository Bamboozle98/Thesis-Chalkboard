import numpy as np
from skimage.segmentation import slic
from PIL import Image


def create_superpixel_image(image, n_segments):
    """
    Applies SLIC to an image and returns a new image where each superpixel has a unique color. This is for testing
    purposes to ensure that the transformer is using the super pixels as tokens. Ideally, we will convolve each super
    pixel into a vector of standardized dimensions.

    Parameters:
    - image (PIL.Image): Input image.
    - n_segments (int): Number of superpixels to generate.

    Returns:
    - PIL.Image: Image with superpixels.
    """
    image_np = np.array(image)

    # Apply SLIC
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=1)

    # Convert segments to image where each superpixel is a unique color
    unique_segments = np.unique(segments)
    superpixel_image = np.zeros_like(image_np)

    for i, seg_id in enumerate(unique_segments):
        mask = (segments == seg_id)
        superpixel_image[mask] = np.random.randint(0, 255, size=3)  # Random color

    return Image.fromarray(superpixel_image.astype('uint8'))
