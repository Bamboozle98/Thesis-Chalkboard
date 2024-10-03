import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image


def create_superpixel_image(image, n_segments):

    # Convert image to numpy array
    image_np = np.array(image)

    # Apply SLIC algorithm to generate superpixels
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=0)  # Start labels at 0

    return segments

