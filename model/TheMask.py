import numpy as np


def mask_superpixel(image, segments, superpixel_id):
    # Create mask for the given superpixel
    mask = np.zeros_like(image)
    mask[segments == superpixel_id] = image[segments == superpixel_id]
    return mask
