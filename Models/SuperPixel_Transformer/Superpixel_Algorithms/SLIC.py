import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image
from Models.SuperPixel_Transformer.config import num_superpixels

def permute_image(x):
    return x.permute(1, 2, 0)

def create_superpixel_image(image, n_segments):

    # Convert image to numpy array
    # print("pre numpy")
    # print(image.size())
    image = permute_image(image)
    image = np.array(image)
    # print("post numpy")

    # Apply SLIC algorithm to generate superpixels
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)  # Start labels at 0

    return segments

