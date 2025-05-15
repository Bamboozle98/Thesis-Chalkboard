import numpy as np
from skimage.segmentation import slic
from PIL import Image


def create_superpixel_image(thumb: Image.Image, full_dims: tuple[int, int], n_segments: int):
    """
    1) Run SLIC on the thumbnail
    2) Upsample the integer label map to full_dims via nearest‐neighbor
    """
    # thumbnail → H×W×3 numpy
    thumb_np = np.array(thumb)
    # slic at thumbnail size
    segments = slic(
        thumb_np,
        n_segments=n_segments,
        compactness=10,
        start_label=0
    )  # shape (h_thumb, w_thumb)

    # now blow it up to (full_W, full_H)
    # PIL wants (width, height)
    seg_img = Image.fromarray(segments.astype(np.uint16))
    seg_up = seg_img.resize(full_dims, resample=Image.NEAREST)

    return np.array(seg_up, dtype=np.int32)  # shape (full_h, full_w)
