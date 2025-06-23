import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
from skimage.segmentation import slic
import torch
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.util import img_as_float
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage.segmentation import slic
import torch
from skimage.io import imread


def create_superpixel_image(image, n_segments):
    """
    Applies SLIC superpixel segmentation to an input image.

    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C).
        n_segments (int): Number of superpixels to generate.

    Returns:
        torch.Tensor: Superpixel label map (H x W), values from 0 to n_segments-1.
    """
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=10, start_label=0)
    return torch.from_numpy(segments).long()


def visualize_superpixel_overlay(image, superpixel_map, title="Superpixel Visualization", save_path=None):
    """
    Overlays superpixel boundaries on the original image and shows/saves the result.

    Args:
        image (np.ndarray): Original image (H x W x C).
        superpixel_map (torch.Tensor): Superpixel label map (H x W).
        title (str): Title for the plot.
        save_path (str, optional): If provided, saves the image to this path.
    """
    boundaries = mark_boundaries(image, superpixel_map.cpu().numpy(), color=(1, 1, 0))  # Red lines
    plt.figure(figsize=(6, 6))
    plt.imshow(boundaries)
    plt.axis('off')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def puzzle_superpixels_by_centroid_gap(image, superpixel_map, canvas_scale=2.0, gap_scale=1.1, save_path=None):
    """
    Places superpixels on a fixed large canvas based on scaled centroid positions,
    with controllable spacing between them.

    Args:
        image (np.ndarray): Original image (H, W, 3)
        superpixel_map (torch.Tensor): Superpixel label map (H, W)
        canvas_scale (float): Size of the output canvas relative to the original
        gap_scale (float): Spread factor for centroid spacing (e.g., 1.1 is tight, 2.0 is wide)
        save_path (str): If provided, saves the output
    """
    image = img_as_ubyte(image)
    sp_map = superpixel_map.cpu().numpy()
    h, w, _ = image.shape

    canvas_h = int(h * canvas_scale)
    canvas_w = int(w * canvas_scale)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white background

    regions = regionprops(sp_map)
    centroids = np.array([r.centroid for r in regions])

    # Normalize centroids to [0, 1]
    norm_centroids = (centroids - centroids.min(axis=0)) / (centroids.max(axis=0) - centroids.min(axis=0))

    # Scale them into the canvas using tighter or looser gap control
    mapped_centroids = (norm_centroids * [canvas_h / gap_scale, canvas_w / gap_scale]).astype(int)

    for region, (cy, cx) in zip(regions, mapped_centroids):
        minr, minc, maxr, maxc = region.bbox
        mask = (sp_map[minr:maxr, minc:maxc] == region.label)
        crop = image[minr:maxr, minc:maxc]
        masked_crop = crop * mask[..., None]

        ch, cw = masked_crop.shape[:2]
        tr = cy - ch // 2
        tc = cx - cw // 2

        # Clip paste region
        tr_clip = max(tr, 0)
        tc_clip = max(tc, 0)
        br_clip = min(tr + ch, canvas_h)
        bc_clip = min(tc + cw, canvas_w)

        # Clip crop region accordingly
        crop_tr = tr_clip - tr
        crop_tc = tc_clip - tc
        crop_br = crop_tr + (br_clip - tr_clip)
        crop_bc = crop_tc + (bc_clip - tc_clip)

        masked_crop_clipped = masked_crop[crop_tr:crop_br, crop_tc:crop_bc]
        mask_clipped = mask[crop_tr:crop_br, crop_tc:crop_bc]

        canvas_slice = canvas[tr_clip:br_clip, tc_clip:bc_clip]
        for c in range(3):
            channel = masked_crop_clipped[..., c]
            canvas_slice[..., c][mask_clipped] = channel[mask_clipped]

    # Show and optionally save
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis('off')
    plt.title(f"Puzzle Spacing (gap scale = {gap_scale})")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()



# Example usage:
if __name__ == "__main__":
    from skimage.io import imread

    # Load image (replace with your own path)
    image_path = r"C:\Users\cbran\Documents\ECU\Thesis\Abyssinian_6.jpg"
    image = imread(image_path)
    file_name = os.path.basename(image_path)
    file_name_wo_ext = os.path.splitext(file_name)[0]

    n_segments = 50

    # Create superpixel map
    sp_map = create_superpixel_image(image, n_segments=n_segments)

    # Visualize
    visualize_superpixel_overlay(image, sp_map, title="~50 Superpixels", save_path=f"C:/Users/cbran/Documents/ECU/Thesis/{file_name_wo_ext}_sp_{n_segments}.png")

    puzzle_superpixels_by_centroid_gap(image, sp_map, canvas_scale=2.0, gap_scale=1.75, save_path=f"C:/Users/cbran/Documents/ECU/Thesis/seperated_{file_name_wo_ext}_sp_{n_segments}.png")
