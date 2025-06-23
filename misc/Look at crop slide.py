from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.measure import regionprops, label
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from skimage.segmentation import mark_boundaries


def crop_by_slic_and_color(arr, n_segments=500, rgb_thresh=0.85):
    """
    Crop an RGB image based on SLIC superpixels whose mean color is dark enough.

    Args:
        arr (np.ndarray): RGB image as ndarray [H, W, 3]
        n_segments (int): Number of SLIC superpixels
        rgb_thresh (float): Max average grayscale value (0-1) to keep superpixel

    Returns:
        np.ndarray: Cropped image region
    """
    # SLIC segmentation
    sp_map = slic(arr, n_segments=n_segments, start_label=1)

    # Convert to grayscale
    gray = rgb2gray(arr)

    # Build mask from filtered superpixels
    mask = np.zeros_like(gray, dtype=bool)
    for region in regionprops(sp_map):
        coords = region.coords
        gray_vals = gray[tuple(coords.T)]
        dark_fraction = np.mean(gray_vals < 0.9)
        if len(coords) > 20 and dark_fraction > 0.15:
            mask[tuple(coords.T)] = True

    # Optional: fill small gaps between tissue regions
    mask = binary_closing(mask, structure=np.ones((5, 5)))

    # Keep only the largest connected region
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if regions:
        largest = max(regions, key=lambda r: r.area)
        mask_clean = np.zeros_like(mask, dtype=bool)
        mask_clean[tuple(largest.coords.T)] = True
        mask = mask_clean

    # Crop based on mask
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        print("No valid superpixels found. Returning full image.")
        return arr
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_arr = arr[y0:y1, x0:x1, :]

    return cropped_arr, sp_map, mask, (y0, y1, x0, x1)


# Example usage
img = np.array(Image.open(r"C:\Users\cbran\Documents\ECU\Thesis\normal_082.png"))[:, :, :3]
cropped, sp_map, mask, (y0, y1, x0, x1) = crop_by_slic_and_color(img, n_segments=50, rgb_thresh=0.7)

# Show cropped image
plt.imshow(cropped)
plt.title("Cropped via SLIC + Color Threshold")
plt.axis("off")
plt.show()

# Show superpixel boundaries on original image
sp_vis = mark_boundaries(img, sp_map, color=(1, 1, 0))  # red boundaries
plt.imshow(sp_vis)
plt.title("Superpixel Map")
plt.axis("off")
plt.show()

cropped_sp_map = sp_map[y0:y1, x0:x1]
sp_cropped_vis = mark_boundaries(cropped, cropped_sp_map)
plt.imshow(sp_cropped_vis)
plt.title("Superpixels (Cropped Region)")
plt.axis("off")
plt.show()


# Get unique superpixel labels from the cropped region (excluding background 0)
unique_labels = np.unique(cropped_sp_map)
unique_labels = unique_labels[unique_labels != 0]

# Randomly pick 3 (or pick first 3)
import random
selected_labels = random.sample(list(unique_labels), k=min(3, len(unique_labels)))


# Plot each selected superpixel
for i, sp_id in enumerate(selected_labels, start=1):
    sp_mask = cropped_sp_map == sp_id
    sp_isolated = cropped.copy()
    sp_isolated[~sp_mask] = 0  # Zero out all non-superpixel pixels

    plt.imshow(sp_isolated)
    plt.title(f"Isolated Superpixel #{sp_id}")
    plt.axis("off")
    plt.show()


