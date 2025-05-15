import os
import matplotlib.pyplot as plt
import numpy as np
import openslide
from skimage.segmentation import mark_boundaries, slic
from skimage.measure    import regionprops
from torchvision import transforms

from Models.SuperPixel_Transformer.PNP_CNNs.customResnet import CustomResNet20
from Models.SuperPixel_Transformer.Model_Pipelines.Camelyon_16.ExperimentalC16 import extract_level1_patch

# ───────────────────────────────────────────────────────────────
# 1) LOAD SLIDE + THUMBNAIL
# ───────────────────────────────────────────────────────────────
wsi_path       = r"E:\Camelyon_16_Data\images\normal_001.tif"
thumbnail_size = (2048, 2048)
num_segments   = 50
patch_level    = 2

slide      = openslide.OpenSlide(wsi_path)
full_w, full_h = slide.level_dimensions[patch_level]
thumb_pil  = slide.get_thumbnail(thumbnail_size).convert("RGB")
slide.close()

thumb_np = np.array(thumb_pil)

# ───────────────────────────────────────────────────────────────
# 1b) BUILD & VISUALIZE AN RGB‐BASED TISSUE MASK
# ───────────────────────────────────────────────────────────────
# basic pink/purple filter: R high, B high, G low
R, G, B = thumb_np[...,0], thumb_np[...,1], thumb_np[...,2]
mask = (R > 160) & (B > 160) & (G < 140)

# show the mask overlay
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
ax1.imshow(thumb_pil); ax1.set_title("Original thumbnail"); ax1.axis('off')
ax2.imshow(mask, cmap='gray'); ax2.set_title("RGB‐threshold mask"); ax2.axis('off')
ax3.imshow(thumb_np); ax3.imshow(mask[...,None]*thumb_np, alpha=0.6)
ax3.set_title("Thumbnail with mask overlay"); ax3.axis('off')
plt.show()

# ───────────────────────────────────────────────────────────────
# 1c) CROP THUMBNAIL & SUPERPIXEL MAP TO MASK BBOX
# ───────────────────────────────────────────────────────────────
ys, xs         = np.where(mask)
y0_crop, y1_crop = ys.min(), ys.max()+1
x0_crop, x1_crop = xs.min(), xs.max()+1

thumb_crop_pil = thumb_pil.crop((x0_crop, y0_crop, x1_crop, y1_crop))
thumb_crop_np  = np.array(thumb_crop_pil)

# ───────────────────────────────────────────────────────────────
# 2) SUPERPIXEL SEGMENTATION ON THE CROPPED THUMBNAIL
# ───────────────────────────────────────────────────────────────
sp_crop = slic(
    thumb_crop_np,
    n_segments=num_segments,
    compactness=10,
    start_label=0
)

# ───────────────────────────────────────────────────────────────
# 3) VISUALIZE SUPERPIXEL BOUNDARIES
# ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mark_boundaries(thumb_crop_np, sp_crop))
ax.set_title("Superpixels on RGB‐cropped tissue thumbnail")
ax.axis('off')
plt.show()

# ───────────────────────────────────────────────────────────────
# 4) EXTRACT ALL HIGH-RES PATCHES AND SHOW THEM
# ───────────────────────────────────────────────────────────────
# open slide once
slide = openslide.OpenSlide(wsi_path)
# sizes at level-0 and at your desired patch_level
level0_w, level0_h = slide.level_dimensions[0]
levN_w,   levN_h   = slide.level_dimensions[patch_level]

# two scales
scale0_x = level0_w / thumb_pil.width
scale0_y = level0_h / thumb_pil.height
scaleN_x = levN_w   / thumb_pil.width
scaleN_y = levN_h   / thumb_pil.height

regions = regionprops(sp_crop.astype(int))
fig, axes = plt.subplots(10, 5, figsize=(25,25))

for r, ax in zip(regions, axes.flatten()):
    minr, minc, maxr, maxc = r.bbox

    # map back into full‐thumbnail coords
    orig_minr = minr + y0_crop
    orig_minc = minc + x0_crop
    orig_maxr = maxr + y0_crop
    orig_maxc = maxc + x0_crop

    # 1) compute level-0 origin
    x0_lvl0 = int(round(orig_minc * scale0_x))
    y0_lvl0 = int(round(orig_minr * scale0_y))

    # 2) compute level-N output size
    wN = int(round((orig_maxc - orig_minc) * scaleN_x))
    hN = int(round((orig_maxr - orig_minr) * scaleN_y))

    # 3) extract
    patch = slide.read_region(
        (x0_lvl0, y0_lvl0),
        patch_level,
        (wN, hN)
    ).convert("RGB")

    ax.imshow(patch)
    ax.axis('off')

plt.suptitle("All superpixel patches — now correctly mapped")
plt.show()
slide.close()

