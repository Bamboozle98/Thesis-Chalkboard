import os
import matplotlib.pyplot as plt
import numpy as np
import openslide
from skimage.segmentation import mark_boundaries, slic
from skimage.measure    import regionprops
from torchvision import transforms

# ───────────────────────────────────────────────────────────────
# Helper to extract patches
# ───────────────────────────────────────────────────────────────
def extract_levelN_patch(slide, bbox, level):
    x, y, w, h = bbox
    return slide.read_region((x, y), level, (w, h)).convert("RGB")

# ───────────────────────────────────────────────────────────────
# 1) CONFIG
# ───────────────────────────────────────────────────────────────
# wsi_path       = r"E:\Camelyon_16_Data\images\normal_001.tif"
wsi_path       = r"Downloads/normal_082.tif"
thumbnail_size = (2048, 2048)
num_segments   = 50
patch_level    = 2

# pull out just the filename (without ext) for labeling
base = os.path.splitext(os.path.basename(wsi_path))[0]

# ───────────────────────────────────────────────────────────────
# 2) LOAD SLIDE & THUMBNAIL
# ───────────────────────────────────────────────────────────────
slide      = openslide.OpenSlide(wsi_path)
full_w, full_h = slide.level_dimensions[patch_level]
thumb_pil  = slide.get_thumbnail(thumbnail_size).convert("RGB")
slide.close()
thumb_np = np.array(thumb_pil)

# ───────────────────────────────────────────────────────────────
# 3) BUILD & VISUALIZE RGB‐BASED TISSUE MASK
# ───────────────────────────────────────────────────────────────
R, G, B = thumb_np[...,0], thumb_np[...,1], thumb_np[...,2]
mask    = (R > 160) & (B > 160) & (G < 140)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
ax1.imshow(thumb_pil)
ax1.set_title(f"{base} — Original thumbnail")
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title(f"{base} — RGB mask")
ax2.axis('off')

ax3.imshow(thumb_np)
ax3.imshow(mask[...,None]*thumb_np, alpha=0.6)
ax3.set_title(f"{base} — Mask overlay")
ax3.axis('off')

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────
# 4) CROP & SLIC
# ───────────────────────────────────────────────────────────────
ys, xs       = np.where(mask)
y0_crop, y1_crop = ys.min(), ys.max()+1
x0_crop, x1_crop = xs.min(), xs.max()+1

thumb_crop_pil = thumb_pil.crop((x0_crop, y0_crop, x1_crop, y1_crop))
thumb_crop_np  = np.array(thumb_crop_pil)

sp_crop = slic(
    thumb_crop_np,
    n_segments=num_segments,
    compactness=10,
    start_label=0
)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mark_boundaries(thumb_crop_np, sp_crop))
ax.set_title(f"{base} — Superpixels")
ax.axis('off')
plt.show()

# ───────────────────────────────────────────────────────────────
# 5) EXTRACT & DISPLAY PATCHES WITH LABELS
# ───────────────────────────────────────────────────────────────
slide = openslide.OpenSlide(wsi_path)
level0_w, level0_h = slide.level_dimensions[0]
levN_w,   levN_h   = slide.level_dimensions[patch_level]

scale0_x = level0_w / thumb_pil.width
scale0_y = level0_h / thumb_pil.height
scaleN_x = levN_w   / thumb_pil.width
scaleN_y = levN_h   / thumb_pil.height

regions = regionprops(sp_crop.astype(int))
fig, axes = plt.subplots(10, 5, figsize=(25,25))

for idx, (r, ax) in enumerate(zip(regions, axes.flatten())):
    minr, minc, maxr, maxc = r.bbox

    # map back to full-thumbnail coords
    orig_minr = minr + y0_crop
    orig_minc = minc + x0_crop
    orig_maxr = maxr + y0_crop
    orig_maxc = maxc + x0_crop

    # compute level-0 origin & level-N size
    x0_lvl0 = int(round(orig_minc * scale0_x))
    y0_lvl0 = int(round(orig_minr * scale0_y))
    wN       = int(round((orig_maxc - orig_minc) * scaleN_x))
    hN       = int(round((orig_maxr - orig_minr) * scaleN_y))

    # extract patch
    patch = extract_levelN_patch(
        slide,
        (x0_lvl0, y0_lvl0, wN, hN),
        level=patch_level
    )

    ax.imshow(patch)
    ax.set_title(f"{base}_SP{idx}", fontsize=8)
    ax.axis('off')

plt.suptitle(f"All superpixel patches — {base}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

slide.close()