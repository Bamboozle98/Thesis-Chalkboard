from openslide import OpenSlide
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import remove_small_objects

def grabcut_crop(thumb: Image.Image,
                 buffer: int = 10,
                 min_area: int = 2000,
                 iter_count: int = 5) -> Image.Image:
    """
    Use GrabCut to separate foreground (tissue) from background (white + black border).
    Returns a tight crop around all remaining foreground pixels.
    """
    # 1) convert to BGR numpy
    bgr = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # 2) init mask & models
    mask     = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # 3) define a rectangle almost as big as the image (minus a small border)
    rect = (buffer, buffer, w - 2*buffer, h - 2*buffer)

    # 4) run GrabCut
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel,
                iterCount=iter_count,
                mode=cv2.GC_INIT_WITH_RECT)

    # 5) build a clean Boolean foreground mask
    fg_mask = np.logical_or(mask==cv2.GC_PR_FGD, mask==cv2.GC_FGD)
    fg_mask = remove_small_objects(fg_mask, min_size=min_area)

    # 6) bounding box of all fg pixels
    ys, xs = np.where(fg_mask)
    if not len(ys) or not len(xs):
        return thumb  # fallback
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1

    # 7) crop and return
    return thumb.crop((x0, y0, x1, y1))


def crop_wsi_thumb_only(path,
                        thumbnail_size=(2048,2048),
                        mask_level=4,
                        method="grabcut",   # <-- new option
                        buffer=10,
                        min_area=2000,
                        **kwargs):
    # load thumbnail
    with OpenSlide(path) as slide:
        w, h   = slide.level_dimensions[mask_level]
        thumb  = slide.read_region((0,0), mask_level, (w,h)).convert("RGB")

    if method == "grabcut":
        cropped = grabcut_crop(
            thumb,
            buffer=buffer,
            min_area=min_area,
            iter_count=kwargs.get("iter_count", 5)
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return cropped, thumb


if __name__ == "__main__":
    import cv2

    path = r"E:\Camelyon_16_Data\images\normal_002.tif"
    cropped, uncropped = crop_wsi_thumb_only(
        path,
        method="grabcut",
        buffer=20,
        min_area=2000,
        iter_count=7
    )

    # display resized
    unc_np  = cv2.cvtColor(np.array(uncropped), cv2.COLOR_RGB2BGR)
    crp_np  = cv2.cvtColor(np.array(cropped),   cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Uncropped", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cropped",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Uncropped", 800, 600)
    cv2.resizeWindow("Cropped",   800, 600)
    cv2.imshow("Uncropped", unc_np)
    cv2.imshow("Cropped",   crp_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
