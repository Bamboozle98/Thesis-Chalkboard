import cv2
import numpy as np
import openslide
from PIL import Image


def crop_all_tissue_regions_with_buffer_color(image, buffer=20, min_area=1000, s_thresh=70, v_thresh=200,
                                              clip_border=20, black_threshold=10):
    """
    Crops an image to include all tissue-colored regions using HSV color segmentation,
    with optional border clipping and exclusion of very dark (black) artifacts.

    The function first clips a fixed border from the image (if clip_border > 0),
    then converts the image to HSV to create a tissue mask based on saturation and value thresholds.
    Next, a morphological closing is applied, and external contours are extracted.
    Contours smaller than min_area or whose mean brightness (V) is below black_threshold are discarded.
    All remaining contours are combined into one bounding box, which is then padded by a buffer before cropping.

    Args:
        image (np.array): Input image in BGR format.
        buffer (int): Number of pixels to pad around the detected tissue regions.
        min_area (int): Minimum area (in pixels) for a contour to be considered.
        s_thresh (int): Lower threshold for saturation (tissue usually has higher saturation).
        v_thresh (int): Upper threshold for brightness; pixels with V below this are likely tissue.
        clip_border (int): Number of pixels to clip from each side of the image.
        black_threshold (int): Minimum average brightness (V in HSV) to consider a region valid.

    Returns:
        np.array: The cropped image covering all valid tissue regions with padding.
    """
    # Optionally clip the border.
    if clip_border > 0:
        image = image[clip_border:-clip_border, clip_border:-clip_border]

    # Convert from BGR to HSV.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to select tissue: high saturation, but not too bright.
    lower_bound = (0, s_thresh, 0)
    upper_bound = (180, 255, v_thresh)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Refine the mask with morphological closing.
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find external contours.
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No tissue regions detected; returning the original image.")
        return image

    # Filter contours:
    # Exclude contours smaller than min_area or those with a mean brightness below black_threshold.
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Create a mask for this contour.
        cnt_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        # Compute mean V value (brightness) in this contour region.
        mean_v = cv2.mean(hsv, mask=cnt_mask)[2]  # index 2 is the V channel.
        if mean_v < black_threshold:
            continue  # Likely a black artifact.
        valid_contours.append(cnt)

    if not valid_contours:
        # If none survive, use all contours.
        valid_contours = contours

    # Combine all valid contours.
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Expand the bounding box with a buffer.
    new_x = max(x - buffer, 0)
    new_y = max(y - buffer, 0)
    new_w = min(w + 2 * buffer, image.shape[1] - new_x)
    new_h = min(h + 2 * buffer, image.shape[0] - new_y)

    # Crop and return.
    cropped_image = image[new_y:new_y + new_h, new_x:new_x + new_w]
    return cropped_image


def rotate_image(image, angle):
    """
    Rotates the image around its center by the given angle (in degrees).
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def on_trackbar(val):
    """
    Trackbar callback: converts the trackbar value (0-360) to an angle (-180 to 180)
    and rotates both the original and cropped images.
    """
    global base_image, cropped_image
    angle = val - 180
    print("Rotation angle:", angle)
    rotated_original = rotate_image(base_image, angle)
    rotated_cropped = rotate_image(cropped_image, angle)
    cv2.imshow("Original Rotated", rotated_original)
    cv2.imshow("Cropped & Rotated", rotated_cropped)


def main():
    global base_image, cropped_image

    # Specify the path to your WSI file.
    image_path = r"E:\Camelyon_16_Data\images\tumor_003.tif"

    # Open the WSI using OpenSlide.
    try:
        slide = openslide.OpenSlide(image_path)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return

    # Define a thumbnail size.
    thumbnail_size = (2048, 2048)
    thumbnail = slide.get_thumbnail(thumbnail_size)  # PIL Image (RGB)
    slide.close()

    # Convert PIL image to NumPy array and from RGB to BGR.
    thumbnail_np = np.array(thumbnail)
    base_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2BGR)

    # Crop using the updated color-based function with border clipping and black artifact filtering.
    cropped_image = crop_all_tissue_regions_with_buffer_color(base_image, buffer=20, min_area=1000, s_thresh=70,
                                                              v_thresh=200, clip_border=20, black_threshold=10)

    # Create windows and a trackbar for rotation.
    cv2.namedWindow("Original Rotated")
    cv2.namedWindow("Cropped & Rotated")
    cv2.createTrackbar("Angle", "Original Rotated", 180, 360, on_trackbar)

    cv2.imshow("Original Rotated", base_image)
    cv2.imshow("Cropped & Rotated", cropped_image)
    print("Use the trackbar to rotate both images. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Alias for backwards compatibility.
crop_all_tissue_regions_with_buffer_color = crop_all_tissue_regions_with_buffer_color

if __name__ == "__main__":
    main()
