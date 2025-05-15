import cv2
import numpy as np
import openslide
from PIL import Image
from skimage.filters import threshold_otsu


def crop_top_two_tissue_regions_with_buffer(image, buffer=20, min_area=1000):
    """
    Crops an image to include the top two largest tissue regions.
    It uses Otsu's thresholding to generate a binary mask, finds all external
    contours, selects the top two by area (ignoring any below min_area), and
    then calculates a bounding box that encompasses both regions. A buffer is
    added to the final bounding box.

    Args:
        image (np.array): Input image in BGR format.
        buffer (int): Number of pixels to pad around the tissue regions.
        min_area (int): Minimum area (in pixels) for a contour to be considered.

    Returns:
        np.array: The cropped image containing the top two tissue regions with padding.
    """
    # Convert image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate a binary mask using Otsu's thresholding with inversion.
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No tissue regions detected; returning the original image.")
        return image

    # Filter out small contours.
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # If we have fewer than two, use whatever is available.
    if not valid_contours:
        valid_contours = contours

    # Sort the valid contours in descending order of area.
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    # Select the top two contours (if available).
    top_contours = valid_contours[:2]

    # Combine the selected contours into one array of points.
    all_points = np.vstack(top_contours)

    # Compute the bounding rectangle that encloses the combined region.
    x, y, w, h = cv2.boundingRect(all_points)

    # Expand the bounding box by the given buffer.
    new_x = max(x - buffer, 0)
    new_y = max(y - buffer, 0)
    new_w = min(w + 2 * buffer, image.shape[1] - new_x)
    new_h = min(h + 2 * buffer, image.shape[0] - new_y)

    # Crop and return the image.
    cropped_image = image[new_y:new_y + new_h, new_x:new_x + new_w]
    return cropped_image


def rotate_image(image, angle):
    """
    Rotates the image around its center by the given angle (in degrees).

    Args:
        image (np.array): Input image.
        angle (float): Angle in degrees.

    Returns:
        np.array: The rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def on_trackbar(val):
    """
    Trackbar callback function. Converts the trackbar value (0-360)
    to an angle in [-180, 180] and rotates both the original and cropped images.
    """
    global base_image, cropped_image
    angle = val - 180  # Convert to angle in [-180, 180]
    print("Rotation angle:", angle)  # Debug print
    rotated_original = rotate_image(base_image, angle)
    rotated_cropped = rotate_image(cropped_image, angle)

    cv2.imshow("Original Rotated", rotated_original)
    cv2.imshow("Cropped & Rotated", rotated_cropped)


def main():
    global base_image, cropped_image

    # Specify the path to your WSI file.
    image_path = r"E:\Camelyon_16_Data\images\tumor_003.tif"

    # Open the whole-slide image using OpenSlide.
    try:
        slide = openslide.OpenSlide(image_path)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return

    # Define a thumbnail size to reduce the huge pixel count.
    thumbnail_size = (2048, 2048)
    thumbnail = slide.get_thumbnail(thumbnail_size)  # PIL Image (RGB)
    slide.close()

    # Convert the PIL image to a NumPy array; convert RGB to BGR.
    thumbnail_np = np.array(thumbnail)
    base_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2BGR)

    # Crop the tissue region.
    cropped_image = crop_top_two_tissue_regions_with_buffer(base_image, buffer=20)

    # Create two windows for the rotated images.
    cv2.namedWindow("Original Rotated")
    cv2.namedWindow("Cropped & Rotated")

    # Create a trackbar for interactive rotation.
    cv2.createTrackbar("Angle", "Original Rotated", 180, 360, on_trackbar)

    # Display the initial (0° rotated) images.
    cv2.imshow("Original Rotated", base_image)
    cv2.imshow("Cropped & Rotated", cropped_image)

    print("Use the trackbar to rotate both images. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



