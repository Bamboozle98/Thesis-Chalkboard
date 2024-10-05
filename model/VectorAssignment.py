import torch
import numpy as np


def assign_features_to_superpixels(cnn_features, superpixel_maps):
    """
    This function assigns CNN feature vectors to superpixels and averages them for each superpixel.

    Args:
    - cnn_features: A list of CNN feature maps, one per image. Each feature map has shape (512, H, W).
    - superpixel_maps: A list of superpixel maps corresponding to each image. Each superpixel map has shape (H, W).

    Returns:
    - superpixel_vectors: A list of averaged feature vectors, one for each superpixel, per image.
    """

    all_superpixel_vectors = []

    # Loop through each image in the batch
    for i in range(len(cnn_features)):
        img_features = cnn_features[i].view(512, -1).permute(1, 0).cpu().numpy()  # (H*W, 512)
        img_superpixel_map = superpixel_maps[i].cpu().numpy()  # (H, W)

        # Determine the number of superpixels
        num_superpixels = np.max(img_superpixel_map) + 1
        superpixel_vectors = np.zeros((num_superpixels, 512))  # To store averaged vectors for each superpixel
        superpixel_counts = np.zeros(num_superpixels)  # To count the number of pixels in each superpixel

        # Sort feature vectors into their corresponding superpixels
        for pixel_idx, sp_idx in enumerate(img_superpixel_map.flatten()):
            superpixel_vectors[sp_idx] += img_features[pixel_idx]
            superpixel_counts[sp_idx] += 1

        # Average the feature vectors for each superpixel
        superpixel_vectors /= superpixel_counts[:, None]

        # Append the result for this image
        all_superpixel_vectors.append(torch.tensor(superpixel_vectors))

    return all_superpixel_vectors  # Return a list of superpixel vectors for each image
