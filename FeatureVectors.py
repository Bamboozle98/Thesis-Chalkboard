# superpixel_assignment.py
import torch

def assign_features_to_superpixels(cnn_features, superpixel_map):
    """
    Assigns CNN features to corresponding superpixel regions.

    Parameters:
    - cnn_features (torch.Tensor): CNN output features of shape (batch_size, feature_dim, H, W).
    - superpixel_map (torch.Tensor): Superpixel map of shape (batch_size, H, W), where each pixel is assigned a superpixel label.

    Returns:
    - torch.Tensor: A tensor of superpixel vectors for each image in the batch.
    """
    batch_size, feature_dim = cnn_features.size(0), cnn_features.size(1)
    superpixel_vectors = []

    for i in range(batch_size):
        superpixel_map_batch = superpixel_map[i]
        unique_superpixels = torch.unique(superpixel_map_batch)  # Get unique superpixel labels
        superpixel_vector = torch.zeros((len(unique_superpixels), feature_dim)).to(cnn_features.device)

        for j, sp in enumerate(unique_superpixels):
            mask = (superpixel_map_batch == sp)  # Create mask for the current superpixel
            # Average the CNN features corresponding to the current superpixel
            masked_features = cnn_features[i] * mask.unsqueeze(0)  # Broadcasting mask to match CNN feature dimensions
            superpixel_vector[j] = masked_features.mean(dim=[1, 2])  # Average along the spatial dimensions (H, W)

        superpixel_vectors.append(superpixel_vector)

    return torch.stack(superpixel_vectors)  # Stack into batch format
