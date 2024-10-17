from tqdm import tqdm  # Import tqdm for progress bar
import torch
import numpy as np


def assign_features_to_superpixels(cnn_features, superpixel_maps, images):
    """
    Map CNN features to superpixel regions and store both the superpixel vectors and the corresponding image.

    Args:
        cnn_features (torch.Tensor): Feature vectors from the CNN of shape (batch_size, num_features).
        superpixel_maps (torch.Tensor): Superpixel maps of shape (batch_size, height, width).
        images (torch.Tensor): Original images of shape (batch_size, channels, height, width).

    Returns:
        combined_data (list of tuples): Each tuple contains superpixel vectors and the corresponding image.
    """
    combined_data = []

    for i in range(min(len(cnn_features), len(superpixel_maps), len(images))):
        img_features = cnn_features[i].view(512, -1).permute(1, 0).cpu().numpy()
        img_superpixel_map = superpixel_maps[i].cpu().numpy()

        superpixels = np.unique(img_superpixel_map)

        image_superpixel_vectors = []
        for sp in superpixels:
            mask = img_superpixel_map == sp
            pixel_indices = np.flatnonzero(mask)
            if pixel_indices.size > 0:
                sp_features = img_features[pixel_indices]
                if sp_features.size > 0:
                    sp_average = sp_features.mean(axis=0)
                    image_superpixel_vectors.append(sp_average)

        superpixel_tensor = torch.tensor(np.array(image_superpixel_vectors))

        # Combine the superpixel vectors with the image
        combined_data.append((superpixel_tensor, images[i]))

    return combined_data


def generate_feature_vectors(cnn_model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    combined_data_list = []

    with torch.no_grad():
        for superpixel_map, images, _ in tqdm(data_loader, desc="Generating superpixel vectors"):
            images = images.to(device)
            features = cnn_model(images)
            combined_data = assign_features_to_superpixels(features, superpixel_map, images)
            combined_data_list.extend(combined_data)  # Append each image+superpixel pair

    return combined_data_list




# Create the dataset and DataLoader
combined_data_list = generate_feature_vectors(cnn_model, data_loader)
dataset = SuperpixelDataset(combined_data_list)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)



