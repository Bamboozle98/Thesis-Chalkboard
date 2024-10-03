import torch
from tqdm import tqdm  # Import tqdm for progress bar

def generate_feature_vectors(cnn_model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing on {device}")

    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    feature_vectors = []

    # Wrap the data_loader with tqdm to get progress updates
    with torch.no_grad():
        for superpixel_map, images, _ in tqdm(data_loader, desc="Generating feature vectors"):
            images = images.to(device)
            features = cnn_model(images)  # Get features from CNN
            feature_vectors.append(features)  # Append features to the list

    return feature_vectors
