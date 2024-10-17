from Transformer import TransformerEncoder
from Data_Loader_SP import data_process_SP
from Training import training
from Vector_Data_Loader import SuperpixelDataset
from Eval import evaluation
from MiniCNN import SuperpixelCNN
from VectorCreation import generate_feature_vectors
from model.VectorCreation import assign_features_to_superpixels
import torch

# Hyperparameters
num_epochs = 5
lr = 1e-5

# Superpixel feature vector size
num_superpixel_features = 512  # Adjust according to your CNN output


def main():
    # Acquire the data from custom Data Loader
    train_loader, val_loader, class_names = data_process_SP()

    # Initialize the CNN model
    cnn_model = SuperpixelCNN()

    # Generate feature vectors from CNN
    cnn_output = generate_feature_vectors(cnn_model, train_loader)

    # Assume superpixel maps are also being loaded in the train_loader
    mapped_vectors = []

    for (superpixel_maps, images, _) in train_loader:
        # Assign features to superpixels
        mapped_vectors_batch = assign_features_to_superpixels(cnn_output, superpixel_maps)
        mapped_vectors.append(mapped_vectors_batch)

    dataset = SuperpixelDataset(mapped_vectors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    train_loader = dataloader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Concatenate all mapped vectors into a single tensor
    print(mapped_vectors)
    mapped_vectors = torch.cat(mapped_vectors, dim=0)


    # Start training with mapped superpixel vectors
    trained_model = training(model, mapped_vectors, num_epochs, lr)


if __name__ == '__main__':
    main()

