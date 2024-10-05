from ViT import TransformerEncoder
from Data_Loader_SP import data_process_SP
from Training import training
from Eval import evaluation
from MiniCNN import SuperpixelCNN
from VectorCreation import generate_feature_vectors
from model.VectorAssignment import assign_features_to_superpixels
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

    # Extract superpixel maps from the data loader
    superpixel_maps = [batch[0] for batch in train_loader]  # Assuming superpixel_map is the first element in each batch

    # Assign features to superpixels
    mapped_vectors = assign_features_to_superpixels(cnn_output, superpixel_maps)

    # Prepare the transformer input (mapped_vectors)
    transformer_input = [torch.stack(batch) for batch in mapped_vectors]  # Stack vectors for each image in the batch

    # Initialize the transformer model
    vit_model = TransformerEncoder(class_names, num_superpixel_features)

    # Train the Vision Transformer with features assigned from the CNN
    vit_model = training(vit_model, transformer_input, train_loader, num_epochs, lr)  # Adjust training function as necessary

    # Evaluate the model
    evaluation(vit_model, val_loader, class_names)

if __name__ == '__main__':
    main()

