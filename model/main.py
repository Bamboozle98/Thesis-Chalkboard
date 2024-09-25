from ViT import transformer_model
from Data_Loader_SP import data_process_SP
from Training import training
from Eval import evaluation
from MiniCNN import SuperpixelCNN
import torch

# This is where we will define Hyperparameters.
num_epochs = 5
lr = 1e-5

# Example size of the superpixel feature vectors (adjust according to your implementation)
num_superpixel_features = 512  # Assuming your SuperpixelCNN output is of size 512


# This function references all others in the repo required to run the model.
def main():
    # Acquire the data from custom Data Loader.
    train_loader, val_loader, class_names = data_process_SP()

    # Initialize the CNN model
    cnn_model = SuperpixelCNN()

    # Adjust the transformer for classifying on Oxford.
    vit_model = transformer_model(class_names, num_superpixel_features)

    # Train the Vision Transformer with features assigned from the CNN
    vit_model = training(vit_model, cnn_model, train_loader, num_epochs, lr)

    # Evaluate the model.
    evaluation(vit_model, val_loader, class_names)


if __name__ == '__main__':
    main()

