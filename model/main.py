from ViT import transformer_model
from Data_Loader import data_process
from Training import training
from Eval import evaluation

# This is where we will define Hyperparameters.
num_epochs = 5
lr = 1e-5

# This function references all others in the repo required to run the model. If we keep things as modular as possible,
# it will be easier to make changes for different iterations.
def main():
    # Acquire the data.
    train_loader, val_loader, class_names = data_process()
    # Adjust the transformer for classifying on Oxford.
    model = transformer_model(class_names)
    # A little fine-tuning gave great results on the pet classification (97% Accuracy).
    model = training(model, train_loader, num_epochs, lr)
    # Evaluate the model.
    evaluation(model, val_loader, class_names)


if __name__ == '__main__':
    main()
