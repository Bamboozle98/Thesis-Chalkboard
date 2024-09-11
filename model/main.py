from ViT import transformer_model
from Data_Loader import data_process
from Training import training
from Eval import evaluation

num_epochs = 5
lr = 1e-5


def main():
    train_loader, val_loader, class_names = data_process()
    model = transformer_model(class_names)
    model = training(model, train_loader, num_epochs, lr)
    evaluation(model, val_loader, class_names)


if __name__ == '__main__':
    main()
