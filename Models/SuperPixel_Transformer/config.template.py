num_epochs = 30
learning_rate = 1e-4

# dataset_option refers to what dataset is used in the model currently. Options are 'oxford_pets' or 'imagenet'
dataset_option = 'imagenet'

# PNP_CNN refers to what CNN model is used to feed Superpixel features to the Transformers.
# Options are 'ResNet18' and 'ResNet50' and 'SuperpixelCNN'
cnn_option = 'ResNet18'

# Directory for the unorganized OxfordPets Dataset. Dataloader will extract all necessary information.
oxford_dataset_dir = r'C:\Users\cbran\PycharmProjects\Thesis-Chalkboard\Data\images\images'

# Directory for organized ImageNet 2012 Classification Dataset Challenge. To organize you will need the ground truths
# file as well as the sysnset class naming file 'map_clsloc'
imagenet_train_dir = r"E:\ImageNet\unpacked\train"
imagenet_val_dir = r"E:\ImageNet\unpacked\validate"

batch_size = 16
