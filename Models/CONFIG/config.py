# Basic Hyperparameters used for all model variations.
num_epochs = 30
learning_rate = 1e-4
batch_size = 4
num_superpixels = 50
match_size = (512, 512) # The pixel dimensions that will be processed by the CNN.

# Whether to load the most recent checkpoint or train from scratch
use_checkpoint = False

# Select which dataset to currently train on.
# Options for dataset_option:
# 'oxford_pets', 'imagenet', 'wsi', 'high_res'
dataset_option = 'high_res'

# Select which ResNet CNN will generate the feature map.
# Options for CNN model:
# 'ResNet18', 'ResNet50', 'SuperpixelCNN', 'CustomResNet20'
cnn_option = 'ResNet18'


# DIRECTORY PATHS FOR ALL USED DATASETS
# Adding new datasets will require changes in the dataloaders and model pipelines.

# Directory for High Res Dataset
high_res_dir = r'E:\HighRes'

# Directory paths for OxfordPets
oxford_dataset_dir = r'/Data/OxfordPets/images\images'
imagenet_train_dir = r"E:\ImageNet\unpacked\train"
imagenet_val_dir = r"E:\ImageNet\unpacked\validate"


# Directory paths for Camelyon 16
wsi_dir = r"E:\Camelyon_16_Data\images"
ann_dir = r'E:\Camelyon_16_Data\annotations'

# Directoy paths for Dr. Gerardt's Dataset
g_dir = r"E:\Geradt\pathology"
g_ann = r"E:\Geradt\rs_filtered.tsv"

