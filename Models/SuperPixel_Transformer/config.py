num_epochs = 30
learning_rate = 1e-4
batch_size = 8
num_superpixels = 50
match_size = (512, 512)

# Whether to load the most recent checkpoint or train from scratch
use_checkpoint = False

# Options for dataset_option:
# 'oxford_pets', 'imagenet', 'wsi', or 'single_image', 'high_res'
dataset_option = 'high_res'

# Options for CNN model:
# 'ResNet18', 'ResNet50', 'SuperpixelCNN', 'CustomResNet20'
cnn_option = 'ResNet18'


# Directory for High Res Dataset
high_res_dir = r'U:\HighRes'

# Directory paths for various dataset types:
oxford_dataset_dir = r'U:\Oxford\images\images'
imagenet_train_dir = r"C:\Users\mccutcheonc18\PycharmProjects\Thesis-Chalkboard\ImageNetDataset\train"
imagenet_val_dir = r"C:\Users\mccutcheonc18\PycharmProjects\Thesis-Chalkboard\ImageNetDataset\validate"


# For Single Image classification (all images stored as .tif's in one folder)
wsi_dir = r"E:\Camelyon_16_Data\images"
ann_dir = r'E:\Camelyon_16_Data\annotations'

# Dr. Gerardt Dataset
g_dir = r"E:\Geradt\pathology"
g_ann = r"E:\Geradt\rs_filtered.tsv"

