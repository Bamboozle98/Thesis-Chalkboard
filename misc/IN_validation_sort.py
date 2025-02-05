import os
import shutil

val_images_dir = "E:/ImageNet/unpacked/validate"
devkit_dir = r"E:\ImageNet\ILSVRC2012_devkit_t12"  # Adjust this if needed
annotations_file = os.path.join(devkit_dir, r"E:\ImageNet\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt")

# Load ground-truth labels
with open(annotations_file, "r") as f:
    labels = f.readlines()

labels = [int(label.strip()) for label in labels]  # Convert to integers

# Create class-named subdirectories and move images
for i, label in enumerate(labels):
    label_dir = os.path.join(val_images_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    img_filename = f"ILSVRC2012_val_{i + 1:08d}.JPEG"
    src_path = os.path.join(val_images_dir, img_filename)
    dst_path = os.path.join(label_dir, img_filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

print("Validation images have been sorted into class folders.")
