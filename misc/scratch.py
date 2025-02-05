from torchvision import datasets

val_dir = "E:/ImageNet/unpacked/validate"
val_dataset = datasets.ImageFolder(val_dir)

print("Class-to-Index Mapping:")
print(val_dataset.class_to_idx)
