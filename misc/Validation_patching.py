import os
import shutil

# Path to your validation set
val_dir = r"D:\Thesis\unpacked\validate"

# Load the label-to-synset mapping
mapping_file = "D:\Thesis\ILSVRC2012_devkit_t12\ILSVRC2012_devkit_t12\data\map_clsloc.txt"  # Adjust path if needed
with open(mapping_file, "r") as f:
    lines = f.readlines()

# Create a dictionary mapping label index to WordNet synset ID
label_to_synset = {}
for line in lines:
    parts = line.strip().split()
    synset_id = parts[0]  # Example: "n01440764"
    label_index = int(parts[1])  # Example: 1-1000
    label_to_synset[label_index] = synset_id

# Rename the folders
for num_label in os.listdir(val_dir):
    old_path = os.path.join(val_dir, num_label)

    if num_label.isdigit():  # Ensure it's a number
        num_label = int(num_label)  # Convert to int (1-1000)

        if num_label in label_to_synset:
            new_folder_name = label_to_synset[num_label]
            new_path = os.path.join(val_dir, new_folder_name)

            # Rename if it doesn't already exist
            if not os.path.exists(new_path):
                shutil.move(old_path, new_path)
                print(f"Renamed {old_path} → {new_path}")

print("Validation set folders renamed successfully.")
