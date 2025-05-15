import os
import tarfile

# Path to the folder containing multiple tar files
tar_folder = r"C:\Users\mccutcheonc18\PycharmProjects\Thesis-Chalkboard\ImageNet\train"

# Destination folder for extracted contents
extract_folder = r"C:\Users\mccutcheonc18\PycharmProjects\Thesis-Chalkboard\ImageNet\train"

# Ensure the extract folder exists
os.makedirs(extract_folder, exist_ok=True)

# Iterate through each file in the folder
for file in os.listdir(tar_folder):
    if file.endswith(".tar"):  # Check if the file is a tar archive
        tar_path = os.path.join(tar_folder, file)
        extract_path = os.path.join(extract_folder, file[:-4])  # Extract to a subfolder

        # Extract the tar file
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(extract_path)

        # Delete the tar file after extraction
        os.remove(tar_path)
        print(f"Extracted and deleted: {file}")

print("All tar files have been extracted and deleted.")


