import openslide
import os

file_dir = r"E:\Geradt\pathology"
dim_list = []


for file in os.listdir(file_dir):
    if file.lower().endswith(('.tif', '.tiff')):
        file_path = os.path.join(file_dir, file)
        slide = openslide.OpenSlide(file_path)
        dim_list.append(slide.level_dimensions[0])

print(max(dim_list))
print(min(dim_list))




