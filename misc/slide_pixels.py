import openslide
import os

file_path = r"C:\Users\cbran\Documents\ECU\Thesis\normal_082.tif"
dim_list = []

slide = openslide.OpenSlide(file_path)
# Print dimensions for all pyramid levels
for i, dims in enumerate(slide.level_dimensions):
    print(f"Level {i}: {dims[0]} x {dims[1]}")




