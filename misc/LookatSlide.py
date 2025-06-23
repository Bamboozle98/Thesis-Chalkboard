import openslide
import matplotlib.pyplot as plt

file_path = r"C:\Users\cbran\Documents\ECU\Thesis\normal_082.tif"
slide = openslide.OpenSlide(file_path)

# Choose a low-resolution level (higher index = more downsampling)
level = 5
thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level])

# Remove alpha channel for display
thumbnail_rgb = thumbnail.convert("RGB")

plt.imshow(thumbnail_rgb)
plt.title(f"Level {level} view of slide")
plt.axis("off")
plt.show()
