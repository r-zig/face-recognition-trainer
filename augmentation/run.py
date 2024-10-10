# Example usage of the augment_image function
from glob import glob  # Correct import for the glob function
import os
from image_augmentation import augment_image


input_base_dir = "./input/"
output_dir = "./output/"

num_augmentations = 10 
print(os.getcwd())
def process_directory(directory):
    # Recursively find all image files in the directory
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    for ext in image_extensions:
        # Use glob to find files with current extension recursively
        for image_file in glob(os.path.join(directory, '**', ext), recursive=True):
            print(f"Processing {image_file}")
            augment_image(image_file, output_dir, num_augmentations)


process_directory(input_base_dir)