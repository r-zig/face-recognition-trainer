import sys
from PIL import Image
import torchvision.transforms as transforms
import os

def augment_image(image_path, output_base_dir, num_augmentations):
    # load the image
    image = Image.open(image_path)
    
    # get the image name without the extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Define transformations for day and night conditions
    day_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.9, hue=0.1)
    ])

    night_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=image.size, scale=(0.5, 1.2)),  # Wider range for distance
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Darker for night
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.5, 2))       # Optional blur for night effect
    ])
    
    
    # Set base_dir dynamically for the first file
    base_dir = os.path.basename(os.path.dirname(image_path))

    # Calculate the relative path from the first encountered directory
    output_dir = os.path.join(output_base_dir, base_dir)
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    # apply the transformations and save the augmented images
    for i in range(num_augmentations):
        print(os.getcwd())
        day_image = day_transform(image)
        night_image = night_transform(image)
        output_night_path = os.path.join(output_dir, f"{image_name}_day_{i+1}.jpg")
        try:
            day_image.save(output_night_path)
            print(f"Saved augmented image {i+1} at {output_night_path}")

            output_day_path = os.path.join(output_dir, f"{image_name}_night_{i+1}.jpg")
            night_image.save(output_day_path)
            print(f"Saved augmented image {i+1} at {output_day_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
            return


if __name__ == "__main__":
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    num_augmentations = int(sys.argv[3])
    augment_image(image_path, output_dir, num_augmentations)