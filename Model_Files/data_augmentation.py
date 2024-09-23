import os
from PIL import Image
import torch
from torchvision import transforms

# Define augmentations
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Resize to 224x224
])

# Number of augmented versions you want to create per image
num_augmentations = 100

# Input and output directories
input_dir = "/home/joel/coding/Datasets/Raw data/Malavika S"
output_dir = "/home/joel/coding/Datasets/newest raw augmented/Malavika S"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB mode

        # Apply augmentations multiple times to the same image
        for i in range(num_augmentations):
            augmented_img = augmentation(img)

            # Save augmented image
            augmented_filename = f"aug_{i}_{filename}"
            save_path = os.path.join(output_dir, augmented_filename)
            augmented_img.save(save_path)

        print(f"Saved {num_augmentations} augmented versions of {filename}")
