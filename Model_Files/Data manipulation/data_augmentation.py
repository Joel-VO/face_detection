import os
from PIL import Image
from torchvision import transforms

# the random augmentations applied to increase the number of data
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Resize to 224x224
])

# Number of augmented images created
num_augmentations = 100

# Input and output directories
input_dir = "path to input dir"
output_dir = "path to output dir"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB mode

        # applied to the same image num_augmentations times
        for i in range(num_augmentations):
            augmented_img = augmentation(img)

            # Save augmented image
            augmented_filename = f"aug_{i}_{filename}"
            save_path = os.path.join(output_dir, augmented_filename)
            augmented_img.save(save_path)

        print(f"Saved {num_augmentations} augmented versions of {filename}")