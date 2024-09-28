import cv2
import albumentations as A
import os
from glob import glob
from matplotlib import pyplot as plt

# Function to visualize the augmented image (optional)
def visualize(image, title=""):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Augmentation pipeline
def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        ], p=0.9),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.5),
        A.CLAHE(clip_limit=2, p=0.2),
        A.RandomResizedCrop(height=416, width=416, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT)
    ])

# Function to augment images and save them
def augment_and_save_image(image_path, output_folder):
    image = cv2.imread(image_path)
    file_name = os.path.basename(image_path)
    
    # Apply augmentations
    augmentations = get_augmentations()
    augmented = augmentations(image=image)
    augmented_image = augmented['image']
    
    # Save the augmented image
    output_image_path = os.path.join(output_folder, f"aug_{file_name}")
    cv2.imwrite(output_image_path, augmented_image)
    
    visualize(augmented_image, title=f"Augmented: {file_name}")  # Optionally visualize

# Function to augment all images in the input folder
def augment_images_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    image_paths = glob(os.path.join(input_folder, "*.jpg"))  # Adjust if your images are in a different format (e.g., PNG)
    
    print(f"Found {len(image_paths)} images to augment.")
    
    for image_path in image_paths:
        augment_and_save_image(image_path, output_folder)
        print(f"Augmented and saved {os.path.basename(image_path)}")

# Main function
if __name__ == "__main__":
    # Input folder containing images captured from the camera
    input_folder = 'input'  # Update with your actual folder path
    output_folder = 'output'  # Folder where augmented images will be saved

    # Augment and save images
    augment_images_from_folder(input_folder, output_folder)
