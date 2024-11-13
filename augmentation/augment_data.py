import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def apply_handheld_effects(image):
    """Apply realistic handheld camera effects to an image."""
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Random slight rotation (-5 to 5 degrees)
    angle = np.random.uniform(-5, 5)
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Random perspective transform (subtle)
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Add small random offsets to corners
    offset = width * 0.02  # 2% of image width
    pts2 = np.float32([
        [np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
        [width + np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
        [np.random.uniform(-offset, offset), height + np.random.uniform(-offset, offset)],
        [width + np.random.uniform(-offset, offset), height + np.random.uniform(-offset, offset)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, matrix, (width, height))
    
    # Motion blur
    # Random kernel size between 3 and 7
    kernel_size = np.random.choice([3, 5, 7])
    # Random angle for motion blur
    kernel_angle = np.random.uniform(0, 360)
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    angle_rad = np.deg2rad(kernel_angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = abs((i-center)*dy - (j-center)*dx)
            if dist <= 0.5:
                kernel[i,j] = 1
    kernel = kernel / kernel.sum()
    
    # Apply motion blur
    image = cv2.filter2D(image, -1, kernel)
    
    # Subtle brightness/contrast variation
    alpha = np.random.uniform(0.95, 1.05)  # Contrast
    beta = np.random.uniform(-5, 5)        # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

def process_image_folder(input_folder, output_folder):
    """Process all images in the input folder and save augmented versions."""
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = list(Path(input_folder).glob('*.jpg')) + \
                 list(Path(input_folder).glob('*.jpeg')) + \
                 list(Path(input_folder).glob('*.png'))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Apply effects
        augmented = apply_handheld_effects(image)
        
        # Save augmented image
        output_file = output_path / f"{img_path.name}"
        cv2.imwrite(str(output_file), augmented)

if __name__ == "__main__":
    input_folder = "../data_v2/images"
    output_folder = "../data_v2/images_handheld"
    
    print("Beginning Processing")
    process_image_folder(input_folder, output_folder)
    print("Processing complete!")