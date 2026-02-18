import cv2
import os
import numpy as np
from tqdm import tqdm

CLASS_MAP = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}

def convert_mask_to_yolo(mask_path, output_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return

    h, w = mask.shape[:2]
    yolo_lines = []

    for pixel_val, class_id in CLASS_MAP.items():
        binary_mask = np.uint8(mask == pixel_val) * 255
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) < 3: 
                continue
            
            poly = cnt.reshape(-1, 2).astype(np.float32)
            poly[:, 0] /= w  
            poly[:, 1] /= h  
            
          
            line = f"{class_id} " + " ".join([f"{coord[0]:.6f} {coord[1]:.6f}" for coord in poly])
            yolo_lines.append(line)

    # Write the text file
    if yolo_lines:
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))

project_path = "C:/Users/LENOVO/Desktop/Final_hackathon"

for split in ['train', 'val']:
    mask_dir = os.path.join(project_path, split, "Segmentation")
    label_dir = os.path.join(project_path, split, "labels")
    
    print(f"Converting {split} set...")
    if not os.path.exists(mask_dir):
        print(f"❌ Error: Could not find {mask_dir}")
        continue

    os.makedirs(label_dir, exist_ok=True)
    
    # Process files
    files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    for filename in tqdm(files):
        convert_mask_to_yolo(
            os.path.join(mask_dir, filename),
            os.path.join(label_dir, filename.replace('.png', '.txt'))

        )
