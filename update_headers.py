# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 8/21/2025
# ----------------------------------------------------------------------->

from pathlib import Path
from ASL import ASL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

MASK_VISUALIZE = True
OUTPUT_DIR = Path('/home/connor/Thesis/updated_masks')

CLASS_MAP = {
    "NULL": 0,
    "CIR": 1,
    "CONE": 2,
    "CYL": 3,
    "PYR": 4,
    "SQ": 5,
    "MV": 6,
    "TABLE": 7,
    "CASE": 8,
    "TENT": 9,
}

# For visualization of new relabeled masks
COLOR_MAP = {
    0: (0, 0, 0),         # Black- background
    1: (255, 0, 0),       # Red- circular panels
    2: (0, 255, 0),       # Green- cones
    3: (0, 0, 255),       # Blue- cylinders
    4: (255, 255, 0),     # Yellow- pyramids
    5: (255, 0, 255),     # Magenta- sq panels
    6: (0, 255, 255),     # Cyan- vehicles
    7: (128, 128, 128),   # Gray- tables
    8: (255, 165, 0),     # Orange- cases
    9: (255, 255, 255),   # White- tents
}

def visualize_mask(mask):

    # Initialize an RGB image
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Map predictions to colors
    for class_value, color in COLOR_MAP.items():
        rgb_image[mask == class_value] = color

    # Create PIL image from RGB array
    Image.fromarray(rgb_image).save("mask_visualization.png")



if __name__ == "__main__":

    masks = Path('/home/connor/MATLAB/masks').glob('*.asl.hdr')

    for file in masks:

        hdr = ASL(file)

        relabeled_masks, valid_pixels = hdr.relabel_mask_by_class_name_map(class_map=CLASS_MAP)
        print(f"Relabeled masks: {relabeled_masks}")

        plt.figure(figsize=(10, 7))
        plt.imshow(relabeled_masks[:, :, 0], cmap="nipy_spectral")
        plt.axis(False)
        plt.show()
        

        # Visualize one relabeled mask
        if MASK_VISUALIZE:
            print("Visualizing first relabeled mask...")
            visualize_mask(relabeled_masks[:, :, 0])
            exit()
            MASK_VISUALIZE = False  # Only visualize the first mask

        # Save relabeled mask as .npy file
        np.savez(os.path.join(OUTPUT_DIR, f'{file}.npy'), relabeled_masks=relabeled_masks, valid_pixels=valid_pixels)




