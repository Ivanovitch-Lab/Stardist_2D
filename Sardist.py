import os
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Path to the input folder containing TIFF files
input_folder = r"C:\Users\ucklkdi\OneDrive - University College London\Desktop\Images to send to Kenzo\Stardist"
# Path to the output folder
output_folder = r"C:\Users\ucklkdi\OneDrive - University College London\Desktop\Images to send to Kenzo\Stardist"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each file in the input folder
for file_name in os.listdir(input_folder):
    # Only process files with "C1" in their name and with a ".tif" extension
    if "C1" in file_name and file_name.lower().endswith(".tif"):
        print(f"Processing file: {file_name}")
        
        # Full path to the input file
        input_path = os.path.join(input_folder, file_name)
        
        # Full path to the output files
        output_path_tif = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_segmentation.tif")
        output_path_npy = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_segmentation.npy")
        
        # Load the image
        img = imread(input_path)
        
        # Normalize the image (StarDist models expect normalized input)
        img_normalized = normalize(img, 1, 99.8, axis=(0, 1))
        
        # Predict instance segmentation
        labels, _ = model.predict_instances(img_normalized, n_tiles=(4, 4))
        
        # Save the segmentation result as a .tif file
        imwrite(output_path_tif, labels.astype(np.uint16))  # Save as 16-bit integer TIFF
        
        # Save the segmentation result as a NumPy array
        np.save(output_path_npy, labels)
        
        # Plot the input image and the prediction
        # Create and save the overlay image instead of displaying it
        overlay = render_label(labels, img=img)
        output_path_png = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_overlay.png")
        plt.imsave(output_path_png, overlay)

        print(f"Segmentation saved as TIFF: {output_path_tif}")
        print(f"Segmentation saved as NumPy array: {output_path_npy}")
    else:
        print(f"Skipping file: {file_name} (does not match criteria)")