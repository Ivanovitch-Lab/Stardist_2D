import os
import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from stardist.plot import render_label
from scipy.stats import zscore

# Input and output folders
input_folder = r"C:\Users\ucklkdi\OneDrive - University College London\Desktop\transfer 10_6_25\medial_stardist"
output_folder = r"C:\Users\ucklkdi\OneDrive - University College London\Desktop\transfer 10_6_25\medial_stardist_filtered"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith("_segmentation.tif"):
        print(f"Filtering: {file_name}")
        seg_path = os.path.join(input_folder, file_name)
        seg = imread(seg_path)

        # Get region properties
        props = regionprops(seg)
        if len(props) == 0:
            print("  No objects found.")
            continue

        areas = np.array([p.area for p in props])
        log_areas = np.log1p(areas)
        z_scores = zscore(log_areas)

        # Keep only objects within z-score range
        keep_mask = (z_scores >= -1) & (z_scores <= 1.5)
        labels_to_keep = [p.label for i, p in enumerate(props) if keep_mask[i]]

        filtered = np.isin(seg, labels_to_keep)
        cleaned = label(filtered)

        # Save cleaned mask
        out_mask_path = os.path.join(output_folder, file_name.replace("_segmentation.tif", "_filtered.tif"))
        imwrite(out_mask_path, cleaned.astype(np.uint16))

        # Save overlay if original image exists
        orig_img_path = os.path.join(input_folder, file_name.replace("_segmentation.tif", ".tif"))
        if os.path.exists(orig_img_path):
            orig_img = imread(orig_img_path)
            if orig_img.ndim > 2:
                orig_img = orig_img[..., 0]
            overlay = render_label(cleaned, img=orig_img)
            out_overlay_path = os.path.join(output_folder, file_name.replace("_segmentation.tif", "_filtered_overlay.png"))
            plt.imsave(out_overlay_path, overlay)

                # Plot and save histogram of object sizes with z-score cutoffs
        props_cleaned = regionprops(cleaned)
        areas_cleaned = [p.area for p in props_cleaned]

        # Plot histogram of log-areas for all objects (before filtering)
        plt.figure()
        plt.hist(log_areas, bins=30, alpha=0.7, label='All objects')
        plt.axvline(np.log1p(areas[keep_mask].min()), color='red', linestyle='--', label='z = -1 cutoff')
        plt.axvline(np.log1p(areas[keep_mask].max()), color='green', linestyle='--', label='z = 1 cutoff')
        plt.title(f"Log-area histogram with z-score cutoffs: {file_name}")
        plt.xlabel("log(Area + 1)")
        plt.ylabel("Count")
        plt.legend()
        hist_path = os.path.join(output_folder, file_name.replace("_segmentation.tif", "_logarea_hist.png"))
        plt.savefig(hist_path)
        plt.close()

        print(f"Saved filtered mask, overlay, and histogram for {file_name}")