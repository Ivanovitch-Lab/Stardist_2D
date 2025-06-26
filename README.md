# Stardist_2D

 StarDist 2D Segmentation Pipeline

This repository contains a script for performing 2D nuclear segmentation on fluorescence microscopy images using the pre-trained [StarDist](https://github.com/stardist/stardist) model.

---

 Purpose

Automatically segment nuclei from TIFF images using the StarDist `2D_versatile_fluo` model and save:

- The segmentation mask as a `.tif` file
- The label array as `.npy`
- An RGB overlay image as `.png`

---

 Input Requirements

- Input directory must contain **TIFF files**.
- Only files with `"C1"` in their filenames and `.tif` extension will be processed (e.g., `sample_C1.tif`).

 Folder Structure

```text
input_folder/
├── image1_C1.tif
├── image2_C1.tif
├── image3_C2.tif  <-- Skipped (not C1)
