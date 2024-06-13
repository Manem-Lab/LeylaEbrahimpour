# Harmonization of Radiomic Features using Nested ComBat Normalization

## Overview

This project focuses on harmonizing the extracted radiomic features using the nested ComBat normalization method, based on the research published in [Nature](https://www.nature.com/articles/s41598-022-08412-9).

## Source Files

### combat_harmonization.py

The `combat_harmonization.py` script is used to correct batch effects in radiomic features extracted from CT images. The script leverages batch parameters obtained from CT images using the `batch_params_from_paradim.ipynb` notebook. Batch effects can be corrected for multiple batch parameters and levels.

### Batch Parameters

ComBat and similar approaches rely on the assumption that samples within a batch share similar characteristics. It is crucial to have sufficient samples within each batch level. In the context of radiomic features from CT images, the following can be considered batch parameters:

- Reconstruction Kernel
- Slice Thickness
- Model of the Scanner
- Manufacturer
- Center

### Harmonization Process

By providing the radiomic features extracted from CT images as input data to the `combat_harmonization.py` script, the effects of the batch parameters are corrected. The output is the corrected radiomic features.

### PCA Visualization

Principal Component Analysis (PCA) is used for better visualization of high-dimensional data before and after harmonization. This helps in understanding the impact of harmonization on the radiomic features.

### Extracting Radiomics from Paradim

Scripts for extracting radiomic features from Paradim, given the IDs of the patients, are also available in this project.

## Usage

1. **Obtain Batch Parameters:**

   Use the `batch_params_from_paradim.ipynb` notebook to extract batch parameters from CT images.

2. **Run Harmonization:**

   Use the `combat_harmonization.py` script to correct batch effects in the radiomic features.
