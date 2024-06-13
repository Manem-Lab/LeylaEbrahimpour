# Synergiqc-Radiomics_Extraction

## Overview

This project involves extracting radiomic features from DICOM CT annotated images using Pyradiomics. The features are then saved into a CSV file for further analysis.

## Data

### Input Data

- **Original DICOM CT Images**: 
  - These are the raw CT images in DICOM format.
- **Segmentation Files**:
  - Annotated regions in DICOM SEG format, which need to be converted to a format readable by SimpleITK in Python.

### Output Data

- **Radiomic Features**:
  - Extracted features saved in a CSV file.

## Scripts

### Radiomics Extraction

- **Conversion**:
  - Convert DICOM CT images and segmentation files to formats readable by SimpleITK.
  
- **Radiomics Extraction**:
  - Use Pyradiomics to extract features from the converted files. 
  - The script supports extracting radiomics from both tumors and non-tumor regions in multi-label segmentation files.
  
- **Harmonized Extraction**:
  - Scripts with a "harmonized" extension implement the harmonization process described in [Frontiers in Oncology](https://doi.org/10.3389/fonc.2023.1196414).
  
### Parameter File

- **Params.yaml**:
  - A parameter file needed to customize the feature extraction process.
  - An example file is provided in the repository. The path to this file should be specified in the radiomics extraction script.

