# Oncotech-Radiomics Tool Comparison

## Overview

This project focuses on extracting radiomic features from CT annotated images of 164 patients provided in an h5 file by Imagia Company. The goal is to compare different radiomic extraction tools and methods.

## Description

### Data

- **Source**: The data consists of CT images and annotations for 164 patients, provided in an h5 file by Imagia Company.
- **Structure**: The `volume.data` and `nodulelist.data` subgroups contain the CT images and annotations for each patient.

### Scripts

- **read_h5.py**: 
  - Reads the h5 file and saves the patient ID, study ID, and series ID as an Excel file for easy reference.

- **rads_extraction_h5.py**:
  - Reads the h5 file, extracts radiomic features for the 164 studies, and saves the results in an Excel file.
  - Requires a parameter file to customize the features before extraction. An example parameter file is provided in `Params.yaml`.

