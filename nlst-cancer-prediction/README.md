# NLST-Cancer Prediction

## Overview

This project investigates the impact of imaging parameters and delta radiomic features on lung cancer risk prediction using a subsample of the NLST data. The study utilizes radiomic features extracted from longitudinal scans, with CT images available for three time slots: T0 (1999), T1 (2000), and T2 (2001). Details of the sample can be found in the [research article](https://doi.org/10.1002/cam4.1852).

## Data

### Contents

- **Cohort IDs**: Contains the IDs of Cohort 1 and Cohort 2 used in the study.
- **Labels and Batch Parameters**: Available for the CT scans of Cohort 1 and 2.
- **Radiomic Features**: Includes T0, T0+delta T1, T0+delta T2, and delta radiomic features.
  - *Delta T1*: Difference in radiomics between T0 and T1.
  - *Delta T2*: Difference in radiomics between T0 and T2.

### Data Subdirectories

- **data/**:
  - Contains the necessary IDs, labels, and batch parameters for Cohort 1 and 2.
  - **data_lasso/**: Training and test cohorts for each radiomic feature set after feature reduction by Lasso.
  - **data_radiomics/**: Extracted radiomics for all time slots and both cohorts.
  - **annotations.json**: Annotations of NLST images used in the Sybil paper (not used in this project).

## Source Files

### src Directory

The `src/` directory contains scripts for various tasks:

- **Radiomics Extraction**: Scripts for extracting radiomics from the NLST subsample.
- **ComBat Harmonization**: Harmonization of radiomic features ([reference](https://rdcu.be/dKFOE)).
- **Feature Reduction**: Lasso feature reduction.
- **ML Pipelines**: Machine learning pipelines for cancer risk prediction.
- **Visualization**:
  - PCA visualization of radiomic features.
  - Bar plots and box plots for ML pipeline results.
  - Plotting ROI by overlaying segmentation files on the corresponding CT images.
- **Additional Scripts**: Reading JSON files, finding NLST IDs, and other auxiliary tasks.

### Results Directory

The `results/` directory includes:

- **Harmonized Radiomic Features**: Results of ComBat harmonization for different batch parameters.
- **ML Pipeline Results**: Results for different combinations of features and models.
- **Plots**: Visual representations of results, including PCA, bar plots, and box plots.
- **Final ML Output**: CSV file with the results of different ML models and feature combinations.

## Usage

1. **Prepare Data:**

   Ensure that the data is properly organized in the `data/` directory. This includes IDs, labels, batch parameters, and extracted radiomic features.

2. **Run Harmonization and Feature Reduction:**

   Use the scripts in the `src/` directory to perform ComBat harmonization, Lasso feature reduction, and ML pipelines.
s.csv

