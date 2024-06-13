# SynergiQc-OS Prediction

## Overview

This project aims to predict Overall Survival (OS) for early-stage (stage 1 and 2) patients using the SynergiQc dataset.

## Data

### Structure

- **Training and Test Datasets**: 
  - Based on the ROI from which radiomics have been extracted (Tumor or Non-Tumor).
  - Harmonized by the method detailed in [Frontiers in Oncology](https://doi.org/10.3389/fonc.2023.1196414).
  - Feature reduction methods applied (Lasso or omitting multicollinearity).

- **data_combat**: 
  - Contains radiomic features harmonized using the Nested-ComBat harmonization method as detailed in [Scientific Reports](https://rdcu.be/dKFOE).

## Scripts

### Directory: Src

- **CRchu Codes**: Initial scripts for the project.
- **Radiomics Extraction**: 
  - Extracted from Tumor (T) or Non-Tumor (N) ROIs used as features in the ML pipeline.
  - Codes using radiomics+clinical data as features (clinical data added after feature selection).
- **Dataset Creation**: 
  - Scripts prefixed with `data_set` are used to create training and test cohorts.
  - For feature reduction methods (Lasso or correlation), the splitting is done post-reduction by Lasso.

### Directory: results-plots

- Contains heatmaps and bar plots of C-index for the best models.
