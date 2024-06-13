# SynergiQc-PFS Prediction

## Overview

This project aims to predict Progression-Free Survival (PFS) for early-stage (stage 1 and 2) patients using the SynergiQc dataset.

## Data

### Structure

- **Training and Test Datasets**: 
  - Contains radiomic features harmonized by the method detailed in [Frontiers in Oncology](https://doi.org/10.3389/fonc.2023.1196414).

## Scripts

### Directory: Src

- **CRchu Codes**: Initial scripts for the project.
- **Radiomics Extraction**: 
  - Extracted from Tumor (T) or Non-Tumor (N) ROIs used as features in the ML pipeline.
  - Codes using radiomics+clinical data as features (clinical data added after feature selection).
- **Dataset Creation**: 
  - Scripts prefixed with `data_set` are used to create training and test cohorts.
  - For feature reduction methods (Lasso), the splitting is done post-reduction by Lasso.
