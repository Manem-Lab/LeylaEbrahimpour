
# SynergiQc-Recurrence Prediction

## Overview

This project aims to predict recurrence for patients in the SynergiQc dataset, with recurrence recorded as a binary outcome (1 for recurrence, 0 for no recurrence).

## Data

The data used in this project includes:

- **Radiomic Features**: Extracted from tumors.
- **Clinical Data**: Associated clinical data for the patients.

## Source Files

### Data Directory

The `data/` directory contains the radiomic features and clinical data required for the analysis.

### Source Code Directory

The `src/` directory contains the machine learning (ML) codes used to predict recurrence based on radiomic features and a combination of radiomic and clinical data.

### Balancing Methods

Due to the unbalanced nature of the dataset, several balancing methods have been applied:

- **Combination of Undersampling and Oversampling**: Techniques to balance the dataset by reducing the number of majority class samples and increasing the number of minority class samples.
- **Weighted Models**: Adjusting model weights to account for class imbalance.

## Usage

1. **Prepare Data:**

   Ensure that the data is properly organized in the `data/` directory. The data should include both radiomic features and clinical data.

2. **Run ML Models:**

   run the ML codes to predict recurrence. The scripts will apply the necessary balancing methods and perform the predictions.
