# Synergiqc-radiomics_extraction

## Name
Radiomics extraction from DICOM CT images

## Description

In this project, we extract the radiomic features from DICOM CT annotated images using Pyradiomics and save them in a csv file. The original DICOM CT images and the segmentation in DICOM SEG format should be converted to files readable by SimpleITK in python. Then the converted files can be given to Pyradiomics in order to extract radiomics. The codes can extract radiomics from tumors and non-tumors regions of the multilabel segmentation file. 
To extract radiomics a parameter file is needed to customize the features before extraction. An example is provided in Params.yaml file. The path of this file should be indicated in the radiomics extraction python script.

The codes with harmonized extension are written based on the harmonization process in https://doi.org/10.3389/fonc.2023.1196414 and the corresponding param file should be called when using them for radiomics extraction.
