# NLST-Cancer Prediction


This project is to investigate the impact of imaging parameters and delta radiomic features on lung cancer risk prediction using a subsample of NLST data by utilizing radiomic features extracted from the longitudinal scans. Radiomics were extracted from segmentations of NLST images with CTs availbale for three time slots (T0 (1999), T1(2000), and T2(2001)).  Details of the sample in https://doi.org/10.1002/cam4.1852. 

data: contains the IDs of Cohort1 and Cohort 2 used in this paper: https://doi.org/10.1002/cam4.1852 for which we have the segmentations. Also the labels and batch parameters of the CT scan of cohort 1 and 2 are available in data. Radiomics used in this study are T0, T0+delta T1, T0+deltaT2, and delta. delta T1 is the difference of radiomics between T0 and T1. delta T2 is the difference of radiomics between T0 and T2 (details of the study design in https://doi.org/10.1002/cam4.1852). data_lasso contains the training and test cohorts for each of radiomic features sets after feature reduction by Lasso. data_radiomics contains the extracted radiomics for all the timeslots and both cohorts. The json file contains the annotaions of NLST images which were used in Sybil paper (https://doi.org/10.1200/jco.22.01345). This data was not used in this prject.

src: This folder contains scripts for radiomics extraction of this subsample of NLST (data available in the lab) ComBat harmonization of radiomic features (https://rdcu.be/dKFOE), Feature reduction by Lasso, ML pipelines, PCA visualization, bar plots and box plots for the results of ML-pipeline, plotting ROI by overlaying the segmentation file on the corresponding CT using the original image and the the image used by pyradiomics for feture extraction. There are also a couple of extra scripts to read json file and finding the NLST IDS of this sample which are available on Paradim.

results: contains harmonized radiomic features, ML pipeline results for different combination of features, harmonization by ComBat for different batch parameters, and SMOTE balancing. Also plots are included. The final ML output for different combination of models has been saved in one csv file. 


