# SynergiQc-OS Prediction

This project is to predict Overal Surival (OS) for early stage (stage 1 and 2) patients of SynergiQc data.

data: contains different training and test datasets based on the ROI from which radiomics have been extracted (Tumor or Non Tumor),harmonization by the method in https://doi.org/10.3389/fonc.2023.1196414, and applying feature readuction methods( Lasso or omitting multicollinearity) 

data_combat: contains radiomic features harmonized by Nested-ComBat harmonization method: https://rdcu.be/dKFOE

Src: contains CRchu codes which are the initial scripts. Radiomics extracted from Tumor (T) or Non-Tumor (N) ROIs used as features in the ML pipeline. Also there are codes using radiomics+clinical data as features (adding clinical data after feature selection methods). The codes starting with data_set are being used to creat training and test cohorts. In case feature reduction methods (Lasso or correlation) have been applied, the splitting is done after reduction by Lasso. 

results-plots: heatmaps and bar plots of C-index for the best models.