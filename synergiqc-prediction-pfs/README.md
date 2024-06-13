# SynergiQc-PFS Prediction

This project is to predict Progression Free Surival (PFS) for early stage (stage 1 and 2) patients of SynergiQc data. 

data: contains radiomic features for training and test datasets harmonized by the method in https://doi.org/10.3389/fonc.2023.1196414.

Src: contains CRchu codes which are the initial scripts. Radiomics extracted from Tumor (T) or Non-Tumor (N) ROIs used as features in the ML pipeline. Also there are codes using radiomics+clinical data as features (adding clinical data after feature selection methods). The codes starting with data_set are being used to creat training and test cohorts. In case Lasso is applied, the splitting is done after reduction by Lasso.