# Nested Combat Normalization


This project is for harmonizing the extracted radiomic features using nested combat normalization based on https://www.nature.com/articles/s41598-022-08412-9. 

Src: Using the combat_harmonization.py code and having batch parameters of the CT images using batch_params_from_paradim.ipynb script, the batch effect can be corrected for as many as batch parameters and batch levels. ComBat or similar approaches, rely on the assumption that samples within a batch share similar characteristics. Hence it is important to have enough samples within each batch level. In terms of radiomic features extracted from CT images, the reconstruction Kernel, the slice thickness, the model of the scanner, the manufacturer and the center can be considered as batch parameters. Giving the radiomics features extracted from CT images to the combat_harmonization.py code as the input data, the effects of these parameters are corrected. The output is the corrected radiomic features. PCA is also used for a better visualization of high dimentional data before and after harmonization. The scripts to extarct radiomics from Paradim giving the IDs of the patients are also available.
