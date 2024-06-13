## Name
Oncotech-radiomics Tool comparison

## Description
In this project, we extract the radiomic features from CT annotated images of 164 patients provided in an h5 file by Imagia Company. The read_h5.py script is to read the h5 file and save the patient_ID, the study_ID, and the series_ID as an excel file in case needed. The subgroups of volume.data and nodulelist.data repersent the CT images and annotations for each patient. The rads_extraction_h5.py is to read the h5 file, to extract radiomics for 164 studies, and to save the results in an excel file.

To extract radiomics a parameter file is needed to customize the features before extraction. An example is provided in Params.yaml file. The path of this file should be indicated in the rads_extraction_h5.py script.


