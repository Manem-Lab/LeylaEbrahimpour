import h5py
import numpy as np
from radiomics import featureextractor, getTestCase
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk
import six
import pandas as pd
import os
import gc
import time

# get the start time
st = time.time()
#the path of main folder which includes the h5 file
main_dir_path = "the path of main folder"
file_h5 = os.path.join(main_dir_path, 'generated_20211222_iso1x1x1.h5')
#the path of parameter file through which the radiomics are customized
params = os.path.join(main_dir_path, "Params.yaml")
#reading the h5 file
f = h5py.File(file_h5)
#list of patients
selected_fkeys = list(f.keys())
data =[]
patient_id_total = []
series_id_total = []
study_id_total = []
extractor = RadiomicsFeatureExtractor(params)
#iterate over sub-groups for each key(patient) in h5 file
for num , patient_id in enumerate(selected_fkeys):
    patient_id_total.append(patient_id)
    group1 = f[patient_id]
    for key_gr1 in group1.keys():
        series_id_total.append(key_gr1)        
        group2 = group1[key_gr1]
        for key_gr2 in group2.keys():
            study_id_total.append(key_gr2)
            group3 = group2[key_gr2]
            #reading the CT image data
            img = sitk.GetImageFromArray(group3['volume.data'])
            #reading the contour data
            seg = sitk.GetImageFromArray(np.multiply(group3['nodulelist.data'], 1))
            seg = seg != 0
            #extract radiomics (label can be speified inside the param file)
            features = extractor.execute(img, seg, label = 1)
            features_key = []
            features_value = []
            for num , (key, val) in enumerate(six.iteritems(features)):
                features_key.append(key)
                features_value.append(val)
            data.append(features_value)
            gc.collect()

# Create a DataFrame from the lists of study information
df_ids = pd.DataFrame({'patient_id': patient_id_total, "series_id": series_id_total, 'study_id': study_id_total})
# Specify the ids file path
file_oncotech_ids = os.path.join(main_dir_path,'oncotech_data_ids_new.xlsx')
# Save the IDs to Excel file
df_ids.to_excel(file_oncotech_ids, sheet_name="IDs", index=False)

# Create a DataFrame from the list of list of features
df_rads = pd.DataFrame(data=data, index = selected_fkeys, columns=features_key)
# Specify the rads file path
file_radiomics = os.path.join(main_dir_path,'radiomicsfeatures_new.xlsx')
# Save the DataFrame to Excel file
df_rads.to_excel(file_radiomics, sheet_name="radiomic features")                                     ## write into excel

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')