import h5py
import numpy as np
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
#reading the h5 file
f = h5py.File(file_h5)
#list of patients
selected_fkeys = list(f.keys())
data =[]
patient_id_total = []
series_id_total = []
study_id_total = []
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
            volume = group3['volume.data']
            #reading the contour data
            seg = group3['nodulelist.data']
            gc.collect()

# Create a DataFrame from the lists of study information
df_ids = pd.DataFrame({'patient_id': patient_id_total, "series_id": series_id_total, 'study_id': study_id_total})
# Specify the ids file path
file_oncotech_ids = os.path.join(main_dir_path,'oncotech_data_ids_new.xlsx')
# Save the IDs to Excel file
df_ids.to_excel(file_oncotech_ids, sheet_name="IDs", index=False)

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')