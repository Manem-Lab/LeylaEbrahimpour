import os
import csv
import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase
from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
import pydicom
from pydicom import dcmread
import dicom2nifti
import pydicom_seg
import io
import tempfile
import gc
import time
import shutil
from dicomweb_client.api import DICOMwebClient
import dicom2nifti.settings as settings
import nibabel as nib
import glob
settings. disable_validate_slice_increment()
#pydicom.config.pixel_data_handlers = ['gdcm_handler']
# get the start timen_jobs

st = time.time()
#tmp_dir = os.getenv("SLURM_TMPDIR")                
# Define the float format
float_format = '%.5f'

#path the main directory
main_dir= "/home/ulaval.ca/lesee/projects/Project-NLST/"
main_dir_seg = "/project/166726187/NLST-mask/C2_nii_unzip"



#path of parameter file which includes the settings for radiomics extraction
params = '/project/166726142/lesee/Synergic-Radiomics/src/Params_test.yaml'
auth_token= 'kkaPH62MZeS01bqNeCBDDO'
header = {'Authorization': 'Bearer ' + auth_token}

#giving the path of web-based dicom files
url="https://platform.paradim.science/api"
client_dcm = DICOMwebClient(url = url, headers = header)

all_studies =[]
data = pd.read_csv(os.path.join(main_dir,'data/data_batch-clinical-labels/Cohort2_labels.csv'))

#patient_ids = ["100005", "100095"]
patient_ids = data['PatientID']
for patient_id in patient_ids:
    search_filters = {'PatientID': patient_id}
    studies = client_dcm.search_for_studies(search_filters=search_filters)
    all_studies.extend(studies)
print("Number of Studies with this ID :", len(all_studies))
      
#find the first 'limit' number of studies  
#studies = client_dcm.search_for_studies()
#studies = client_dcm.search_for_studies(limit = 1000, get_remaining=False)
#studies = client_dcm.search_for_studies(offset=1290)
studies_dicom = [pydicom.dataset.Dataset.from_json(d) for d in all_studies]

CT_modality = ['CT']
data_features =[]
index_studies =[]
index_studies_10_ct_slice = []
date_studies_10_ct_slice = []
for num_std , study_dicom in enumerate(studies_dicom): 
    try: 
        #find the series instance ID from SEG file in each study
        study_instance_uid = study_dicom['StudyInstanceUID'].value
        study_date = study_dicom['StudyDate'].value
        patient_ID = study_dicom['PatientID'].value
        patient_name = study_dicom['PatientName'].value  
        print('Study No. = ', num_std+1 )
        print('Patient ID = ', patient_ID )
        print('Study Date = ', study_date )
    #    metadata_study = client_dcm.retrieve_study_metadata(study_instance_uid = study_instance_uid)
    #        modality_values = study_dicom["00080061"]
    #        print("Modality Values:", modality_values)


        series = client_dcm.search_for_series(study_instance_uid=study_instance_uid)           
    #    num_CT = 0
    #    series_len_instances= []
    #    series_instance_uid = []
        max_slices = 10
        series_with_max_slices = None
        for ind, serie in enumerate(series):
            series_CT_dicom = pydicom.dataset.Dataset.from_json(serie)
            if series_CT_dicom.Modality in CT_modality:
    #                num_CT +=1      
                instances_ct = client_dcm.retrieve_series(study_instance_uid = study_instance_uid, 
                                                          series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value)
                num_slices = len(instances_ct)
                if num_slices > max_slices:
                    max_slices = num_slices
                    series_with_max_slices = serie
                    series_with_max_slices_instance_uid = series_CT_dicom['SeriesInstanceUID'].value
                    instances_ct_with_max_slices = instances_ct
                    convolution_kernel_with_max_slices = instances_ct[0][0x00181210].value
                    
                    
        if max_slices ==10:
            index_studies_10_ct_slice.append(patient_ID)
            date_studies_10_ct_slice.append(study_date)
            continue


        print("Serie ID with maximum number of slices:", series_with_max_slices_instance_uid)
        #convert the CT dicom images to one nifti file
        with tempfile.TemporaryDirectory() as tmp_dir:      
            img_CT = dicom2nifti.convert_dicom.dicom_array_to_nifti(instances_ct_with_max_slices,os.path.join(tmp_dir+'_CT.nii'), reorient_nifti=True)["NII"]
            CT_nifti_path = os.path.join(tmp_dir+'_CT.nii')                            
            #read the CT nifit as a sitk image file                 
            img_CT_stk = sitk.ReadImage(CT_nifti_path)
        #remove the temporary nifti file after reading
            os.remove(CT_nifti_path)

        # check if there is any segmentation file for this study
        if study_date == "19990102":
        # Pattern to match the subfolder
            subfolder_seg_pattern_T0 = os.path.join(main_dir_seg, f'*{patient_ID}*_T0.nii')
            subfolder_seg_T0 = glob.glob(subfolder_seg_pattern_T0)[0]  # Take the first matching subfolder
            # Pattern to match the NIfTI file within the subfolder
            nifti_seg_file_pattern_T0 = os.path.join(subfolder_seg_T0, f'*{patient_ID}*T0.nii')
            nifti_seg_file_path= glob.glob(nifti_seg_file_pattern_T0)[0]  # Take the first matching file

        elif study_date == "20000102":

        # Pattern to match the subfolder
            subfolder_seg_pattern_T1 = os.path.join(main_dir_seg, f'*{patient_ID}*_T1.nii') 
            subfolder_seg_T1 = glob.glob(subfolder_seg_pattern_T1)[0]  # Take the first matching subfolder                    
            # Pattern to match the NIfTI file within the subfolder
            nifti_seg_file_pattern_T1 = os.path.join(subfolder_seg_T1, f'*{patient_ID}*T1.nii')
            nifti_seg_file_path= glob.glob(nifti_seg_file_pattern_T1)[0]  # Take the first matching file                    

        elif study_date == "20010102":
        # Pattern to match the subfolder
            subfolder_seg_pattern_T2 = os.path.join(main_dir_seg, f'*{patient_ID}*_T2.nii')                    
            subfolder_seg_T2 = glob.glob(subfolder_seg_pattern_T2)[0]  # Take the first matching subfolder                                            
            # Pattern to match the NIfTI file within the subfolder
            nifti_seg_file_pattern_T2 = os.path.join(subfolder_seg_T2, f'*{patient_ID}*T2.nii')                        
            nifti_seg_file_path= glob.glob(nifti_seg_file_pattern_T2)[0]  # Take the first matching file    
        else: 

            print ("No mask for Patient ID " + str(patient_ID)+" with Study Date "+ str (study_date))
            continue

            #read the seg nifit as a sitk image file
        img_mask = sitk.ReadImage(nifti_seg_file_path)
        mask_array = sitk.GetArrayFromImage(img_mask)

        print("Number of maximum Slices =", max_slices)
        print("Number of segmentation Slices =", len(mask_array))

        # Extract radiomic features
        extractor = featureextractor.RadiomicsFeatureExtractor(params, additionalInfo=True)
        extractor.settings['n_jobs'] = -1
    #   features = extractor.execute(img_CT_stk, img_seg_resampled, label=int(labels_resampled[-1]))
    #    features = extractor.execute(img_CT_stk, img_mask, label=int(tumor_segment_number))
        features = extractor.execute(img_CT_stk, img_mask, label = 1)
        features_key = ['PatientName', 'PatientID', 'StudyDate', 'kernel', 'CtSlices', 'SegSlices', 'StudyInstanceUID', 'SeriesInstanceUID']
        features_value = [patient_name, patient_ID, study_date, convolution_kernel_with_max_slices, max_slices, len(mask_array), study_instance_uid, series_with_max_slices_instance_uid]
        for num , (key, val) in enumerate(six.iteritems(features)):
            features_key.append(key)
            features_value.append(val)
        data_features.append(features_value)
        index_studies.append(num_std+1)
    #                    break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    #print(features_key)        
        
# write into excel      
writer = pd.ExcelWriter(os.path.join(main_dir,'results/radiomicsfeatures_kheops-NLST-Dmitrii-max_slice_Cohort2_repeat.xlsx'), engine='xlsxwriter')


wb  = writer.book
df_features = pd.DataFrame(data=data_features, index = index_studies, columns=features_key)
#df_features.to_csv(os.path.join(main_dir,'results/radiomicsfeatures_kheops-NLST-Dmitrii-max_slice_Cohort1_repeat.csv'))
df_features.to_excel(writer, sheet_name="radiomic features")                                     
wb.close()


# Create a DataFrame for studies with maximum 10 CT slices
df_10_slices = pd.DataFrame({
    'PatientID': index_studies_10_ct_slice,
    'StudyDate': date_studies_10_ct_slice
})

# To save as CSV
df_10_slices.to_csv(os.path.join(main_dir,'results/10_slices-NLST-Dmitrii-max_slice_Cohort2_repeat.csv'), index=False)

# To save as Excel
#df.to_excel('output.xlsx', index=False, engine='openpyxl')


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')