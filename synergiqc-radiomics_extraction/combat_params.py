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
import openpyxl
from dicomweb_client.api import DICOMwebClient
import dicom2nifti.settings as settings

settings. disable_validate_slice_increment()
#pydicom.config.pixel_data_handlers = ['gdcm_handler']

# get the start timen_jobs
st = time.time()
#tmp_dir = os.getenv("SLURM_TMPDIR")                
# Define the float format
float_format = '%.5f'
main_dir= "/project/166726142/lesee/Synergic-Radiomics/"

auth_token= 'jsQXPhk1MDJDRCMSmVb4No'
header = {'Authorization': 'Bearer ' + auth_token}

#giving the path of web-based dicom files
url="https://platform.paradim.science/api"
client_dcm = DICOMwebClient(url = url, headers = header)

all_studies = client_dcm.search_for_studies(offset=1300)
#all_studies = client_dcm.search_for_studies(limit = 1, get_remaining=False)
#all_studies = client_dcm.search_for_studies(offset=1290)
   
studies_dicom = [pydicom.dataset.Dataset.from_json(d) for d in all_studies]

seg_modality = ['SEG']

#combat_params_total = []
#index_studies = []

combat_key = ['PatientName', 'manufacturer', 'model', 'kernel', 'slice_thickness', 'pixel_spacing']

with open(os.path.join(main_dir,'results/combat_params_5.csv'), mode='w', newline='') as csv_file1:
    csv_writer_params = csv.writer(csv_file1)

    #to check studies case by case
    for num_std , study_dicom in enumerate(studies_dicom): 
        print('Study No. = ', num_std+1 )
        study_instance_uid = study_dicom['StudyInstanceUID'].value
        patient_ID = study_dicom['PatientID'].value
        patient_name = study_dicom['PatientName'].value         
        print('Patient Name = ', patient_name )
        metadata_study = client_dcm.retrieve_study_metadata(study_instance_uid = study_instance_uid)
        modality_values = study_dicom["00080061"]
        if any('SEG' in modality for modality in modality_values):
    #        print(modality_values)
            for meta_instance in metadata_study:
            #Find the SEG (Segmentation) object with the SOP Class UID which is typically '1.2.840.10008.5.1.4.1.1.66.4' for SEG files.            
                if meta_instance['00080016']['Value'][0] == '1.2.840.10008.5.1.4.1.1.66.4':
                    # This is a SEG file
                    SeriesInstanceUID_seg = meta_instance["00081115"]["Value"][0]['0020000E']['Value'][0]           
                else:
                    continue    

            #find the segmentation dicom dataset        
            series = client_dcm.search_for_series(study_instance_uid=study_instance_uid)
        #    print(series[6])
            num_seg = 0
            for serie in series:
                series_CT_dicom = pydicom.dataset.Dataset.from_json(serie)
                if series_CT_dicom.Modality in seg_modality:
                    num_seg +=1

                    #find the references CT series corresponding to the SEG binary image 
                if not series_CT_dicom.Modality in seg_modality:
                    instances = client_dcm.retrieve_series(study_instance_uid = study_instance_uid,
                        series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value)

                    if instances[0][0x0020000e].value == SeriesInstanceUID_seg:

                        print(series_CT_dicom.SeriesDescription)
    #                    print(instances[0])
                        manufacturer = instances[0][0x00080070].value
                        model_name = instances[0][0x00081090].value
                        convolution_kernel = instances[0][0x00181210].value
                        slice_thickness = instances[0][0x00180050].value
                        pixel_spacing = instances[0][0x00280030].value

                        combat_param = [patient_name, manufacturer, model_name, convolution_kernel, slice_thickness, pixel_spacing]
#                        combat_params_total.append(combat_param)
#                        index_studies.append(num_std+1)                    
#                        instances_CT = instances
            if num_seg >1:
    #            studies_two_seg.append(patient_name)
                print('Patient Name with more than 1 segmentation file = ', patient_name )
                continue
        else: 

            print ("no SEG file for this study")
            continue
        csv_writer_params.writerow(combat_param)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')        
exit()


writer = pd.ExcelWriter(os.path.join(main_dir,'results/combat_params.xlsx'), engine='xlsxwriter')


wb  = writer.book
df = pd.DataFrame(data=combat_params_total, index = index_studies, columns=combat_key)
#df = pd.DataFrame(features_value_total, index = selected_fkeys, columns= features_key)          #put into a dataframe format
df.to_excel(writer, sheet_name="combat params")                                     ## write into excel
wb.close()