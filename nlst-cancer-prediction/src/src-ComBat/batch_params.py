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
#float_format = '%.5f'

#path the main directory
main_dir= "/home/ulaval.ca/lesee/projects/Project-NLST/"

auth_token= 'kkaPH62MZeS01bqNeCBDDO'
header = {'Authorization': 'Bearer ' + auth_token}

#giving the path of web-based dicom files
url="https://platform.paradim.science/api"
client_dcm = DICOMwebClient(url = url, headers = header)
all_studies =[]
data = pd.read_csv(os.path.join(main_dir,'data/data_batch-clinical-labels/Cohort2_labels.csv'))
patient_ids = data['PatientID']
#patient_ids = ['100012']
for patient_id in patient_ids:
    search_filters = {'PatientID': patient_id}
    studies = client_dcm.search_for_studies(search_filters=search_filters)
    all_studies.extend(studies)
    print("Number of Studies with ID "+str(patient_id), len(studies))
    
print(len(all_studies))

studies_dicom = [pydicom.dataset.Dataset.from_json(d) for d in all_studies]

combat_key = ['PatientName', 'manufacturer', 'model', 'kernel', 'slice_thickness', 'pixel_spacing']
CT_modality = ['CT']
with open(os.path.join(main_dir,'results/combat_params_cohort2.csv'), mode='w', newline='') as csv_file1:
    csv_writer_params = csv.writer(csv_file1)
    combat_key = ['PatientName', 'PatientID', 'StudyDate', 'StudyInstanceUID', 'SeriesInstanceUID', 'SeriesDescription', 'ImageType', 'Manufacturer', 'ModelName', 'ConvolutionKernel',
                  'ReconDiameter','SliceThickness', 'kVp','mAs', 'PixelSpacing']
  
    # Write the header
    csv_writer_params.writerow(combat_key)
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
            metadata_study = client_dcm.retrieve_study_metadata(study_instance_uid = study_instance_uid)
            series = client_dcm.search_for_series(study_instance_uid=study_instance_uid)           
#            num_CT = 0
            for serie in series:
                series_CT_dicom = pydicom.dataset.Dataset.from_json(serie)
                if series_CT_dicom.Modality in CT_modality:                                  

                    instances_ct = client_dcm.retrieve_series(study_instance_uid = study_instance_uid, 
                                                              series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value)
                    if len(instances_ct) > 5:                        
                        series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value
                        sereies_description = series_CT_dicom.SeriesDescription
                        image_type = instances_ct[0][0x00080008].value
                        manufacturer = instances_ct[0][0x00080070].value
                        model_name = instances_ct[0][0x00081090].value
                        convolution_kernel = instances_ct[0][0x00181210].value
                        recon_diameter = instances_ct[0][0x00181100].value
                        slice_thickness = instances_ct[0][0x00180050].value
                        kVp = instances_ct[0][0x00180060].value
                        mAs = instances_ct[0][0x00181152].value
                        pixel_spacing = instances_ct[0][0x00280030].value

                        
                        combat_value = [patient_name, patient_ID, study_date, study_instance_uid, series_instance_uid, sereies_description, image_type,  manufacturer, model_name, convolution_kernel,
                                        recon_diameter, slice_thickness, kVp, mAs, pixel_spacing]

                        csv_writer_params.writerow(combat_value)             
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            

     

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

