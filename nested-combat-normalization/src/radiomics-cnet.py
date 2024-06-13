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

settings. disable_validate_slice_increment()
#pydicom.config.pixel_data_handlers = ['gdcm_handler']

# get the start timen_jobs

st = time.time()
#tmp_dir = os.getenv("SLURM_TMPDIR")                
# Define the float format
float_format = '%.5f'

#path the main directory
#main_dir= "/project/166726142/lesee/Synergic-Radiomics"
main_dir= "/home/ulaval.ca/lesee/projects/Project-CNET/"
#path of parameter file which includes the settings for radiomics extraction
#params = os.path.join(main_dir,'src/Params_test.yaml')
params = '/project/166726142/lesee/Synergic-Radiomics/src/Params_test.yaml'
auth_token= 'YenuGm9Uk8Oekc5uPt1cBa'
header = {'Authorization': 'Bearer ' + auth_token}

#giving the path of web-based dicom files
url="https://platform.paradim.science/api"
client_dcm = DICOMwebClient(url = url, headers = header)
all_studies =[]
data = pd.read_csv(os.path.join(main_dir,'data/patients_id_cnet.csv'))
patient_names = data['PatientName']
for patient_name in patient_names:
    search_filters = {'PatientName': patient_name}
    studies = client_dcm.search_for_studies(search_filters=search_filters)
    all_studies.extend(studies)
print(len(all_studies))
      
studies_dicom = [pydicom.dataset.Dataset.from_json(d) for d in all_studies]

seg_modality = ['SEG']
header = ['PatientName', 'patientID' 'StudyInstanceUID'] 

data_features =[]
index_studies =[]

for num_std , study_dicom in enumerate(studies_dicom): 
    try: 
        #find the series instance ID from SEG file in each study
        print('Study No. = ', num_std+1 )
        study_instance_uid = study_dicom['StudyInstanceUID'].value
        patient_ID = study_dicom['PatientID'].value
        patient_name = study_dicom['PatientName'].value         
        print('Patient Name = ', patient_name )
        metadata_study = client_dcm.retrieve_study_metadata(study_instance_uid = study_instance_uid)
        modality_values = study_dicom["00080061"]
        # check if there is any segmentation file for this study
        if any('SEG' in modality for modality in modality_values):

            for meta_instance in metadata_study:
    #            print(meta_instance['0008103E'])
    #            if meta_instance['0008103E']['Value'][0]=='Segmentation':
            #Find the SEG (Segmentation) object with the SOP Class UID which is typically '1.2.840.10008.5.1.4.1.1.66.4' for SEG files.            
                if meta_instance['00080016']['Value'][0] == '1.2.840.10008.5.1.4.1.1.66.4':

                    SeriesInstanceUID_seg = meta_instance["00081115"]["Value"][0]['0020000E']['Value'][0]
                else:
                    continue    
            #find the segmentation dicom dataset        
            series = client_dcm.search_for_series(study_instance_uid=study_instance_uid)
            num_seg = 0  
            
            for serie in series:
                series_CT_dicom = pydicom.dataset.Dataset.from_json(serie)
                if series_CT_dicom.Modality in seg_modality:
                    num_seg +=1
                    print(series_CT_dicom.SeriesDescription)                    
                    instances_seg = client_dcm.retrieve_series(study_instance_uid = study_instance_uid,
                        series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value)                    
                    #read the SEG dicom dataset as a sitk image file
                    reader_seg = pydicom_seg.MultiClassReader()
                    result = reader_seg.read(instances_seg[0])
                    img_seg = result.image

                    segment_metadata = instances_seg[0].SegmentSequence
                    label_name = []
                    label_number =[] 
                    for segment in segment_metadata:
                        segment_number = segment.SegmentNumber
                        segment_label = segment.SegmentLabel
    #                        segment_description = segment.SegmentDescription
                        label_number.append(segment_number)
                        label_name.append(segment_label)


                #find the references CT series corresponding to the SEG binary image 
                if not series_CT_dicom.Modality in seg_modality:
                    instances = client_dcm.retrieve_series(study_instance_uid = study_instance_uid,
                        series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value)
                    if instances[0][0x0020000e].value == SeriesInstanceUID_seg:
                        print(series_CT_dicom.SeriesDescription)
                        instances_CT = instances
                        #convert the CT dicom images to one nifti file
                        with tempfile.TemporaryDirectory() as tmp_dir:      
                            img_CT = dicom2nifti.convert_dicom.dicom_array_to_nifti(instances_CT,os.path.join(tmp_dir+'_CT.nii'), reorient_nifti=True)["NII"]
                            CT_nifti_path = os.path.join(tmp_dir+'_CT.nii')
                            #read the CT nifit as a sitk image file                 
                            img_CT_stk = sitk.ReadImage(CT_nifti_path)
                        #remove the temporary nifti file after reading
                        os.remove(CT_nifti_path)
                        
                        
            # find if the study has more than 1 segmentation file
            if num_seg >1:
    #            studies_two_seg.append(patient_name)
                print('Patient Name with more than 1 segmentation file = ', patient_name ) 
                continue            
                        

            # Find number of labels (segments)
            N_segments = len(label_name)

            # Find all labels that start with "Tumor"
            tumor_labels = [label for label in label_name if label.startswith("T")]

            # Loop through all the "Tumor" labels and extract radiomics for each tumor
            for tumor_label in tumor_labels:
            # Extract the name after "Tumor"
#                tumor_name = tumor_label[len("Tumor"):]        
                tumor_segment_number = label_number[label_name.index(tumor_label)]     

                # Extract radiomic features
                extractor = featureextractor.RadiomicsFeatureExtractor(params, additionalInfo=True)
                extractor.settings['n_jobs'] = -1
            #   features = extractor.execute(img_CT_stk, img_seg_resampled, label=int(labels_resampled[-1]))
                features = extractor.execute(img_CT_stk, img_seg, label=int(tumor_segment_number))
                features_key = ['PatientName', 'PatientID', 'StudyInstanceUID' ]
                features_value = [patient_name, patient_ID, study_instance_uid]
                for num , (key, val) in enumerate(six.iteritems(features)):
                    features_key.append(key)
                    features_value.append(val)
                data_features.append(features_value)
                index_studies.append(num_std+1)

        else: 
#            studies_no_seg.append(patient_name)
    
            print ("No SEG file for this study =", patient_name)
            continue
        
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        #print(features_key)
        
# write into excel      
writer = pd.ExcelWriter(os.path.join(main_dir,'results/radiomicsfeatures_kheops_CNET_updated_bin_0.05.xlsx'), engine='xlsxwriter')

wb  = writer.book
df = pd.DataFrame(data=data_features, index = index_studies, columns=features_key)
df.to_excel(writer, sheet_name="radiomic features")                                     
wb.close()
# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')