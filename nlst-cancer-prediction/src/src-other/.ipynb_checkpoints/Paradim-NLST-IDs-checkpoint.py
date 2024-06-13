import numpy as np
import pydicom
from pydicom import dcmread
import time
from dicomweb_client.api import DICOMwebClient
import os
import csv
import pandas as pd


#pydicom.config.pixel_data_handlers = ['gdcm_handler']



# get the start time
st = time.time()

main_dir= "/home/ulaval.ca/lesee/projects/Project-NLST/"
auth_token= 'kkaPH62MZeS01bqNeCBDDO'
header = {'Authorization': 'Bearer ' + auth_token}

#giving the path of web-based dicom files

url="https://platform.paradim.science/api"
client_dcm = DICOMwebClient(url = url, headers = header)


#ind_studyUID = []
#ind_patientID = []
#ind_seriesUID = []

#studies = client_dcm.search_for_studies(limit=3, get_remaining=False)

data = pd.read_csv(os.path.join(main_dir,'data/data_radiomics/Cohort1_labels.csv'))    
patient_ids = data['PatientID']

studies = []

for patient_id in patient_ids:
    search_filters = {'PatientID': patient_id}
    filtered_studies = client_dcm.search_for_studies(search_filters=search_filters)
    studies.extend(filtered_studies)
    
print("Number of studies= " , len(studies))

    
studies_dicom = [pydicom.dataset.Dataset.from_json(d) for d in studies]

header = ['patientID', 'StudyDate','StudyInstanceUID', 'SeriesInstanceUID', 'NumberInstances']        

with open(os.path.join(main_dir,'paradim_NLST_ids_Dmitrii_cohort1_anti_paradim.csv'), mode='w', newline='') as csv_file1:
    csv_writer_IDs= csv.writer(csv_file1)
    csv_writer_IDs.writerow (header)
    
    for num_std , study_dicom in enumerate(studies_dicom): 
        try: 
            #find the series instance ID from SEG file in each study
            print('Study No. = ', num_std+1 )
            study_instance_uid = study_dicom['StudyInstanceUID'].value
            patient_ID = study_dicom['PatientID'].value
            study_date = study_dicom['StudyDate'].value
#            ind_patientID.append(patient_ID )
#            ind_studyUID.append(study_instance_uid )
            
            #find the series  
            series = client_dcm.search_for_series(study_instance_uid=study_instance_uid)
            for serie in series:
                series_CT_dicom = pydicom.dataset.Dataset.from_json(serie)
                series_instance_uid = series_CT_dicom['SeriesInstanceUID'].value
#                ind_seriesUID.append(series_instance_uid)
                # Fetch the instances for the current series
                instances = client_dcm.retrieve_series(study_instance_uid = study_instance_uid, series_instance_uid = series_instance_uid)    
                instance_count = len(instances)


               # Print all DICOM tags and their values for the first instance
#                for tag, value in instances[0].items():
#                    print(f"{tag}: {value}")
                # Fetch the series date from the first instance in the series
#                series_date = instances[0].get('AcquisitionDate')
#                print(series_date)
                csv_writer_IDs.writerow([patient_ID]+[study_date]+[study_instance_uid]+[series_instance_uid]+[instance_count])
#                csv_writer_IDs.writerow([patient_ID, study_instance_uid, series_instance_uid, instance_count])

        except Exception as e:
            # Handle any exceptions that occur during the loopinstance_count]
            print(f"An error occurred: {str(e)}")
            
            