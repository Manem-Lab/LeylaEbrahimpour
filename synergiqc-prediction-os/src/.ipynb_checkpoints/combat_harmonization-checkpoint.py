import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os
import NestedcomBat as nested
#import GMMComBat as gmmc

#Load the data
main_dir= "/home/ulaval.ca/lesee/projects/Project2-synergiqc/OS/"

data = pd.read_excel(os.path.join(main_dir,'data/T-SynergiQc-annotated_centers_combat_radiomics_clinical_normal.xlsx'))

# Filter the data for stage
filtered_data = data[data['Stage'].isin([1,2])]

# Explicitly create a copy of the data
filtered_data_copy = filtered_data.copy()

#drop columns not used in this work
filtered_data_copy.drop(columns = ['Recurrence', 'PFS_months', 'PFS_days']
                        
#Remove rows with missing data
filtered_data_copy.dropna(inplace=True)
                        
#specify the batch list                                            
batch_list = ['Center', 'manufacturer', 'model', 'kernel', 'slice_thickness']

# specify the categorical and continous clinical covariates
categorical_cols = ['Smoking', 'Subtype', 'Sex']
continuous_cols = ['Age']
patient_name = filtered_data_copy['PatientName']
                        
#define a string-based covariate datframe                       
covars_string = pd.DataFrame()
covars_string[categorical_cols] = filtered_data_copy[categorical_cols].copy()
covars_string[batch_list] = filtered_data_copy[batch_list].copy()
                        
#load the continous clinical covariate                       
covars_quant = filtered_data_copy[continuous_cols]

#specify the features
data_df = filtered_data_copy.drop(columns=['PatientName', 'PatientID', 'StudyInstanceUID', 'Smoking', 'Age', 'Subtype', 'VitalStatus', 'Sex', 'Stage', 'OS_months', 'OS_days', ,'Center', 'manufacturer', 'model', 'kernel', 'slice_thickness', 'pixel_spacing'])

dat = dat.T.apply(pd.to_numeric)                        

#label-encode the string covariates
covars_cat = pd.DataFrame()
for col in covars_string:
    stringcol = covars_string[col]
    le = LabelEncoder()
    le.fit(list(stringcol))
    covars_cat[col] = le.transform(stringcol)

#concatenate the label-enoded categorical (batch+clinical) and continous clinical covariates                         
covars = pd.concat([covars_cat, covars_quant], axis=1)

output_data = nested.NestedComBat(dat, covars, batch_list, categorical_cols=categorical_cols,
                                   continuous_cols=continuous_cols, drop=True, write_p=True, filepath=main_dir)
#output_data = gmmc.GMMComBat(dat, caseno, covars,  filepath=filepath2, categorical_cols=categorical_cols,
#                             continuous_cols=continuous_cols, write_p=True, plotting=True)

# output_df = pd.DataFrame.from_records(output_data.T)
# output_df.columns = feature_cols
write_df = pd.concat([patient_name, output_data], axis=1)
write_df.to_csv(os.path.join(main_dir,'data_combat/combat_harmonized_features.csv', float_format='%.7f')
                        
nested.feature_kstest_histograms(output_data, covars, batch_list, main_dir)

f_dict = nested.MultiComBat(output_data.T, covars, batch_list, filepath=main_dir, categorical_cols=categorical_cols,
                          continuous_cols=continuous_cols, write_p=True, plotting=True)
for col in batch_list:
    write_df = pd.concat([patient_name, f_dict[col]])
    write_df.to_csv(os.path.join(main_dir, 'data_combat/combat_'+col+'_harmonized_features.csv')