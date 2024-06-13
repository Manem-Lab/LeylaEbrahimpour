import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from skrebate import ReliefF, SURF, MultiSURF
from sklearn.linear_model import Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct, WhiteKernel, RBF, Matern, ConstantKernel, ExpSineSquared, RationalQuadratic, Product)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from imblearn.under_sampling import RandomUnderSampler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
#from skopt import BayesSearchCVba
import os
import json
import time
import joblib
# get the start time
st = time.time()
file_name = 'T-OS_st1-2_rad_dataset-corr'
#Load the data
main_dir= "/home/ulaval.ca/lesee/projects/Project2-synergiqc/OS/"

data = pd.read_excel(os.path.join(main_dir,'data/matched_clinical_T_NT_1711-clinical-rads.xlsx'))
# Identify and drop columns with all NaN values
#data.dropna(axis=1, how='all', inplace=True)

# Filter the data for stage
filtered_data = data[data['Stage'].isin([1,2])]

# Explicitly create a copy of the data
filtered_data_copy = filtered_data.copy()

filtered_data_copy.drop(columns=['NSujet', 'PatientID', 'StudyInstanceUID', 'Stage',  'recurrence', 'VitalStatus', 'PFS-months', 'PFS-days'], inplace=True)

#Remove rows with missing data
filtered_data_copy.dropna(inplace=True)
                                            
features = filtered_data_copy.drop(columns=['OS-months', 'OS-days'])
print("number of features = ", np.shape(features))

# Set the target variable as recurrence for stage 1 and stage 2
target = filtered_data_copy['OS-months']
#event = filtered_data_copy['VitalStatus']
print("number of features = ", np.shape(features))

#Remove constant radiomic features
#constant_features = features.columns[features.nunique() == 1]
#features.drop(constant_features, axis=1, inplace=True)
#print("number of non-constant features = ", np.shape(features))
# Store the feature names
feature_names = features.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test= train_test_split(features, target, test_size=0.3, random_state=42)
# Define the list of columns to drop
columns_clinical_to_drop = ['Smoking', 'Age', 'Subtype', 'Sex']
# Drop the clinical data from X_train_scaled and X_test_scaled
X_train_filtered = X_train.drop(columns=columns_to_drop)
X_test_filtered = X_test.drop(columns=columns_to_drop)

#Feature selection based on training set
#remove all features that are constant 
X_train_filtered = X_train_filtered.loc[:, X_train_filtered.var() != 0.0]

print("Shape of the training features after removing constant features =", np.shape(X_train_filtered))

#get correlations of each features in dataset and remove one of each highly correlated to each other 
corr_matrix = X_train_filtered.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find name of feature columns with correlation greater than 0.9 which may be dropped
column_to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

#drop the desired fetures among the ones with high correlation 

X_train_filtered.drop(labels = column_to_drop, axis=1, inplace=True)
X_test_filtered = X_test_filtered[X1_train_filtered.columns]

print("Shape of the training features after dropping high correlated features =", np.shape(X1_train_filtered))

scaler = preprocessing.StandardScaler().fit(X_train_filtered)
X_train_scaled = scaler.transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# Extract the columns to be dropped from X_train_scaled and X_test_scaled
X_train_clinical = X_train[columns_clinical_to_drop]
X_test_clinical = X_test[columns_clinical_to_drop]

scaler = preprocessing.StandardScaler().fit(X_train_clinical)
X_train_clinical_scaled = scaler.transform(X_train_clinical)
X_test_clinical_scaled = scaler.transform(X_test_clinical)


y_train = y_train.values
y_test = y_test.values

X_train_clinical_df = pd.DataFrame(X_train_clinical_scaled)
X_test_clinical_df = pd.DataFrame(X_test_clinical_scaled)
# Save the clinical data for training and test sets to CSV files
X_train_clinical_df.to_csv(os.path.join(main_dir,'data/T-train_data_os_st1-2_clinical_corr.csv'), index=False, float_format='%.7f')
X_test_clinical_df.to_csv(os.path.join(main_dir,'data/T-test_data_os_st1-2_clinical_corr.csv'), index=False, float_format='%.7f')




# Define the cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=3)
def my_scorer(y_test, y_predicted):
    error = concordance_index(y_test,y_predicted)
    return error
my_func = make_scorer(my_scorer, greater_is_better=True)
# Perform Lasso feature selection
lasso_model= Lasso(max_iter=10000)
param_grid = {'alpha': [i * 0.5 for i in range(1, 100)]}
grid_search = GridSearchCV(lasso_model, param_grid, cv=cv, scoring=my_func)
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
lasso_model_best = Lasso(alpha=best_alpha)
lasso_model_best.fit(X_train_scaled, y_train)

coefficients= lasso_model_best.coef_
zero_coefficient_indices = list(np.where(coefficients == 0)[0])
X_train_selected = X_train_scaled[:, ~np.isin(np.arange(X_train_scaled.shape[1]), zero_coefficient_indices)]
X_test_selected = X_test_scaled[:, ~np.isin(np.arange(X_test_scaled.shape[1]), zero_coefficient_indices)]
# Get the indices of the selected features after Lasso-based feature selection
selected_feature_indices_after_lasso = np.where(~np.isin(np.arange(X_train_scaled.shape[1]), zero_coefficient_indices))[0]
# Get the names of the features after Lasso-based feature selection
selected_feature_names_after_lasso = [feature_names[idx] for idx in selected_feature_indices_after_lasso]
# Combine the selected features with clinical column names
final_selected_features = selected_feature_names_after_lasso 
print("shape of X_test after Lasso = ", np.shape(X_test_selected))
print("shape of X_train after Lasso = ", np.shape(X_train_selected))
print("shape of y_test after Lasso = ", np.shape(y_test))      
print("No. of features after Lasso = " , len(final_selected_features)) 
#print(len(zero_coefficient_indices))

X_train_df = pd.DataFrame(X_train_selected)
y_train_df = pd.DataFrame({'y_test': y_train})

X_test_df = pd.DataFrame(X_test_selected)
y_test_df = pd.DataFrame({'y_test': y_test})

# Concatenate the DataFrames horizontally (side by side)
combined_df_train = pd.concat([X_train_df, y_train_df], axis=1)
combined_df_test = pd.concat([X_test_df, y_test_df], axis=1)

# Save the combined DataFrame to a CSV file
combined_df_train.to_csv(os.path.join(main_dir,'data/T-train_data_os_st1-2_rad_corr.csv'), index=False, float_format='%.7f')
combined_df_test.to_csv(os.path.join(main_dir,'data/T-test_data_os_st1-2_rad_corr.csv'), index=False, float_format='%.7f')