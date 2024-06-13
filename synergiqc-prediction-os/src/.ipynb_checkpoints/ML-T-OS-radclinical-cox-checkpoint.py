import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, RepeatedStratifiedKFold,  RandomizedSearchCV
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
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
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
from lifelines import CoxPHFitter

from lifelines.datasets import load_rossi

rossi = load_rossi()
# get the start time
st_initial = time.time()

#specify the file name so that the results for different models will be saved in subfolders of a main folder with the following name
file_name = 'T-OS_st1_2_all_rad_clinical_corr_log_harmonized_cox_opposite'

#specify the path of main folder 
main_dir= "/home/ulaval.ca/lesee/projects/Project2-synergiqc/OS/"

# Check if the results folder and its subfolder with the "file_name" exists, and create it if it doesn't
if not os.path.exists(os.path.join(main_dir, 'results')):
    os.makedirs(os.path.join(main_dir, 'results'))
if not os.path.exists(os.path.join(main_dir, 'results/'+file_name)):
    os.makedirs(os.path.join(main_dir, 'results/'+file_name))

#load the training and test dataset that have been created by dataset_feature_type.py code
df_training_data = pd.read_csv(os.path.join(main_dir,'data/T-train_data_os_st1_2_rad_corr_log_harmonized.csv'))
df_test_data = pd.read_csv(os.path.join(main_dir,'data/T-test_data_os_st1_2_rad_corr_log_harmonized.csv'))

#load the clinical data for the same training and test dataset that have been created by dataset_feature_type.py code
df_training_data_clinical = pd.read_csv(os.path.join(main_dir,'data/T-train_data_os_st1_2_clinical_corr_log_harmonized.csv'))
df_test_data_clinical = pd.read_csv(os.path.join(main_dir,'data/T-test_data_os_st1_2_clinical_corr_log_harmonized.csv'))

X_train_selected = df_training_data.iloc[:, :-1].values  # Select all columns except the last one (training features)
y_train = df_training_data.iloc[:, -1].values # Select only the last column 

X_test_selected = df_test_data.iloc[:, :-1].values  # Select all columns except the last one (test features)
y_test = df_test_data.iloc[:, -1].values # Select only the last column

X_train_clinical = df_training_data_clinical.iloc[:, :-1].values  # Select all columns except the last one (training clinical data)
X_test_clinical= df_test_data_clinical.iloc[:, :-1].values  # Select all columns except the last one (test clinical data)

event_train = df_training_data_clinical.iloc[:, -1].values #Select only the last column
opposite_event_train = np.logical_not(event_train).astype(int)
event_test = df_test_data_clinical.iloc[:, -1].values #Select only the last column
opposite_event_test = np.logical_not(event_test).astype(int)

X_train_selected_df = pd.DataFrame(X_train_selected)  # Replace with actual feature names
X_test_selected_df = pd.DataFrame(X_test_selected)
y_train_df = pd.DataFrame(y_train, columns=['OS-train'])
event_train_df = pd.DataFrame(opposite_event_train, columns=['vital_status_train'])

# Rename the columns in X_train_clinical_df and X_test_clinical_df
clinical_columns = ['Smoking', 'Age', 'Subtype', 'Sex']

# Create a DataFrame with the array and the new column names
X_train_clinical_df = pd.DataFrame(X_train_clinical, columns=clinical_columns)
X_test_clinical_df = pd.DataFrame(X_test_clinical, columns=clinical_columns)


"""
# Define the hyperparameter grids for Adaboost model
hyperparameter_grids = {
    'AdaBoost': {
        'n_estimators': [50, 100, 200],  # Number of weak learners
        'learning_rate': [0.01, 0.1, 0.2],  # Contribution of each weak learner
        'loss': ['linear', 'square', 'exponential'],  # Loss function to use for updating weights
        # Additional AdaBoostRegressor-specific parameters can be added here
    }
}
"""

# Define the hyperparameter grids for XGBoost model
"""
hyperparameter_grids = {
    
    'XGBoost': {
    'n_estimators': [50, 100, 150],           # Reduced number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.15],       # Adjusted learning rate
    'max_depth': [1, 3],                   # Reduced maximum depth
    'min_child_weight': [1, 2],               # Adjusted minimum child weight
    'subsample': [0.7, 0.9],             # Reduced subsample
    'colsample_bytree': [0.7, 0.8, 0.9],      # Reduced colsample_bytree
    'gamma': [0, 0.05, 0.1],                  # Adjusted gamma
    'reg_alpha': [0, 0.05, 0.1],              # Adjusted reg_alpha
    'reg_lambda': [0, 0.05, 0.1]              # Adjusted reg_lambda
}

}

"""
# Define the hyperparameter grids for each model
hyperparameter_grids = {
    'SVM': { 'kernel': ['linear', 'rbf'],
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.2, 0.3, 0.4]},
    'Ridge': { 'alpha': [0.01, 0.1, 1.0, 10.0]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10],  'min_samples_split': [2, 5],'min_samples_leaf': [1, 2],  'bootstrap': [True, False]},
    'NeuronalNetwork': {'hidden_layer_sizes': [(100,), (150,), (300,)],'activation': ['relu', 'tanh'],'alpha': [0.0001, 0.001, 0.01],'learning_rate':['constant','adaptive'], 'random_state': [0, 5, 10], 'solver': ['sgd']},
   'GradientBoosting': { 'n_estimators': [100, 200],'learning_rate': [0.01, 0.1],'max_depth': [3, 4],'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
        'DecisionTree' : {    'max_depth': [None, 5, 10],  'min_samples_split': [2, 5, 10],  'min_samples_leaf': [1, 2, 4]},
    'AdaBoost': {
        'n_estimators': [50, 100, 200],  # Number of weak learners
        'learning_rate': [0.01, 0.1, 0.2],  # Contribution of each weak learner
        'loss': ['linear', 'square', 'exponential']  # Loss function to use for updating weights
        # Additional AdaBoostRegressor-specific parameters can be added here
    }
}


#This function is not used in this code
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, event_train):
    """
    Train the given model on the training data and evaluate its performance on the test data.

    Parameters:
    - model: The machine learning model to train and evaluate.
    - X_train: The feature matrix of the training data.
    - y_train: The target values of the training data.
    - X_test: The feature matrix of the test data.
    - y_test: The target values of the test data.
    - event_train: (Assumed to be an event indicator for survival analysis)

    Returns:
    - c_index: Concordance index for the predictions.
    - mse: Mean squared error of the predictions.
    - rmse: Root mean squared error of the predictions.
    - mae: Mean absolute error of the predictions.
    - r2: R-squared score of the predictions.
    - y_pred: Predictions made by the model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    c_index = concordance_index(y_test, y_pred)     
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return c_index, mse, rmse, mae, r2, y_pred

#This function is not used in this code
def X_test_after_feature_selection(X, y, method, n_features):
    """
    Perform feature selection on the input data using the specified method and number of features.

    Parameters:
    - X: The feature matrix.
    - y: The target values.
    - method: The feature selection method ('mutual_info', 'reliefF', 'surf', 'multisurf', 'f_test').
    - n_features: The number of features to select.

    Returns:
    - X_test_new: The feature matrix after feature selection.
    """    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
    elif method == 'reliefF':
        selector = ReliefF(n_neighbors=100, n_features_to_select=n_features)
    elif method == 'surf':
        selector = SURF(n_features_to_select=n_features)
    elif method == 'multisurf':
        selector = MultiSURF(n_features_to_select=n_features)
    elif method == 'f_test':
        selector = SelectKBest(score_func=f_regression, k=n_features)
    else:
        raise ValueError('Invalid feature selection method.')
    selector.fit(X, y)
#    selector.fit(X_train, y)
    X_test_new = selector.transform(X)
    return X_test_new
#This function is used in this code
def select_features(method, X_train, y_train, X_test, n):
    """
    Perform feature selection on the input data using the specified method and number of features.

    Parameters:
    - method: The feature selection method ('mutual_info', 'reliefF', 'surf', 'multisurf', 'f_test').
    - X_train: The feature matrix of the training data.
    - y_train: The target values of the training data.
    - X_test: The feature matrix of the test data.
    - n: The number of features to select.

    Returns:
    - X_train_new: The feature matrix of the training data after feature selection.
    - X_test_new: The feature matrix of the test data after feature selection.
    - selector: The trained feature selector object.
    """      
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=n)
    elif method == 'reliefF':
        selector = ReliefF(n_neighbors=100, n_features_to_select=n)
    elif method == 'surf':
        selector = SURF(n_features_to_select=n)
    elif method == 'multisurf':
        selector = MultiSURF(n_features_to_select=n)
    elif method == 'f_test':
        selector = SelectKBest(score_func=f_regression, k=n)
    else:
        raise ValueError('Invalid feature selection method.')

    selector.fit(X_train, y_train)
    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)
#    if method == 'mutual_info' or method == 'f_test':
#        selected_feature_indices = selector.get_support(indices=True)
#    else:
    # Get the feature importance scores
#        feature_importances = selector.feature_importances_

# Sort the features by importance scores and get the indices of the top n features
#        selected_feature_indices = np.argsort(feature_importances)[-n:]
    return X_train_new, X_test_new,  selector

#This function is used in this code
def my_scorer(y_test, y_predicted):
    """
    Custom scoring function for model evaluation.

    Parameters:
    - y_test: The true target values.
    - y_predicted: The predicted values.

    Returns:
    - error: Concordance index for the predictions.
    """
    error = concordance_index(y_test,y_predicted)
    return error

#This function is used in this code
def predict_with_model(X_test, best_model):
    """
    Use the best_model to make predictions on the given feature matrix.

    Parameters:
    - X_test: The feature matrix for making predictions.
    - best_model: The trained machine learning model.

    Returns:
    - model_prediction: Predictions made by the model.
    """    
    # Use the best_model to make predictions
    model_prediction = best_model.predict(X_test)
    return model_prediction


#This function is used in this code

def cox_select_features(X_train, y_train, X_test, n, event_train):
    """
    Perform feature selection using Cox Proportional Hazard model based on p-values.

    Parameters:
    - X_train: The feature matrix of the training data.
    - y_train: Overall Survival (time to event or censoring) for training data.
    - X_test: The feature matrix of the test data.
    - n: The number of features to select.
    - event_train: The binary column indicating whether the event (death) has occurred for training data.


    Returns:
    - X_train_new: The feature matrix of the training data after feature selection.
    - X_test_new: The feature matrix of the test data after feature selection.
    - cox_model: Fitted Cox Proportional Hazard model.
    """
    cox_model = CoxPHFitter()
    cox_model.fit(X_train, duration_col=y_train, event_col=event_train)

    # Get the summary of the fitted model, including p-values
    cox_summary = cox_model.summary

    # Sort features based on p-values
    sorted_features = cox_summary.sort_values(by='p', ascending=True)

    # Select the top n features
    selected_feature_indices = sorted_features.head(n).index.tolist()

    # Get the new feature matrices
    X_train_new = X_train[selected_feature_indices]
    X_test_new = X_test[selected_feature_indices]

    return X_train_new, X_test_new, cox_model

# Create a custom scoring function (my_func) using make_scorer,
# based on the my_scorer function, with greater_is_better set to True.

my_func = make_scorer(my_scorer, greater_is_better=True)

# Define ML models

models = {
    'SVM': SVR(),
    'Ridge' : Ridge(),
    'RandomForest': RandomForestRegressor(),
    'NeuronalNetwork': MLPRegressor(max_iter=100000, early_stopping=True),
    'GradientBoosting': GradientBoostingRegressor(),
    'DecisionTree' : DecisionTreeRegressor()
#    'XGBoost': XGBRegressor(tree_method="hist", n_jobs = -1)
    ,'AdaBoost': AdaBoostRegressor()
}

                            
# Define the cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, random_state=42, n_repeats=3) 

#define the minimum number of features selected by FS methods
n_features_initial = 3

#define the maximum number of features selected by FS methods
n_features_final=min(50,np.shape(X_train_selected)[1])
#n_features_final=np.shape(X_train_selected)[1]

#list of umber of features selected by FS methods
iteration_features = list(range(n_features_initial, n_features_final + 1))

# Create an empty DataFrame to store the results
results_df = pd.DataFrame()

# Outermost loop for models
for model_name, model in models.items():
    st = time.time()
    print("ML Model = ", model_name)
    
    # Perform GridSearchCV
    param_grid = hyperparameter_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, scoring=my_func, refit=True, cv=5, n_jobs=-1) 
#    grid_search = RandomizedSearchCV(model, param_grid,scoring=my_func, refit=True, cv=5, n_jobs=-1) 
    
    #initialize lists to store data
    n_selected_features = []
#    method_names = []
    c_index_values_disovery =[]
    c_index_values_validation = []
    confidence_interval_values_discovery = []
    confidence_interval_values_validation = []
    
    # Create an empty list to store maximum score dataframes for each n_features
    max_scores_dfs = []
    # Create an empty DataFrame to store the average scores over folds for each n_features and Grid Config
    averages_df = pd.DataFrame(columns=['n_features', 'Grid Configuration', 'Average Mean Score-training'])

    for index, n_features in enumerate(iteration_features):
        # Create a list to store the mean test scores for each grid configuration
        grid_scores = []
        grid_scores_c_index = []
        print("Number of features = ", n_features)   

        # Loop for cross-validation folds
        for i, (train_idx, val_idx) in enumerate(cv.split(X_train_selected)):
            print ("Fol No. ", i+1 )
            
            X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            event_train_fold, event_val_fold = opposite_event_train[train_idx],opposite_event_train[val_idx]
            X_train_fold_df = pd.DataFrame(X_train_fold)  # Replace with actual feature names
            X_val_fold_df = pd.DataFrame(X_val_fold)  # Replace with actual feature names            
            y_train_fold_df = pd.DataFrame(y_train_fold, columns=['OS-train'])
            event_train_fold_df = pd.DataFrame(event_train_fold, columns=['vital_status_train'])


            # Merge X_train_fold_df, y_train_fold_df, and event_train_fold_df
            merged_df = pd.concat([X_train_fold_df, y_train_fold_df, event_train_fold_df], axis=1)



            

            # Perform feature selection on the fold using the current method           
#                X_train_fold_new, X_val_fold_new, selector = select_features(
#                        method, X_train_fold, y_train_fold, X_val_fold, n_features)



            cox_model = CoxPHFitter(penalizer=0.1)  # Adjust the penalizer value as needed
            cox_model.fit(merged_df, duration_col='OS-train', event_col='vital_status_train')

            # Get the summary of the fitted model, including p-values
            cox_summary = cox_model.summary

            # Sort features based on p-values
            sorted_features = cox_summary.sort_values(by='p', ascending=True)

            # Select the top n features
            selected_feature_indices = sorted_features.head(n_features).index.tolist()
#            print(selected_feature_indices)

            # Get the new feature matrices
            X_train_fold_new = X_train_fold_df[selected_feature_indices]
#            print(np.shape((X_train_fold_new)))
#            print(len(y_train_fold))
            X_val_fold_new = X_val_fold_df[selected_feature_indices]
#            selector = cox_model
    

            # Perform grid search cross-validation for hyperparameter tuning
            grid_search.fit(X_train_fold_new, y_train_fold)

            # Access the results for each grid configuration
            results= pd.DataFrame(grid_search.cv_results_)

            # Add columns for n_features and n_fold
            results['n_features'] = n_features
            results['n_fold'] = i+1

            # Concatenate the current results with the cumulative results DataFrame             
            results_df = pd.concat([results_df, results], ignore_index=True)

            # Store the mean test scores for the current fold
            grid_scores.extend(results['mean_test_score'])

        # Calculate the average mean_test_score for each grid configuration
        grid_avg_scores = []
        for grid_idx in range(len(results)):
            grid_avg_score = np.mean(grid_scores[grid_idx::len(results)])
            grid_avg_scores.append(grid_avg_score)

        # Create a DataFrame for the average scores of each grid configuration
        avg_scores_df = pd.DataFrame({
            'n_features': [n_features] * len(grid_avg_scores),
            'Grid Configuration': grid_search.cv_results_['params'],
            'Average Mean Score-training': grid_avg_scores
        })

        # Concatenate the current averages with the cumulative averages DataFrame
        averages_df = pd.concat([averages_df, avg_scores_df], ignore_index=True)

        # Find the maximum 'Average Mean Score-training' for each feature
        max_c_index_scores_df = avg_scores_df.groupby('n_features')['Average Mean Score-training'].max().reset_index()

        # Merge max_c_index_scores_df with avg_scores_df to get the corresponding 'Grid Configuration'
        max_scores_with_config = pd.merge(avg_scores_df, max_c_index_scores_df, on=['n_features','Average Mean Score-training'], suffixes=('', '_max'))
        # Rename columns for clarity
        max_scores_with_config = max_scores_with_config.rename(columns={'Grid Configuration': 'Max Grid Configuration'})                      
        # Append the max_scores_with_config for this n_features to the list
        max_scores_dfs.append(max_scores_with_config)

    # Concatenate all the dataframes into a single dataframe
    final_max_scores_df = pd.concat(max_scores_dfs, ignore_index=True)

    # Check if the subfolder with the "model_name" exists, and create it if it doesn't
    if not os.path.exists(os.path.join(main_dir, 'results/'+file_name+'/'+model_name)):
        os.makedirs(os.path.join(main_dir, 'results/'+file_name+'/'+model_name))   

#        # Save the grid search DataFrame to a CSV file
#        results_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name +'_gridsearch_' + method+ '.csv'), index=False,  float_format='%.7f')

    # Save the averages DataFrame to a CSV file        
#        averages_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name+ '_average_scores__with_config_' + method+ '.csv'), index=False,  float_format='%.7f')       

    # Save the maximum scores Daraframe to a CSV file in a folder with model's name             
    final_max_scores_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name + '_max_scores_with_config_'+ '.csv'), index=False, float_format='%.7f')      

    # Find the row with the maximum average c-index within the current final_max_scores_df
    max_row = final_max_scores_df.iloc[final_max_scores_df['Average Mean Score-training'].idxmax()]

    # Extract the desired information from the row
    n_features_max = max_row['n_features']
    max_grid_config = max_row['Max Grid Configuration']
    max_average_mean_score = max_row['Average Mean Score-training']

    # Perform feature selection for the corrent FS method using the bext number of features           
#    X_train_selected_final, X_test_selected_final, selector = cox_select_features(X_train_selected, y_train, X_test_selected, n_features_max, event_train_df)



    # Merge X_train_fold_df, y_train_fold_df, and event_train_fold_df
    merged_df_train = pd.concat([X_train_selected_df, y_train_df, event_train_df], axis=1)

    cox_model_train = CoxPHFitter(penalizer=0.1)  # Adjust the penalizer value as needed
    cox_model_train.fit(merged_df_train, duration_col='OS-train', event_col='vital_status_train')

    # Get the summary of the fitted model, including p-values
    cox_summary_train = cox_model_train.summary

    # Sort features based on p-values
    sorted_features_train = cox_summary_train.sort_values(by='p', ascending=True)

    # Select the top n features
    selected_feature_indices_train = sorted_features_train.head(n_features_max).index.tolist()
#            print(selected_feature_indices)

    # Get the new feature matrices
    X_train_selected_final = X_train_selected_df[selected_feature_indices_train]
#            print(np.shape((X_train_fold_new)))
#            print(len(y_train_fold))
    X_test_selected_final = X_test_selected_df[selected_feature_indices_train]
    
    #set the model with best performing hyperparameters
    model_selected = model.set_params(**max_grid_config)        
#**********************************************************************************
    # make X_train and X_test as dataframes
    X_train_selected_final_df = pd.DataFrame(X_train_selected_final) 
    X_test_selected_final_df = pd.DataFrame(X_test_selected_final)

    # Standardize the "Age" column using the same scaler used for other continuous features
    scaler = StandardScaler()
    X_train_clinical_df['Age'] = scaler.fit_transform( X_train_clinical_df['Age'].values.reshape(-1, 1))
    X_test_clinical_df['Age'] = scaler.fit_transform( X_test_clinical_df['Age'].values.reshape(-1, 1))

    # Perform one-hot encoding for the categorical features in the clinical data
    X_train_clinical_categorical_df_encoded = pd.get_dummies(X_train_clinical_df, columns=['Smoking', 'Subtype', 'Sex'])
    X_test_clinical_categorical_df_encoded = pd.get_dummies(X_test_clinical_df, columns=['Smoking', 'Subtype', 'Sex'])                 
    # Concatenate the one-hot encoded categorical features with the continuous feature (Age)
    X_train_clinical_final_df = X_train_clinical_categorical_df_encoded
    X_test_clinical_final_df = X_test_clinical_categorical_df_encoded 

#        print(X_train_clinical_final_df)

  # Concatenate the radiomics features with clinical features   
    X_train_final = pd.concat([X_train_selected_final_df, X_train_clinical_final_df], axis=1)
    X_test_final = pd.concat([X_test_selected_final_df, X_test_clinical_final_df], axis=1) 

    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    # Train the model on the entire training set
    model_selected.fit(X_train_final, y_train)

    # Use the best_model to make predictions on the test dataset
    validation_prediction = predict_with_model(X_test_final, model_selected)

    # Calculate the average c-index using cross_val_score
    cindex_values_discovery = cross_val_score(model_selected, X_train_final, y_train, cv=cv, scoring=my_func)

    # Perform bootstrapping to estimate the 95% CI
    n_bootstrap = 1000 

    bootstrapped_cindex_values_discovery = []
    bootstrapped_cindex_values_validation = []

    for _ in range(n_bootstrap):
        num_samples = len(y_test)
        resampled_indices_validation = np.random.choice(num_samples, size=num_samples, replace=True)
        resampled_y_test = y_test[resampled_indices_validation]
        # Resample with replacement from the c-index values
        resampled_cindices = np.random.choice(cindex_values_discovery, size=len(cindex_values_discovery), replace=True)
        bootstrapped_cindex_discovery = np.mean(resampled_cindices)
        bootstrapped_cindex_values_discovery.append(bootstrapped_cindex_discovery)

        # Resample with replacement from the test dataset predictions
        resampled_predictions = validation_prediction[resampled_indices_validation]

        # Calculate the c-index for the resampled predictions
        bootstrapped_cindex_validation = concordance_index(resampled_y_test, resampled_predictions)
        bootstrapped_cindex_values_validation.append(bootstrapped_cindex_validation)

    # Calculate the average of c-index for the CV on training dataset            
    c_index_discovery = np.mean(cindex_values_discovery)
    # Calculate the 95% confidence interval
    confidence_interval_discovery = np.percentile(bootstrapped_cindex_values_discovery, [2.5, 97.5])           

    # Calculate the c-index for the predictions on the test dataset
    c_index_validation = concordance_index(y_test, validation_prediction)
    # Calculate the 95% confidence interval
    confidence_interval_validation = np.percentile(bootstrapped_cindex_values_validation, [2.5, 97.5])
#**********************************************************
    # Append the c-index value to the list
    c_index_values_disovery.append(c_index_discovery)
    c_index_values_validation.append(c_index_validation)
    confidence_interval_values_discovery.append(confidence_interval_discovery)
    confidence_interval_values_validation.append(confidence_interval_validation)

#    method_names.append(method)
    n_selected_features.append(n_features_max)
    # Save the trained model to a file
    # Path to the file
    best_trained_model_path = os.path.join(main_dir,'results/'+file_name+'/'+model_name + '/'+ model_name + '_'+ '_n_f_'+str(n_features_max)+'_best_model.pkl')

    # Save the best trained model
    with open(best_trained_model_path, 'wb') as pickle_file:
        joblib.dump(model_selected, pickle_file)        
    data = {
        'n_features': n_selected_features, 
        'c_index_discovery': c_index_values_disovery,  
        'c_index_discovery_CI95' : confidence_interval_values_discovery,
        'c_index_validation': c_index_values_validation,
        'c_index_validation_CI95' : confidence_interval_values_validation        
        
    }        

# Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
    df.to_csv(os.path.join(main_dir,'results/'+file_name+'/'+model_name + '/'+ model_name + '_final_results.csv'), index=False, float_format='%.6f')               

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time_model = et - st
    print('Execution time for model '+model_name+ ' =', elapsed_time_model, 'seconds')

# get the end time
et_final = time.time()

# get the execution time
elapsed_time = et_final - st_initial
print('Total Execution time =', elapsed_time, 'seconds')

