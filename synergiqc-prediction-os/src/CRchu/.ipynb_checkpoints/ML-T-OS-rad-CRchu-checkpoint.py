import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
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
st_initial = time.time()

file_name = 'T-OS_st1_2_rad'
#Load the data
main_dir= "/home/ulaval.ca/lesee/projects/Project2-synergiqc/OS/"

# Check if the folder exists, and create it if it doesn't
if not os.path.exists(os.path.join(main_dir, 'results')):
    os.makedirs(os.path.join(main_dir, 'results'))
if not os.path.exists(os.path.join(main_dir, 'results/'+file_name)):
    os.makedirs(os.path.join(main_dir, 'results/'+file_name))

df_training_data = pd.read_csv(os.path.join(main_dir,'data/T-train_data_os_st1_2_rad.csv'))
df_test_data = pd.read_csv(os.path.join(main_dir,'data/T-test_data_os_st1_2_rad.csv'))

X_train_selected = df_training_data.iloc[:, :-1].values  # Select all columns except the last one
y_train = df_training_data.iloc[:, -1].values # Select only the last column

X_test_selected = df_test_data.iloc[:, :-1].values  # Select all columns except the last one
y_test = df_test_data.iloc[:, -1].values # Select only the last column




# Define the hyperparameter grids for each model
hyperparameter_grids = {
    'SVM': { 'kernel': ['linear', 'rbf'],
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.2, 0.3, 0.4]},
    'Ridge': { 'alpha': [0.01, 0.1, 1.0, 10.0]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10],  'min_samples_split': [2, 5],'min_samples_leaf': [1, 2],  'bootstrap': [True, False]},
    'NeuronalNetwork': {'hidden_layer_sizes': [(100,), (150,), (300,)],'activation': ['relu', 'tanh'],'alpha': [0.0001, 0.001, 0.01],'learning_rate': ['constant','adaptive'], 'random_state': [0, 5, 10], 'solver': ['sgd']},
#       'GaussianProcesses' : {
#    'kernel': [RBF(length_scale=1.0), Matern(length_scale=1.0), RationalQuadratic(length_scale=1.0)],
#    'alpha': [1e-10, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 20.0],
#    'n_restarts_optimizer': [0, 1, 5],
#    'random_state': [0]},
        'GradientBoosting': { 'n_estimators': [100, 200],'learning_rate': [0.01, 0.1],'max_depth': [3, 4],'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
        'DecisionTree' : {    'max_depth': [None, 5, 10],  'min_samples_split': [2, 5, 10],  'min_samples_leaf': [1, 2, 4]}
}
def select_features(method, X_train, y_train, X_test, n):
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
#    selector.fit(X_train, y)
    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)
    if method == 'mutual_info' or method == 'f_test':
        selected_feature_indices = selector.get_support(indices=True)
    else:
    # Get the feature importance scores
        feature_importances = selector.feature_importances_

# Sort the features by importance scores and get the indices of the top n features
        selected_feature_indices = np.argsort(feature_importances)[-n:]
    return X_train_new, X_test_new,  selector

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, event_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    c_index = concordance_index(y_test, y_pred)     
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return c_index, mse, rmse, mae, r2, y_pred

def my_scorer(y_test, y_predicted):
    error = concordance_index(y_test,y_predicted)
    return error

my_func = make_scorer(my_scorer, greater_is_better=True)

def X_test_after_feature_selection(X, y, method, n_features):
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

def predict_with_model(X_test, best_model):
    # Use the best_model to make predictions
    model_prediction = best_model.predict(X_test)
    return model_prediction
# Define models
models = {
    'SVM': SVR(),
    'Ridge' : Ridge(),
     'RandomForest': RandomForestRegressor(),
    'NeuronalNetwork': MLPRegressor(max_iter=100000, early_stopping=True),
#    'GaussianProcesses': GaussianProcessRegressor( n_restarts_optimizer=10),
    'GradientBoosting': GradientBoostingRegressor(),
    'DecisionTree' : DecisionTreeRegressor()    
}
# Define feature selection methods and the maximum number of features to consider
feature_selection_methods = ['f_test'
                             , 'mutual_info'
                             , 'reliefF'
                             , 'surf'
                             , 'multisurf'
                            ]
# Initialize dictionaries to store results
#models_list = ['SVM']
#best_results = {}
cv_scores_folds = np.array([])

# Define the cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, random_state=42, n_repeats=3)  

n_features_initial = 3
n_features_final=min(50,np.shape(X_train_selected)[1])
#n_features_final=4
#print(n_features_final)
iteration_features = list(range(n_features_initial, n_features_final + 1))
# Initialize a list to store c-index values
# Create an empty DataFrame to store the results
results_df = pd.DataFrame()

# Outermost loop for models
for model_name, model in models.items():
    st = time.time()
    print("ML Model = ", model_name)
    param_grid = hyperparameter_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, scoring=my_func, refit=True, cv=5, n_jobs=-1)         
    c_index = []
    n_selected_features = []
    selected_model = []
    method_names = []
    c_index_values_disovery =[]
    c_index_values_validation = []
    confidence_interval_values_discovery = []
    confidence_interval_values_validation = []
    # Loop for feature selection methods
    for method in feature_selection_methods:
        print("Method = ", method)

        # Initialize a list to store cross-validation scores for each fold
        # Initialize the best_results dictionary
  
        c_index_n_features_folds_total = []
        best_model_n_features_folds_total = []
        c_indices_n_features = []
        # Create an empty list to store dataframes for each n_features
        max_scores_dfs = []
        # Create an empty DataFrame to store the averages
        averages_df = pd.DataFrame(columns=['n_features', 'Grid Configuration', 'Average Mean Score-training'])
       
        for index, n_features in enumerate(iteration_features):
            # Create a list to store the mean test scores for each grid configuration
            grid_scores = []
            grid_scores_c_index = []
            print("Number of features = ", n_features)   
            c_index_n_features = []
            best_model_n_features = [] 
            c_indices_folds = []
            c_indices = []
#            fold_data = []
                    # Loop for cross-validation folds
            for i, (train_idx, val_idx) in enumerate(cv.split(X_train_selected)):
                print ("Fol No. ", i+1 )
                X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                # Loop for the number of features
                # Store the fold data in a dictionary

                # Perform feature selection on the fold using the current method           
                X_train_fold_new, X_val_fold_new, selector = select_features(
                        method, X_train_fold, y_train_fold, X_val_fold, n_features)
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
            # Calculate the average mean_test_score and c-index for each grid configuration
            grid_avg_scores = []
#            grid_avg_c_index_scores = []
            for grid_idx in range(len(results)):
                grid_avg_score = np.mean(grid_scores[grid_idx::len(results)])
#                grid_avg_c_index_score =  np.mean(grid_scores_c_index[grid_idx::len(results)])
                grid_avg_scores.append(grid_avg_score)
#                grid_avg_c_index_scores.append(grid_avg_c_index_score)
            # Create a DataFrame for the average scores of each grid configuration
            avg_scores_df = pd.DataFrame({
                'n_features': [n_features] * len(grid_avg_scores),
                'Grid Configuration': grid_search.cv_results_['params'],
                'Average Mean Score-training': grid_avg_scores
#                ,'Average c_index_folds': grid_avg_c_index_scores
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
        # Save the result to a CSV file
        
        if not os.path.exists(os.path.join(main_dir, 'results/'+file_name+'/'+model_name)):
            os.makedirs(os.path.join(main_dir, 'results/'+file_name+'/'+model_name))        
        
        final_max_scores_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name + '_max_scores_with_config_'+ method+ '.csv'), index=False, float_format='%.7f')            

        # Save the DataFrame to a CSV file
        results_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name +'_gridsearch_' + method+ '.csv'), index=False,  float_format='%.7f')
        # Save the averages DataFrame to a CSV file
        
        averages_df.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name +'/'+ model_name+ '_average_scores__with_config_' + method+ '.csv'), index=False,  float_format='%.7f')       
   
        # Find the row with the maximum average c-index within the current final_max_scores_df
        max_row = final_max_scores_df.iloc[final_max_scores_df['Average Mean Score-training'].idxmax()]
        # Extract the desired information from the row
        n_features_max = max_row['n_features']
        max_grid_config = max_row['Max Grid Configuration']
        max_average_mean_score = max_row['Average Mean Score-training']
#        max_c_index = max_row['Average c_index_folds']
        X_train_selected_final, X_test_selected_final, selector = select_features(
            method, X_train_selected, y_train, X_test_selected, n_features_max)        
        model_selected = model.set_params(**max_grid_config)        

        # Define the cross-validation strategy
        
        # Train the model on the entire training set
        model_selected.fit(X_train_selected_final, y_train)
        # Use the best_model to make predictions on the test dataset
        validation_prediction = predict_with_model(X_test_selected_final, model_selected)

        # Calculate the average c-index using cross_val_score
        cindex_values_discovery = cross_val_score(model_selected, X_train_selected_final, y_train, cv=cv, scoring=my_func)
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
        # Perform cross-validation and calculate the average c-index on the training dataset        

        # Append the c-index value to the list
        c_index_values_disovery.append(c_index_discovery)
        c_index_values_validation.append(c_index_validation)
        confidence_interval_values_discovery.append(confidence_interval_discovery)
        confidence_interval_values_validation.append(confidence_interval_validation)
        
        method_names.append(method)
        n_selected_features.append(n_features_max)
# Save the trained model to a file
# Path to the file
        best_trained_model_path = os.path.join(main_dir,'results/'+file_name+'/'+model_name + '/'+ model_name + '_'+ method +'_n_f_'+str(n_features_max)+'_best_model.pkl')

        # Save the best trained model
        with open(best_trained_model_path, 'wb') as pickle_file:
            joblib.dump(model_selected, pickle_file)        
    data = {
        'Method': method_names,
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
    elapsed_time = et - st
    print('Execution time for model '+model_name+ ' =', elapsed_time, 'seconds')

# get the end time
et_final = time.time()

# get the execution time
elapsed_time = et_final - st_initial
print('Total Execution time =', elapsed_time, 'seconds')