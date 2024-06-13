import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from skrebate import ReliefF, SURF, MultiSURF
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (DotProduct, WhiteKernel, RBF, Matern, ConstantKernel, ExpSineSquared, RationalQuadratic, Product)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from imblearn.under_sampling import RandomUnderSampler
#from skopt import BayesSearchCV
import os
import json
import time
import joblib
# get the start time
st = time.time()
file_name = 'recurrence_st-all_rad_combined'
#Load the data
main_dir= "/home/lebrahimpour/Recurrence"
#os.getenv('JOBDIR')

data = pd.read_excel(os.path.join(main_dir,'data/recurrence_data.xlsx'))

# Check if the folder exists, and create it if it doesn't
if not os.path.exists(os.path.join(main_dir, 'results')):
    os.makedirs(os.path.join(main_dir, 'results'))
if not os.path.exists(os.path.join(main_dir, 'results/'+file_name)):
    os.makedirs(os.path.join(main_dir, 'results/'+file_name))




# Preprocess the data

# Filter the data for stage 1 and stage 2
filtered_data = data[data['stage'].isin([1,2,3,4])]

# Explicitly create a copy of the data
filtered_data_copy = filtered_data.copy()

#Remove rows with missing data
filtered_data_copy.dropna(inplace=True)

# Set the target variable as recurrence for stage 1 and stage 2
target= filtered_data_copy['recurrence'].copy()

# Set the features as the radiomic features
features = filtered_data_copy.drop(columns=['PatientName', 'stage', 'recurrence', 'OS-months', 'OS-days', 'PFS-months', 'PFS-days', 'VitalStatus', 'Smoking', 'Age', 'Subtype', 'Sex'])
#features = filtered_data_copy.drop(columns=['PatientName', 'stage', 'recurrence', 'OS-months', 'OS-days', 'PFS-months', 'PFS-days', 'VitalStatus'])

print("number of features = ", np.shape(features))

#Remove constant radiomic features
constant_features = features.columns[features.nunique() == 1]
features.drop(constant_features, axis=1, inplace=True)
print("number of non-constant features = ", np.shape(features))

# Store the feature names
feature_names = features.columns.tolist()

#constant_features = [col for col in features if features[col].nunique() == 1]
#features.drop(constant_features, axis=1, inplace=True)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Separate positive and negative cases in the training set
positive_cases_X_train = X_train[(y_train == 1)]
negative_cases_X_train = X_train[(y_train == 0)]

# Separate positive and negative cases in the test set
positive_cases_X_test = X_test[(y_test == 1)]
negative_cases_X_test = X_test[(y_test == 0)]


print("number of positive_train = ", len(positive_cases_X_train))
print("number of negative_train = ", len(negative_cases_X_train))

print("number of positive_test = ", len(positive_cases_X_test))
print("number of negative_test = ", len(negative_cases_X_test))


# Create a SMOTE instance
#smote = SMOTE(sampling_strategy='minority')

# Apply SMOTE to the training data det
#X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# Create a SMOTE instance
smote = SMOTE(sampling_strategy=0.5)  

# Apply SMOTE to the training data
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

positive_cases_X_train_oversampled = X_train_oversampled[(y_train_oversampled == 1)]
negative_cases_X_train_oversampled = X_train_oversampled[(y_train_oversampled == 0)]

print("number of positive_train_oversampled = ", len(positive_cases_X_train_oversampled))
print("number of negative_train_oversampled = ", len(negative_cases_X_train_oversampled))
print("number of oversampled, non-constant features in training data  = ", np.shape(X_train_oversampled))

# Create a RandomUnderSampler instance
rus = RandomUnderSampler(sampling_strategy='majority') 

# Apply RandomUnderSampler to the oversampled data
X_train_balanced, y_train_balanced = rus.fit_resample(X_train_oversampled, y_train_oversampled)


# Create an instance of RandomUnderSampler for undersampling the negative cases
#rus = RandomUnderSampler(sampling_strategy='majority')


# Apply RandomUnderSampler to the training data
#X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)


positive_cases_X_train_balanced = X_train_balanced[(y_train_balanced == 1)]
negative_cases_X_train_balanced = X_train_balanced[(y_train_balanced == 0)]

print("number of positive_train_balanced = ", len(positive_cases_X_train_balanced))
print("number of negative_train_balanced = ", len(negative_cases_X_train_balanced))

print("number of balanced, non-constant features in training data  = ", np.shape(X_train_balanced))
print("number of non-constant in test data = ", np.shape(X_test))
# Reassess the correlation matrix (optional)
#correlation_matrix_after_drop = np.corrcoef(X_train_balanced.T)

print("number of selected features = " , len(feature_names))
#print(correlated_features)
#print(selected_feature_names)

# Combine the oversampled positive cases with the original negative cases
#X_train_balanced = pd.concat([negative_cases_X_train, maxX_pos_oversampled])
#y_train_balanced = pd.concat([negative_cases_Y_train, y_pos_oversampled])


# Upsample the positive cases to match the number of negative cases
#positive_cases_upsampled = resample(positive_cases_train, n_samples=len(negative_cases_train), random_state=42)
#print("number of positive_train_unsampled = ", len(positive_cases_upsampled))

# Combine the upsampled positive cases with the negative cases
#X_train_balanced = pd.concat([negative_cases_train, positive_cases_upsampled])
#y_train_balanced = pd.concat([pd.Series([0]*len(negative_cases_train)), #pd.Series([1]*len(positive_cases_upsampled))])

# Scale the features of the balanced training set
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameter grids for each model
hyperparameter_grids = {
    'GaussianProcesses': {'n_restarts_optimizer': [0, 1, 2]},'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']} 
    ,'GradientBoosting': {'n_estimators': [50, 100, 200],'learning_rate': [0.1, 0.5, 1.0], 'max_depth': [3, 5, 10]} 
    ,'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]} 
    ,'DecisionTree' : {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]} 
    ,'NeuronalNetwork': {'hidden_layer_sizes': [(100,), (100, 50), (200, 100)], 'activation': ['relu', 'tanh'],  'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
    'batch_size': [32, 64, 128]  # Batch size
    }
#    ,'XGBoost': {
#    'n_estimators': [100, 200, 300],           # Number of boosting rounds
#    'learning_rate': [0.01, 0.1, 0.2],         # Step size at each boosting iteration
#    'max_depth': [3, 4, 5],                    # Maximum depth of each tree
#    'min_child_weight': [1, 2, 3],             # Minimum sum of instance weight needed in a child
#    'subsample': [0.8, 0.9, 1.0],             # Fraction of samples used for fitting trees
#    'colsample_bytree': [0.8, 0.9, 1.0],      # Fraction of features used for fitting trees
#    'gamma': [0, 0.1, 0.2],                    # Minimum loss reduction required to make a further partition
#    'reg_alpha': [0, 0.1, 0.2],                # L1 regularization term on weights
#    'reg_lambda': [0, 0.1, 0.2]                # L2 regularization term on weights
#}
}
def select_features(method, X_train, y, X_test, n, feature_names):
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=n)
    elif method == 'reliefF':
        selector = ReliefF(n_neighbors=100, n_features_to_select=n)
    elif method == 'surf':
        selector = SURF(n_features_to_select=n)
    elif method == 'multisurf':
        selector = MultiSURF(n_features_to_select=n)

    elif method == 'f_test':
        selector = SelectKBest(score_func=f_classif, k=n)
    else:
        raise ValueError('Invalid feature selection method.')

#    selector.fit(X, y.values)
    X_train_new = selector.fit_transform(X_train, y.values)
    X_test_new = selector.transform(X_test)
    if method == 'mutual_info' or method == 'f_test':
        selected_feature_indices = selector.get_support(indices=True)
    else:
    # Get the feature importance scores
        feature_importances = selector.feature_importances_

# Sort the features by importance scores and get the indices of the top n features
        selected_feature_indices = np.argsort(feature_importances)[-n:]
    selected_features = [feature_names[i] for i in selected_feature_indices]
    return X_train_new, X_test_new, selected_features, selector

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall
    return accuracy, sensitivity, specificity, f1, auc

# Define models
models = {
    'GaussianProcesses': GaussianProcessClassifier()
    ,'SVM': SVC(probability=True)
    ,'GradientBoosting': GradientBoostingClassifier() 
    ,'RandomForest': RandomForestClassifier() 
    ,'DecisionTree' : DecisionTreeClassifier() 
    ,'NeuronalNetwork': MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1)
#    ,'XGBoost': XGBClassifier()

}

# Create a LogisticRegression model with L1 penalty
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
# Define the range of C values to search
param_grid = {'C': [i * 0.5 for i in range(1, 51)]}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_balanced_scaled, y_train_balanced)
# Get the best C value
best_C = grid_search.best_params_['C']
# Train Lasso model on the entire training data
lasso_model_best = LogisticRegression(penalty='l1', solver='liblinear', C=best_C)
lasso_model_best.fit(X_train_balanced_scaled, y_train_balanced)

# Get the coefficients and feature names
coefficients = lasso_model_best.coef_
#feature_names = ['feature_' + str(i) for i in range(X_train_balanced_scaled.shape[1])]
# Identify zero coefficients and their corresponding feature names
zero_coefficient_indices = np.where(coefficients == 0)[1]
#zero_coefficient_features = [feature_names[idx] for idx in zero_coefficient_indices]
# Remove zero-coefficient features from the dataset
X_train_selected = X_train_balanced_scaled[:, ~np.isin(np.arange(X_train_balanced_scaled.shape[1]), zero_coefficient_indices)]
X_test_selected = X_test_scaled[:, ~np.isin(np.arange(X_test_scaled.shape[1]), zero_coefficient_indices)]

# Get the indices of the selected features after Lasso-based feature selection
selected_feature_indices_after_lasso = np.where(~np.isin(np.arange(X_train_balanced_scaled.shape[1]), zero_coefficient_indices))[0]

# Get the names of the features after Lasso-based feature selection
selected_feature_names_after_lasso = [feature_names[idx] for idx in selected_feature_indices_after_lasso]


# Define feature selection methods and the maximum number of features to consider
feature_selection_methods = ['f_test', 'mutual_info', 'reliefF', 'surf', 'multisurf']
#feature_selection_methods = [ 'surf']
max_features = X_train_selected.shape[1]
print("Number of features = ", max_features)

#Train and evaluate models using the balanced and scaled training set with grid search cross-validation
results = [] 
best_results = {}
best_features = {}

step_size = 1
end_value = max_features

iterations = list(range(step_size, end_value + 1, step_size))
if end_value % step_size != 0:
    iterations.append(end_value)

for method in feature_selection_methods:
    print('\n')
    print("Method = ", method)
    print('\n')
    for n_features in iterations:
        print('\n')
        print("Number of features = ", n_features)
        print('\n')
        X_train_new, X_test_new, selected_features, selector = select_features(method, X_train_selected, y_train_balanced, X_test_selected, n_features, selected_feature_names_after_lasso)
#        X_test_new = X_test_selected[:, selector.get_support()]
        
        for model_name, model in models.items():
            print('\n')
            print("ML Model = ", model_name)
            print('\n')

            # Perform grid search cross-validation for hyperparameter tuning
            param_grid = hyperparameter_grids[model_name]
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
            grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv)
#            grid_search = BayesSearchCV(model, param_grid, scoring='roc_auc', cv=cv, random_state=42)
            grid_search.fit(X_train_new, y_train_balanced)
            best_model = grid_search.best_estimator_
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

            # Perform cross-validation on the best model
            cv_scores = cross_val_score(best_model, X_train_new, y_train_balanced, cv=cv, scoring='roc_auc')
            cv_auc = cv_scores.mean()            
            # Evaluate the best model on the test set
            accuracy, sensitivity, specificity, f1, auc = train_and_evaluate_model(best_model, X_train_new, y_train_balanced, X_test_new, y_test)
#            results.append((method, n, model_name, cv_accuracy, accuracy, precision, recall, f1))
            if model_name not in best_results or auc > best_results[model_name]['auc']:
                best_results[model_name] = {
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1': f1,
                    'auc': auc,
                    'cv_auc' : cv_auc,
                    'selector': method,
                    'n_features': n_features
                }
                feature_names_selected = selected_features
                best_features[model_name] = {"selecetd_feature_names":feature_names_selected}
                # Save a trained model to a file
                # Path to the ffile
                results_best_trained_model = os.path.join(main_dir,'results/'+file_name+'/'+'model_'+model_name+'.pkl')

                # Save the best_results dictionary to a JSON file
                with open(results_best_trained_model, 'wb') as pickle_file:
                    joblib.dump(best_model, pickle_file)

        print(best_results)            

# Path to the results JSON file
results_file_summary = os.path.join(main_dir,'results/'+file_name+'/'+'best_results'+'.json')

# Path to the features names of the results JSON file
results_file_features = os.path.join(main_dir,'results/'+file_name+'/'+'best_features'+'.json')

# Save the best_results dictionary to a JSON file
with open(results_file_summary, 'w') as json_file:
    json.dump(best_results, json_file)

# Save the best_features dictionary to a JSON file
with open(results_file_features, 'w') as json_file:
    json.dump(best_features, json_file)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
