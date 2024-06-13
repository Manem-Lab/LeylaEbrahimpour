import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline, make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif
from skrebate import ReliefF, SURF, MultiSURF
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import VotingClassifier
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURF
from skrebate import MultiSURFstar
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
from scipy.stats import randint as sp_randint
import warnings
import time
import os
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, KMeansSMOTE
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# get the start time
st = time.time()
file_name = 'cancerprediction_rads_AddDelta_lasso_NoSmote'

#Load the data
main_dir= "/home/ulaval.ca/lesee/projects/Project-NLST/"

data_train_T0 = pd.read_excel(os.path.join(main_dir,'data/data_radiomics/radiomicsfeatures_kheops-NLST-Dmitrii-Cohort1_Laptop_v2-cleaned-T0-ML.xlsx'))
data_train_T1 = pd.read_excel(os.path.join(main_dir,'data/data_radiomics/radiomicsfeatures_kheops-NLST-Dmitrii-Cohort1_Laptop_v2-cleaned-T1-ML.xlsx'))
data_test_T0 = pd.read_excel(os.path.join(main_dir,'data/data_radiomics/radiomicsfeatures_kheops-NLST-Dmitrii-Cohort2_Laptop_v2-cleaned-T0-ML.xlsx'))
data_test_T2 = pd.read_excel(os.path.join(main_dir,'data/data_radiomics/radiomicsfeatures_kheops-NLST-Dmitrii-Cohort2_Laptop_v2-cleaned-T2-ML.xlsx'))

# Check if the folder exists, and create it if it doesn't
if not os.path.exists(os.path.join(main_dir, 'results')):
    os.makedirs(os.path.join(main_dir, 'results'))
if not os.path.exists(os.path.join(main_dir, 'results/'+file_name)):
    os.makedirs(os.path.join(main_dir, 'results/'+file_name))
    
# Preprocess the data

# Explicitly create a copy of the data
data_train_T0_copy = data_train_T0.copy()
data_train_T1_copy = data_train_T1.copy()
data_test_T0_copy = data_test_T0.copy()
data_test_T2_copy = data_test_T2.copy()

# Encoding with map
data_train_T0_copy['label_encoded'] = data_train_T0_copy['Label'].map({'benign': 0, 'malignant': 1})
data_test_T0_copy['label_encoded'] = data_test_T0_copy['Label'].map({'benign': 0, 'malignant': 1})

# Set the target variable
target_train= data_train_T0_copy['label_encoded'].copy()
target_test= data_test_T0_copy['label_encoded'].copy()

# Set the features as the radiomic features
features_train_T0 = data_train_T0_copy.drop(columns=['label_encoded','Label','PatientID'])
features_train_T1 = data_train_T1_copy.drop(columns=['Label','PatientID'])
features_test_T0 = data_test_T0_copy.drop(columns=['label_encoded','Label','PatientID'])
features_test_T2 = data_test_T2_copy.drop(columns=['Label','PatientID'])

print("number of features for T0 training set = ", np.shape(features_train_T0))
print("number of features for T1 training set = ", np.shape(features_train_T1))


print("number of features for T0 test set = ", np.shape(features_test_T0))
print("number of features for T2 test set = ", np.shape(features_test_T2))

# Assuming features_train_T0, features_train_T1, features_test_T0, and features_test_T2 
# are pandas DataFrames and have the same structure and corresponding indices after the drop operation

# Calculate the delta for the training features
features_train = features_train_T0 - features_train_T1

# Calculate the delta for the test features
features_test = features_test_T0 - features_test_T2


#Remove constant radiomic features
constant_features = features_train.columns[features_train.nunique() == 1]
features_train.drop(constant_features, axis=1, inplace=True)
features_test.drop(constant_features, axis=1, inplace=True)
print("number of non-constant features in training data = ", np.shape(features_train))
print("number of non-constant features in test data = ", np.shape(features_test))

# Store the feature names
feature_names = features_train.columns.tolist()
print("Initial number of features = " , len(feature_names))

# Split the data into training and testing sets
X_train = features_train
X_validation =  features_test
y_train = target_train
y_validation = target_test

X_train_init = X_train.copy()
X_validation_init = X_validation.copy()

# Scale the features of the balanced training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

# Separate positive and negative cases in the training set
positive_cases_X_train = X_train[(y_train == 1)]
negative_cases_X_train = X_train[(y_train == 0)]

# Separate positive and negative cases in the test set
positive_cases_X_test = X_validation[(y_validation == 1)]
negative_cases_X_test = X_validation[(y_validation == 0)]


print("number of positive_train = ", len(positive_cases_X_train))
print("number of negative_train = ", len(negative_cases_X_train))

print("number of positive_test = ", len(positive_cases_X_test))
print("number of negative_test = ", len(negative_cases_X_test))

    
# Classifier list
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000),  # Neural Network
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'XGBoost': XGBClassifier( random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'Logistic': LogisticRegression(random_state=42),
    'Kneighbors': KNeighborsClassifier()
}



param_rf = {
        "model__max_depth": sp_randint(3, 20),
        "model__n_estimators": sp_randint(50, 500),
        "model__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False]
    
    

    }
param_dt = {
        "model__max_depth": sp_randint(3, 20),
        "model__min_samples_leaf": sp_randint(1, 10),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__criterion": ['gini', 'entropy']
    }


param_xgb = {
    "model__n_estimators": randint(10, 100),
    "model__max_depth": sp_randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.3),
    'model__min_child_weight': randint(1, 6),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__gamma": uniform(0, 5)
}


param_gbc = {
    "model__n_estimators": randint(50, 300),
    "model__max_depth": sp_randint(3, 12),
    "model__learning_rate": uniform(0.01, 0.3),
    "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 20)
}


param_ada = {
    "model__n_estimators": randint(50, 500),
    "model__learning_rate": uniform(0.01, 2.00),
    "model__algorithm": ["SAMME", "SAMME.R"]
}


param_svc = {
    'model__C': uniform(0.1, 10),
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__gamma': ['scale', 'auto'] + list(np.logspace(-9, 3, 13)),
    'model__degree': [1, 2, 3, 4, 5]  # Only used for 'poly' kernel
}



param_lr = {
    "model__C": np.logspace(-4, 4, 20),
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear", "saga"],
    'model__max_iter' : [100, 1000, 2500, 5000]
}


param_knn = {
    "model__n_neighbors": randint(3, 30),
    "model__weights": ["uniform", "distance"],
#    "model__algorithm": ["ball_tree", "kd_tree", "brute"],
#    "model__leaf_size": randint(20, 40),
    'model__metric': ['euclidean']
}


param_gnb = {
    'model__var_smoothing': np.logspace(0,-9, num=100)
}

param_mlp = {
    "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "model__activation": ["identity", "logistic", "tanh", "relu"],
    "model__solver": ["sgd", "adam"],
    "model__alpha": uniform(0.0001, 0.001),
    "model__learning_rate": ["constant", "adaptive"],
}


param_model = [param_dt, param_gnb, param_svc, param_rf, param_mlp, param_ada, param_xgb, param_gbc, param_lr, param_knn]
# Define feature selection methods 
feature_selection_methods = ['f_test', 'mutual_info', 'reliefF', 'surf', 'multisurf']



def train_and_evaluate_model(selected_pipe, X_train, y_train, X_test, y_test):
    selected_pipe.fit(X_train, y_train)
    y_pred = selected_pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
#    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
#    sensitivity = recall
    return accuracy, sensitivity, specificity, f1, auc, tn, fp, fn, tp, precision

def select_features(method, n):
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
    return selector


# Assuming feature_names is defined and contains the names of your features
# Assuming X_train_scaled, y_train, X_validation_scaled, and y_validation are already defined

# Define the cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=10)

# For classification, define an appropriate scorer. For example, using accuracy:
def my_scorer(y_test, y_pred):
    score = roc_auc_score(y_test, y_pred)
    return score

my_func = make_scorer(my_scorer, greater_is_better=True)

# Use Logistic Regression with L1 penalty for feature selection
lasso_logistic = LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000)

param_grid = {'C': [1 / (i * 0.5) for i in range(1, 100)]}  # Note: C is the inverse of alpha

grid_search = GridSearchCV(lasso_logistic, param_grid, cv=cv, scoring=my_func)
grid_search.fit(X_train_scaled, y_train)

best_C = grid_search.best_params_['C']
lasso_logistic_best = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=5000)
lasso_logistic_best.fit(X_train_scaled, y_train)

coefficients = lasso_logistic_best.coef_[0]  # Get the coefficients of the fitted model
zero_coefficient_indices = list(np.where(coefficients == 0)[0])

X_train_scaled_lasso = X_train_scaled[:, ~np.isin(np.arange(X_train_scaled.shape[1]), zero_coefficient_indices)]
X_validation_scaled_lasso = X_validation_scaled[:, ~np.isin(np.arange(X_validation_scaled.shape[1]), zero_coefficient_indices)]

# Get the indices of the selected features after Lasso-based feature selection
selected_feature_indices_after_lasso = np.where(~np.isin(np.arange(X_train_scaled.shape[1]), zero_coefficient_indices))[0]

# Get the names of the features after Lasso-based feature selection
selected_feature_names_after_lasso = [feature_names[idx] for idx in selected_feature_indices_after_lasso]

final_selected_features = selected_feature_names_after_lasso 

print("Shape of X_test after Lasso = ", np.shape(X_validation_scaled_lasso))
print("Shape of X_train after Lasso = ", np.shape(X_train_scaled_lasso))
print("Shape of y_test after Lasso = ", np.shape(y_validation))
print("No. of features after Lasso = ", len(final_selected_features))

n_splits = 5  #number of folds for CV
feature_range = range(3, np.shape(X_train_scaled_lasso)[1])  #range of number of features
print(feature_range)
#feature_range = range(3, 4)  #range of number of features
#n_iter = 1  # Number of iterations for GridSearch
n_iter = 100  # Number of iterations for GridSearch
#best_scores_cv = []




# Cross-validation setup
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for (model_name, model), params in zip(classifiers.items(), param_model):    
    model_scores = []
    print('---------------------------------------- '+ model_name +' --------------------------------------------------')
    for method in feature_selection_methods:
        print('---------------------------------------- '+ method +' --------------------------------------------------')


        best_score_cv = 0
        best_params = {'n_features_to_select': None, 'params': None}
        #start the ML pipeline with number of features selected from a range of values                    
        for n_features in feature_range:
            #pipline with feature selection method, balancing method and the ML model
            pipe = Pipeline([
                ('method', select_features(method, n_features)),
                ('model', model)
            ])

            # Configurer RandomizedSearchCV
    #        clf = GridSearchCV(pipe, params, scoring=make_scorer(roc_auc_score), cv=cv)
            clf = RandomizedSearchCV(pipe, params, random_state = 42, n_iter=n_iter, scoring=make_scorer(roc_auc_score), cv=cv)
            clf.fit(X_train_scaled_lasso, np.ravel(y_train.values))

            if clf.best_score_ > best_score_cv:
                best_score_cv = clf.best_score_
                best_params['n_features_to_select'] = n_features
                best_params['params'] = clf.best_params_
                # Index of the best score
                best_index = clf.best_index_

                # Standard deviation of the best score
                std_dev_of_best_score_cv = clf.cv_results_['std_test_score'][best_index]

            # Append the name and the score of best model to best_scores 
    #    best_scores_cv.append({'Model': type(model).__name__, 'Best CV Score': best_score_cv})
        # print the paraneters and the AUC score of the best model on training data
#        print(f"Best Parameters for {type(model).__name__} with {method}: {best_params}, AUC in CV: {best_score_cv}")
        print(f"Best number of feautures for {type(model).__name__} with {method}: {best_params['n_features_to_select']} features with AUC score in CV: {best_score_cv} +- {std_dev_of_best_score_cv} ")
                 

        # Find the parameters of the best model
        model_params = {k.split("__")[1]: v for k, v in best_params['params'].items() if k.startswith('model__')}

        # Create the final model
        final_model = type(model)(**model_params)
        final_pipe = Pipeline([
            ('method', select_features(method, best_params['n_features_to_select'])),
            ('model', final_model)])

        final_pipe.fit(X_train_scaled_lasso, np.ravel(y_train.values))
        accuracy, sensitivity, specificity, f1, auc_test, tn, fp, fn, tp, precision = train_and_evaluate_model(final_pipe, X_train_scaled_lasso, np.ravel(y_train.values), X_validation_scaled_lasso, y_validation)

        # Append the model name and the validation scores 
        model_scores.append({'Model': type(model).__name__,'Selection Method': method, 'n_features': best_params['n_features_to_select'], 'accuracy': accuracy,'precision' : precision, 'TP': tp, 'FP':fp, 'FN':fn, 'TN':tn,
                             'sensitivity': sensitivity,'specificity': specificity, 'f1': f1, 'auc_test': auc_test, 'auc_cv' : best_score_cv, 'std_auc_cv' : std_dev_of_best_score_cv})
        print(f"AUC score on Validation dataset for the best scored trained model for {type(model).__name__} : {auc_test}") 

    df_scores_method = pd.DataFrame(model_scores)
    df_scores_method.set_index(df_scores_method.columns[0], inplace=True)
    #df_scoresmi.rename(columns={'Validation Score': 'MI'}, inplace=True)
    print(df_scores_method)
    
    # Save the maximum scores Daraframe to a CSV file in a folder with model's name 
    if not os.path.exists(os.path.join(main_dir, 'results/'+file_name+'/'+model_name)):
        os.makedirs(os.path.join(main_dir, 'results/'+file_name+'/'+model_name))

    df_scores_method.to_csv(os.path.join(main_dir, 'results/'+file_name+'/'+model_name+'/'+ model_name+'_scores_with_NoSmote_addDeltaRads.csv'), index=False, float_format='%.7f') 


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')