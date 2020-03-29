import os
import sys
import csv
import time
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, auc

# Mute warning messages
warnings.filterwarnings("ignore")

# Set working directory
os.chdir("/nfs/home/qiche/Project/CoCA/WP7/SB_visit/")

# Set log file
sys.stdout = open("./Output/log_grid_search.txt", "w")

# Load data
train = pd.read_csv("./Input/train.csv")
test = pd.read_csv("./Input/test.csv")

outcome = '90'

outcome_index = train.columns.get_loc('sb' + outcome)

# Separate features and target variables in the training set
X_train = train.iloc[:,1:426]
y_train = train.iloc[:, outcome_index]
groups_train = train.iloc[:, 0]

# Separate features and target variables in the test set
X_test = test.iloc[:,1:426]
y_test = test.iloc[:, outcome_index]
groups_test = test.iloc[:, 0]

'''
filter_col = [col for col in train if col.startswith(('dis','med')) and not col.endswith(('_r'))]

outcome = '90'

outcome_index = train.columns.get_loc('sb' + outcome)

# Separate features and target variables in the training set
X_train = train[['lopnr', 'sex', 'dia_age'] + filter_col]
y_train = train.iloc[:, outcome_index]
groups_train = train.iloc[:, 0]

# Separate features and target variables in the test set
X_test = test[['lopnr', 'sex', 'dia_age'] + filter_col]
y_test = test.iloc[:, outcome_index]
groups_test = test.iloc[:, 0]

'''

# Set log file
#sys.stdout = open("./Output/log_predict_%s.txt" % outcome, "w")
print('\nOutcome: suicide attempt/death within %s days' % outcome)
print('\nDimension of train set: ', X_train.shape)
print('\nDimension of test set: ', X_test.shape)

# Define classification estimators
def choose_model(model="mlp"):

    if model == "en":
        parameters = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05],
                      'l1_ratio': [0.01, 0.03, 0.05, 0.07, 0.09],
                      'penalty': ['elasticnet']}
        model_name = "Elastic Net"

    elif model == "rf":
        parameters = {'criterion': ['entropy'],
                      'max_features': [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1],
                      'n_estimators': [100, 200, 300, 400, 500],
                      'max_depth' : [3, 5, 7, 9, 11]}
        model_name = "Random Forest"

    elif model == "gb":
        parameters = {'n_estimators': [50, 100, 200, 300],
                      'learning_rate': [0.05, 0.1, 0.3],
                      'subsample': [0.5, 0.7, 0.9],
                      'max_depth': [3, 4, 5, 6, 7],
                      'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                      'gamma': [1]}
        model_name = "Gradient Boosting"
        
    elif model == "mlp":
        parameters = {'solver': ['sgd'],
                      'hidden_layer_sizes': [(6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (8, 2)],
		      'alpha': [0.5, 0.6, 0.7]}
        model_name = "Neural Network"
        
    return parameters, model, model_name
 


def run_model(model, parameters, cv):

    en = SGDClassifier(loss='log', random_state=33, max_iter=1000)
    rf = RandomForestClassifier(random_state=33)
    gb = XGBClassifier(objective='binary:logistic', random_state=33)
    nnet = MLPClassifier(random_state=33)                                        
                                              

    if model == "en": grid_cv = GridSearchCV(en, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 10, return_train_score=True)
    if model == "rf": grid_cv = GridSearchCV(rf, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 2, return_train_score=True)
    if model == "gb": grid_cv = GridSearchCV(gb, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 10, return_train_score=True)
    if model == "mlp": grid_cv = GridSearchCV(nnet, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 10, return_train_score=True)                                     
   
    return grid_cv



start_time = time.time()
print('\nGrid search for hyperparameters...')
print('\n---------------------------------------------------------------------------------------------------------\n')

clfs={}
params={}
if __name__ == '__main__':
    for ml_model in ['en', 'rf', 'gb', 'mlp']:
        parameters, model, model_name = choose_model(model=ml_model)
        cv = GroupKFold(n_splits=10).split(X_train.values, y_train.values, groups_train.values)
        grid_cv = run_model(model, parameters, cv = cv)
        print('%s' % model_name)
        grid_cv.fit(X_train, y_train)
        print('\n\nThe best classifier from GridSearchCV:')
        print(grid_cv.best_estimator_)
        clfs[ml_model]=grid_cv.best_estimator_
        print('\n\nThe best set of parameters:')
        print(grid_cv.best_params_)
        params[ml_model]=grid_cv.best_params_
        print('\n\nGrid scores on development set:')
        means_train = grid_cv.cv_results_['mean_train_score']
        std_train = grid_cv.cv_results_['std_train_score']
        means_test = grid_cv.cv_results_['mean_test_score']
        std_test = grid_cv.cv_results_['std_test_score']
        for train, train_std, test, test_std, params in zip(means_train, std_train, means_test, std_test, grid_cv.cv_results_['params']):
            print('Training %0.3f (sd: %0.3f) validation %0.3f (sd: %0.3f) for %r'
                  % (train, train_std, test, test_std, params))
        print('\n---------------------------------------------------------------------------------------------------------\n')

print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
sys.stdout.close()
sys.stdout = sys.__stdout__
