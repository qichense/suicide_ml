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
from sklearn.model_selection import learning_curve, cross_val_score, cross_validate, train_test_split, GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, auc
from DeLong import *

# Mute warning messages
warnings.filterwarnings("ignore")

# Set working directory
os.chdir("/nfs/home/qiche/Project/CoCA/WP7/SB_visit/")

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
sys.stdout = open("./Output/log_grid_search_%s.txt" % outcome, "w")
print('\nOutcome: suicide attempt/death within %s days' % outcome)
print('\nDimension of train set: ', X_train.shape)
print('\nDimension of test set: ', X_test.shape)

# All models
en_clf=SGDClassifier(alpha=0.001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.05, learning_rate='optimal', loss='log', max_iter=1000,
              n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
              power_t=0.5, random_state=33, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)

rf_clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=7, max_features=0.05, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=33, verbose=0, warm_start=False)

gb_clf=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.5, gamma=1,
       learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=33,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=0.7, verbosity=1)

mlp_clf=MLPClassifier(activation='relu', alpha=0.6, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(7, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=33, shuffle=True, solver='sgd', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)

start_time = time.time()
eclf1 = VotingClassifier(estimators=[('en', en_clf), ('rf', rf_clf)], voting='soft')
eclf2 = VotingClassifier(estimators=[('en', en_clf), ('gb', gb_clf)], voting='soft')
eclf3 = VotingClassifier(estimators=[('en', en_clf), ('mlp', mlp_clf)], voting='soft')
eclf4 = VotingClassifier(estimators=[('rf', rf_clf), ('gb', gb_clf)], voting='soft')
eclf5 = VotingClassifier(estimators=[('rf', rf_clf), ('mlp', mlp_clf)], voting='soft')
eclf6 = VotingClassifier(estimators=[('gb', gb_clf), ('mlp', mlp_clf)], voting='soft')
eclf7 = VotingClassifier(estimators=[('en', en_clf), ('rf', rf_clf),('gb', gb_clf)], voting='soft')
eclf8 = VotingClassifier(estimators=[('en', en_clf), ('gb', gb_clf), ('mlp', mlp_clf)], voting='soft')
eclf9 = VotingClassifier(estimators=[('en', en_clf), ('rf', rf_clf), ('mlp', mlp_clf)], voting='soft')
eclf10 = VotingClassifier(estimators=[('rf', rf_clf),('gb', gb_clf), ('mlp', mlp_clf)], voting='soft')
eclf11 = VotingClassifier(estimators=[('en', en_clf), ('rf', rf_clf),('gb', gb_clf), ('mlp', mlp_clf)], voting='soft')

all_clfs=[en_clf, rf_clf, gb_clf, mlp_clf, eclf1, eclf2, eclf3, eclf4, eclf5, eclf6, eclf7, eclf8, eclf9, eclf10, eclf11]

i=0
for clf in all_clfs:
    cv = GroupKFold(n_splits=10).split(X_train.values, y_train.values, groups_train.values)
    scores = cross_validate(clf, X_train, y_train, scoring='roc_auc', return_train_score=True, cv=cv, n_jobs = 10)
    i+=1
    print('clf%0.0f' % i)
    print('Training AUC: %0.8f (%0.8f)' % (scores['train_score'].mean(), scores['train_score'].std()))
    print('Validation AUC: %0.8f (%0.8f)' % (scores['test_score'].mean(), scores['test_score'].std()))
    print('\n')

       
print("--- %s minutes ---" % ((time.time() - start_time)/60))

sys.stdout.close()
sys.stdout = sys.__stdout__

