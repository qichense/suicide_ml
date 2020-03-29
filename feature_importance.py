import os
import sys
import csv
import warnings
import random
import numpy as np
import pandas as pd
import operator as op
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, auc, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from operator import itemgetter
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger
from DeLong import *

# Mute warning messages
warnings.filterwarnings("ignore")

# Set working directory
os.chdir("/nfs/home/qiche/Project/CoCA/WP7/SB_visit/")


# Load data.dta
train = pd.read_csv("./Input/train.csv")
test = pd.read_csv("./Input/test.csv")

outcome = '90'

# Set log file
#sys.stdout = open("./Output/log_predict_%s.txt" % outcome, "w")
#print('\nOutcome: suicide attempt/death within %s days' % outcome)

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

en_clf=SGDClassifier(alpha=0.001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.05, learning_rate='optimal', loss='log', max_iter=1000,
              n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
              power_t=0.5, random_state=33, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)
  
en_clf.fit(X_train, y_train)

feature = pd.Series(data = np.abs(en_clf.coef_)[0], index = X_train.columns)
top30_label = feature.sort_values(ascending = False).head(30)
feature_importance = pd.Series(data = en_clf.coef_[0], index = X_train.columns)
top30 = feature_importance[top30_label.index]
top30.to_csv("./Output/en_top30_%s.csv" % outcome, index=True, header = False)


rf_clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=7, max_features=0.05, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=33, verbose=0, warm_start=False)

rf_clf.fit(X_train, y_train)


feat_importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
top30 = feat_importances.nlargest(30)
top30.to_csv("./Output/rf_top30_%s.csv" % outcome, index=True, header = False)

gb_clf=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.5, gamma=1,
       learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=33,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=0.7, verbosity=1)

gb_clf.fit(X_train, y_train)

feat_importances = pd.Series(gb_clf.feature_importances_, index=X_train.columns)
top30 = feat_importances.nlargest(30)
top30.to_csv("./Output/gb_top30_%s.csv" % outcome, index=True, header = False)

