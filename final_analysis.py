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
sys.stdout = open("./Output/log_predict_%s.txt" % outcome, "w")
print('\nOutcome: suicide attempt/death within %s days' % outcome)

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

cv = GroupKFold(n_splits=10).get_n_splits(X_train.values, y_train.values, groups_train.values)

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

best_clf=VotingClassifier(estimators=[('en', en_clf), ('rf', rf_clf),('gb', gb_clf), ('mlp', mlp_clf)], voting='soft')
	   
best_clf.fit(X_train, y_train)


# Predict test set
prediction = best_clf.predict(X_test)
error = np.absolute(prediction - y_test)
prob = best_clf.predict_proba(X_test)
prob_pos =prob[:,1]
score = best_clf.score(X_test, y_test)
results = np.vstack((groups_test, y_test, prediction, prob_pos, error))
final_results = pd.DataFrame(np.transpose(results))
columns = ("lopnr", "y_true", "y_pred", "y_prob", "y_error")
final_results.to_csv("./Output/Pred_test_%s.csv" % outcome, header=columns, index = False)


# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, prob_pos, pos_label=1)
roc_auc = auc(fpr, tpr)

# Estimate 95% CI of the AUC
alpha = .95
auc, auc_cov = delong_roc_variance(y_test, prob_pos)

auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(
    lower_upper_q,
    loc=auc,
    scale=auc_std)

ci[ci > 1] = 1

print('\nAUC (95% CI): {:.3f} ({:.3f}–{:.3f})'.format(auc, ci[0], ci[1]))


fig = plt.figure(figsize = (6, 6))
lw = 1
plt.plot(fpr, tpr, color='sienna', lw=lw, label='AUC (95% CI): {:.3f} ({:.3f}–{:.3f})'.format(roc_auc, ci[0], ci[1]))
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='dotted')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Suicide attempt/death within %s days following a visit' % outcome)
plt.annotate('AUC (95% CI): {:.3f} ({:.3f}–{:.3f})'.format(roc_auc, ci[0], ci[1]), xy=(0.50, 0.01))
plt.gcf().subplots_adjust(bottom = 0.2)
plt.tight_layout()
fig.savefig("./Output/ROC_%s.pdf" % outcome)
plt.show()
plt.close(fig)


# CALIBRATION

name = "Ensemble model"
cv = GroupKFold(n_splits=2).split(X_train.values, y_train.values, groups_train.values)
isotonic = CalibratedClassifierCV(best_clf, cv=cv, method='isotonic')
isotonic.fit(X_train, y_train)
y_pred = isotonic.predict(X_test)

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
if hasattr(best_clf, "predict_proba"):
    prob_pos = isotonic.predict_proba(X_test)[:, 1]
else:  # use decision function
    prob_pos = isotonic.decision_function(X_test)
    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
isotonic_score = brier_score_loss(y_test, prob_pos, pos_label=1)
print("%s:" % name)
print("\tBrier: %1.3f" % (isotonic_score))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=9)
plt.plot(mean_predicted_value, fraction_of_positives, "o", markersize = 8, label=name)

plt.ylabel("Observed Proportion of Positives")
plt.ylim([-0.05, 1.06])
plt.xlabel("Mean Predicted Risk")
plt.legend(loc="lower right")
plt.title('Suicide attempt/death within %s days following a visit' % outcome)

plt.plot(prob_pos, np.full(prob_pos.shape, 1.03), '|', color = 'lightpink', ms=8, alpha = 0.5)
plt.tight_layout()
fig = plt.savefig('./Output/Calibration_curve_%s.pdf' % outcome)
plt.close(fig)

# LEARNING CURVE

cv = GroupShuffleSplit(n_splits=10, test_size = 0.20, random_state = 33).get_n_splits(X_train.values, y_train.values, groups_train.values)
train_sizes, train_scores, test_scores =learning_curve(estimator=best_clf,
                                                       X=X_train,
                                                       y=y_train,
                                                       train_sizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                                                       scoring = 'roc_auc',
                                                       cv=cv,
                                                       n_jobs=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


fig = plt.figure(figsize = (6,6))
plt.plot(train_sizes, train_mean,
         color='steelblue', marker='o',
         markersize=8, linestyle='--',
         label='Training AUC')

plt.plot(train_sizes, test_mean,
         color='darkkhaki', linestyle='--',
         marker='o', markersize=8,
         label='Validation AUC')

plt.title('Suicide attempt/death within %s days following a visit' % outcome)
plt.xlabel('Training Sample Size')
plt.ylabel('AUC')
plt.legend(loc='lower right')
plt.ylim([0.80, 1.00])
plt.tight_layout()
fig.savefig("./Output/Learning_curve_%s.pdf" % outcome)
plt.close(fig)

print('\nDONE!\n')

sys.stdout.close()
sys.stdout = sys.__stdout__
