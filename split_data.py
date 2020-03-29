import os
import csv
import warnings
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Mute warning messages
warnings.filterwarnings("ignore")

# Set working directory
os.chdir("/nfs/home/qiche/Project/CoCA/WP7/SB_visit/")

# Load data.dta
df = pd.read_csv("./Input/df.csv")

# Split the sample into training set and test set by lopnr
train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 33).split(df, groups=df['lopnr']))
train = df.iloc[train_inds]
test = df.iloc[test_inds]

train.to_csv('./Input/train.csv', index = False)
test.to_csv('./Input/test.csv', index = False)