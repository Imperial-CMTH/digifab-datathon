# %% Imports + Data loading
from traceback import clear_frames
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
#scaling
from sklearn.preprocessing import MinMaxScaler
#feature selection
from sklearn.feature_selection import f_regression, chi2, RFE, SelectKBest
#model selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.svm import SVR

#classifiers to try
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import glob
print('loading data')
train_csvs = glob.glob("./data/train_*.csv")
train = {Path(t).stem : pd.read_csv(t) for t in train_csvs}
print(train.keys())

def read_descriptors(path):
        headers = [*pd.read_csv(path, nrows=1)]
        return pd.read_csv(path, usecols=[c for c in headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']])


features = pd.concat([
    read_descriptors('./data/train_descriptors.csv'),
    #train['train_rdk'].drop('0', axis = 1),
    #train['train_mord3d'].drop(['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles'], axis = 1),
    train['train_mol2vec'],
    ], axis = 1)

data = pd.read_csv('./data/train_crystals.csv')
# %% Train / test splitting
target = data['is_centrosymmetric']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.33, random_state=42)

y_train = y_train.to_numpy()

#clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features="sqrt",  n_jobs=-1)
clf = RandomForestClassifier(min_samples_split=2, n_jobs=-1, max_features = 'sqrt', n_estimators = 200)

# %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', verbose=0)),
    ('scaler', MinMaxScaler()),
    ('feature_sel', SelectKBest(chi2, k = 100)),
    ('fitting', clf),
], memory = './cachdir', )

tuner = GridSearchCV(estimator = pclf,
                     param_grid = dict(
                         fitting__max_features = ['sqrt'],
                         feature_sel__k = [80,90,100],
                         fitting__n_estimators = [100,150,200],
                                      ),
                     scoring = 'f1_macro',
                     n_jobs=-1,
                     refit=True, verbose=4)

# %% Tuning
#print('Tuning')
#tuner.fit(features, target)
#print(tuner.best_score_, tuner.best_params_)

# %% Prediction
pclf.fit(X_train, y_train)
y_pred = pclf.predict(X_test)
print('f1 score: ', f1_score(y_test, y_pred, average = 'macro')) 

# print('Outputting Predictions')
# test_csvs = glob.glob("./data/test_*.csv")
# tests = {Path(t).stem : pd.read_csv(t) for t in test_csvs}

# test_data = pd.concat([
#     read_descriptors('./data/test_descriptors.csv'),
#     #tests['test_rdk'].drop('0', axis = 1),
#     tests['test_mord3d'].drop(['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles'], axis = 1),
#     #tests['test_mol2vec'],
#     ], axis = 1)

# pclf.fit(features, target)
# test_pred = pclf.predict(test_data)
# #%% saving
# with open('./out/task_2_predictions.csv', 'w') as f:
#     f.write("\n".join('True' if i else 'False' for i in test_pred))
# # %%
