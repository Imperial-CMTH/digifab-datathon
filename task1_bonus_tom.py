# %% Imports + Data loading
from traceback import clear_frames
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier

import glob
train_csvs = glob.glob("./data/train_*.csv")
train = {Path(t).stem : pd.read_csv(t) for t in train_csvs}
print(train.keys())

def read_descriptors(path):
        headers = [*pd.read_csv(path, nrows=1)]
        return pd.read_csv(path, usecols=[c for c in headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']])

features = pd.concat([
    read_descriptors('./data/train_descriptors.csv'),
    train['train_rdk'].drop('0', axis = 1),
    train['train_mord3d'].drop(['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles'], axis = 1),
    train['train_mol2vec'],
    ], axis = 1)

data = pd.read_csv('./data/train_crystals.csv')
# %% Train / test splitting
target = data['packing_coefficient']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.33, random_state=42)

y_train = y_train.to_numpy()

# %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', verbose=1)),
    ('scaler', MinMaxScaler()),
    ('feature_sel', SelectKBest(f_regression, k = 300)),
    ('fitting', KernelRidge())
])
# %% Fitting
pclf.fit(X_train, y_train)

# %% Prediction
y_pred = pclf.predict(X_test)
print('mean_absolute_error: ', mean_absolute_error(y_test, y_pred))
# %% testing

test_csvs = glob.glob("./data/test_*.csv")
tests = {Path(t).stem : pd.read_csv(t) for t in test_csvs}

test_data = pd.concat([
    read_descriptors('./data/test_descriptors.csv'),
    tests['test_rdk'].drop('0', axis = 1),
    tests['test_mord3d'].drop(['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles'], axis = 1),
    tests['test_mol2vec'],
    ], axis = 1)

pclf.fit(features, target)
test_pred = pclf.predict(test_data)
# %% saving
np.savetxt('./out/bonus_1_predictions.csv', test_pred)
# %%
