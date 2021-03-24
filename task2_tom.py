# %% Imports + Data loading
from traceback import clear_frames
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

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

def read_descriptors(path):
        headers = [*pd.read_csv(path, nrows=1)]
        return pd.read_csv(path, usecols=[c for c in headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']])

features = read_descriptors('./data/train_descriptors.csv')         
data = pd.read_csv('./data/train_crystals.csv')
# %% Train / test splitting
target = data['is_centrosymmetric']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.33, random_state=42)

y_train = y_train.to_numpy()

# %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', verbose=1)),
    ('scaler', MinMaxScaler()),
    ('feature_sel', SelectKBest(chi2, k = 10)),
    ('fitting', RandomForestClassifier(random_state=0))
])
# %% Fitting
pclf.fit(X_train, y_train)

# %% Prediction
y_pred = pclf.predict(X_test)
print('f1 score: ', f1_score(y_test, y_pred, average = 'macro'))
# %% testing     
test_data = read_descriptors('./data/test_descriptors.csv') 
test_pred = pclf.predict(test_data)
#%% saving
np.savetxt('./out/task_2_predictions.csv', np.bool(test_pred))
# %%
