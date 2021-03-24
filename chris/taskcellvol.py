# Literally just Gino's task1 code, with 2 strings changed...

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
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
# feature_headers = headers = [*pd.read_csv('../data/train_descriptors.csv', 
#                              nrows=1)]
# features = pd.read_csv(
#     '../data/train_descriptors.csv',
#     usecols=[c for c in headers if not c in [
#         'identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']]
# )

feature_headers = headers = [*pd.read_csv('../data/train_mord3d.csv', 
                             nrows=1)]
features = pd.read_csv(
    '../data/train_mord3d.csv',
    usecols=[c for c in headers if not c in [
        'identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles']]
)
data = pd.read_csv('../data/train_crystals.csv')
#%% Train / test splitting
target = data['cell_volume']
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42)
y_train = y_train.to_numpy()
# # %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', verbose=1)),
    ('scaler', MinMaxScaler()),
    ('feature_sel', SelectKBest(f_regression, k = 10)), # using it all seems better..!?
    ('fitting', KernelRidge())
])
#%% Fitting
pclf.fit(X_train, y_train)
#%% Prediction
y_pred = pclf.predict(X_test)
print("MAE::::::", mean_absolute_error(y_test, y_pred))
#%% plotting
plt.plot(y_test, y_pred, 'r.')
plt.show()
#%% saving
# test_data = pd.read_csv(
#     '../data/test_descriptors.csv',
#     usecols=[c for c in headers if not c in [
#         'identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']]
# ).to_numpy()
# test_pred = pclf.predict(test_data)
# # np.savetxt('task_cellvol_predictions.csv', test_pred)
