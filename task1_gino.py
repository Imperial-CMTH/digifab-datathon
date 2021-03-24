#%% Imports + Data loading
from traceback import clear_frames
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

feature_headers = headers = [*pd.read_csv('./data/train_descriptors.csv', nrows=1)]
features = pd.read_csv(
    './data/train_descriptors.csv',
    usecols=[c for c in headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']]
)
data = pd.read_csv('./data/train_crystals.csv')
#%% Train / test splitting
target = data['calculated_density']
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.33, random_state=42)
y_train = y_train.to_numpy()
# %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', verbose=1)),
    ('scaler', MinMaxScaler()),
    ('feature_sel', SelectKBest(f_regression, k = 10)),
    ('fitting', KernelRidge())
])
#%% Fitting
pclf.fit(X_train, y_train)
#%% Prediction
y_pred = pclf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
#%% plotting
plt.plot(y_test, y_pred, 'r.')
plt.show()
#%% saving
np.savetxt('./out/task_1_predictions.csv', y_pred)
# %%
