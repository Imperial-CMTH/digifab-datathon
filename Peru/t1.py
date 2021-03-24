import pandas as pd
import sklearn as sk
import numpy as np
from rdkit import Chem
from pathlib import Path
import glob

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

test_csvs = glob.glob("../data/test_*.csv")
tests = {Path(t).stem: pd.read_csv(t) for t in test_csvs}
print(tests.keys())

train_csvs = glob.glob("../data/train_*.csv")
train = {Path(t).stem: pd.read_csv(t) for t in train_csvs}
print(train.keys())

X = train['train_descriptors']
X = X.drop(['identifiers', 'name', 'InchiKey', 'SMILES'], axis = 1)

