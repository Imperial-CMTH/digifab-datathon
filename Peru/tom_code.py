import pandas as pd
import sklearn as sk
import numpy as np
from rdkit import Chem
from pathlib import Path
import glob

test_csvs = glob.glob("../data/test_*.csv")
tests = {Path(t).stem : pd.read_csv(t) for t in test_csvs}
tests.keys()


train_csvs = glob.glob("../data/train_*.csv")
train = {Path(t).stem : pd.read_csv(t) for t in train_csvs}
train.keys()


train['train_crystals'][]



