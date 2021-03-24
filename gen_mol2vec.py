import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

model = word2vec.Word2Vec.load('./example_data/model_300dim.pkl')

def compute_vecs(descriptors_path):
    mdf = pd.read_csv(descriptors_path)
    mdf['mol'] = mdf['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]
    return np.array([x.vec for x in mdf['mol2vec']])

print('doing train')
vecs = compute_vecs('./data/train_descriptors.csv')
header = " ,".join(str(i) for i in range(300))
np.savetxt('./data/train_mol2vec.csv', vecs, delimiter = ',', header = header)

print('doing test')
vecs = compute_vecs('./data/test_descriptors.csv')
header = " ,".join(str(i) for i in range(300))
np.savetxt('./data/test_mol2vec.csv', vecs, delimiter = ',', header = header)