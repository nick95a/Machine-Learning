import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/close_prices.csv')
dataset = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/djia_index.csv')
print(data.head())
X = data.iloc[:,1:]
dow = dataset.iloc[:,1]
dow = np.array(dow)
pca = PCA(n_components = 10)
pca.fit(X)
new = pca.transform(X)
result = []

def first(arr):
    i = 0
    for i in range(374):
     a = arr[i][0]
     result.append(a)
     i += 1
    return result
comp = first(new)
comp = np.array(comp)
print(np.max(pca.components_[0]))
print(pca.components_[0])