from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = datasets.load_boston()
X = data['data']
X = scale(X)
Y = data['target']
kf = KFold(n_splits=5,random_state=42,shuffle = True)

res = {}
i = 1

for k in np.linspace(1,10,200):
        neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p= 1 + k)
        res[i] = np.mean(cross_val_score(neigh,X,Y,cv = kf,scoring= 'neg_mean_squared_error'))
        i = i + 1

def keys_values(dic):
    maximum = max(dic.values())
    keys = [p for p,v in dic.items() if v == maximum]
    return keys,maximum

print(keys_values(res))
