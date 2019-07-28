import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import operator
from sklearn.preprocessing import scale

data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/wine.txt',header = None , names = ['Class','Alcohol','Malic_Acid','Ash','Alcalinity_of_Ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315','Proline'])
X = data.iloc[:,1:14]
Y = data.iloc[:,0]
kf = KFold(n_splits = 5 , shuffle= True,random_state=42)

res = {}

X = scale(X)
for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    res[k] = np.mean(cross_val_score(neigh,X,Y,cv = kf,scoring='accuracy'))


def max_keys(dic):
    maximum = max(dic.values())
    keys = [k for k,v in dic.items() if v == maximum]
    return keys,maximum

print(max_keys(res))