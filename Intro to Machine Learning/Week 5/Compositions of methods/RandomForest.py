import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: -1 if x == "F" else 0 if x == 'I' else 1)
X = data.iloc[:,:-1]
Y = data.loc[:,'Rings']

#metric = make_scorer(foo)

kf = KFold(n_splits = 5,shuffle = True,random_state = 1)
res = {}

for l in range(1,51):
  clf = RandomForestRegressor(random_state = 1,n_estimators = l)
  clf = clf.fit(X,Y)
  pred = clf.predict(X)
  res[l-1] = np.mean(cross_val_score(clf,X,Y,cv = kf,scoring = 'r2'))

print(res)






