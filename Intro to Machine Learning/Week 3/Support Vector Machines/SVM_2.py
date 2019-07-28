import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

newsgroups = datasets.fetch_20newsgroups(subset = 'all',categories=['alt.atheism','sci.space'])
X = np.array(newsgroups.data)
Y = np.array(newsgroups.target)
TF = TfidfVectorizer()
X_train = TF.fit_transform(X)
X_test = TF.transform(X)
kf = KFold(n_splits = 5,shuffle = True , random_state = 241)
#grid = {'C': np.power(10.0,np.arange(-5,6))}


clf = SVC(kernel = 'linear',random_state = 241,C = 1.0)
#gs = GridSearchCV(clf,grid,cv = kf,scoring = 'accuracy')
#gs = gs.fit(X_train,Y)

clf = clf.fit(X_train,Y)
coef = clf.coef_
coef = coef.toarray()[0]
print(type(coef))
a = coef
coef = np.sort(coef)
first = coef[0:10]
last = coef[-10:]
print(first)
print(last)
print(np.where(a >1.0242292))
arr = np.argsort(clf.coef_.todense())
feature_map = TF.get_feature_names()
arr = np.array(arr)
new = [item for sublist in arr for item in sublist]

arr1 = []
arr1.append([feature_map[17802],feature_map[22936],feature_map[23673],feature_map[24019],feature_map[5088],feature_map[5093],feature_map[5776],feature_map[7597],feature_map[12871],feature_map[15606],feature_map[21850]])
newarr = [item for sublist in arr1 for item in sublist]
newarr.sort()
print(newarr)