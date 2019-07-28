import pandas as pd
import numpy as np

from sklearn.svm import SVC

data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/svm-data.csv',header = None)

X = np.array(data.iloc[:,1:])
Y = np.array(data.iloc[:,0])

clf = SVC(random_state=241,C=10000,kernel='linear')
clf = clf.fit(X,Y)
res = np.sort(clf.support_)
print(res)