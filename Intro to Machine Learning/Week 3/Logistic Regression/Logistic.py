import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score


data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/data_logistic.csv',header = None)

X = np.array(data.iloc[:,1:])
Y = np.array(data.iloc[:,0])

a = 0
w1 = 0
w2 = 0
k = 0.1
C = 10
l = len(Y)
x1 = X[:,0]
x2 = X[:,1]
n = 0
d = 10


y = []

while n < 10000 and d > 0.00001 :
  v1 = w1
  v2 = w2
  a = 1 + np.exp(-Y*(w1 * x1 + w2 * x2))
  w1 = w1 + (k/l)*(np.sum(Y*x1*(1 - 1/a))) - k*C*w1
  w2 = w2 + (k/l)*(np.sum(Y*x2*(1 - 1/a))) - k*C*w2
  d = np.sqrt((w1 - v1)**2 + (w2-v2)**2)
  n = n + 1
  y.append(a)

y = [item for sublist in y for item in sublist]
y[:] = [x - 1 for x in y]
print(type(y))
zero = []
np.array(zero)
zero.append(np.zeros((39,),dtype=np.int))
zero1 = [item for sublist in zero for item in sublist]
np.append(Y,zero1)
y = np.asarray(y)

print(roc_auc_score(y, Y))
