import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/classification.csv')
dataset = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/scores.csv')
true = data.iloc[:,0]
pred = data.iloc[:,1]

t = dataset.iloc[:,0]
logreg = dataset.iloc[:,1]
svm = dataset.iloc[:,2]
knn = dataset.iloc[:,3]
tree = dataset.iloc[:,4]
mat = confusion_matrix(true,pred).ravel()

report = classification_report(true,pred)


roc_1 = roc_auc_score(t,logreg)
roc_2 = roc_auc_score(t,svm)
roc_3 = roc_auc_score(t,knn)
roc_4 = roc_auc_score(t,tree)

prec,recall,threshold = precision_recall_curve(t,knn)
prec1 = prec[:40]
print(max(prec1))
print(dataset)