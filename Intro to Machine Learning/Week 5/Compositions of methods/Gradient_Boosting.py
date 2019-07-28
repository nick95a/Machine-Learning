import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/Users/nikolaymaltsev/Documents/учеба/ML/gbm_data.csv')

X = dataset.iloc[:,1:].values
Y = dataset['Activity'].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.8,random_state = 241)
set = [1,0.5,0.3,0.2,0.1]

res = []
def sigma(Y_pred):
    return 1/(1 + math.exp(-Y_pred))

def logLossMetric(model,X,Y):
   for score in model.staged_decision_function(X):
       res.append(log_loss(Y,[sigma(Y_pred) for Y_pred in score]))
   return res


def plotLogLoss(learning_rate,test_loss,train_loss):
    plt.show()
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])

    min_loss = min(test_loss)
    min_loss_index = test_loss.index(min_loss)

    return min_loss,min_loss_index

def build(learning_rate):
    model = GradientBoostingClassifier(n_estimators = 250,verbose = True,random_state = 241,learning_rate = learning_rate)
    model.fit(X_train,Y_train)

    train_loss = logLossMetric(model,X_train,Y_train)
    test_loss = logLossMetric(model,X_test,Y_test)
    return plotLogLoss(learning_rate,test_loss,train_loss)

min_loss_results = {}
for learning_rate in set:
    min_loss_results[learning_rate] = build(learning_rate)




min_loss_value,min_loss_index = min_loss_results[0.2]
print(min_loss_value,min_loss_index)

clf = RandomForestClassifier(random_state = 241,n_estimators = min_loss_index)
clf.fit(X_train,Y_train)
y_pred = clf.predict_proba(X_test)[:,1]
test_loss = log_loss(Y_test,y_pred)
print(test_loss)



