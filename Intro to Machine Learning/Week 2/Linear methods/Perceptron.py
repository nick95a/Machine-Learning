import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

Train = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/perceptron-train.csv',header = None)
Test = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/perceptron-test.csv',header = None)


X_train = np.array(Train.iloc[:,1:])
X_test = np.array(Test.iloc[:,1:])
Y_train = np.array(Train.iloc[:,0])
Y_test = np.array(Test.iloc[:,0])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_1 = Perceptron(random_state= 241)
clf_1 = clf_1.fit(X_train_scaled,Y_train)
pred = clf_1.predict(X_test_scaled)
accuracy_1 = accuracy_score(Y_test,pred)

clf_2 = Perceptron(random_state= 241)
clf_2 = clf_2.fit(X_train,Y_train)
pred = clf_2.predict(X_test)
accuracy_2 = accuracy_score(Y_test,pred)

print(accuracy_1)
print(accuracy_2)

print(accuracy_2-accuracy_1)