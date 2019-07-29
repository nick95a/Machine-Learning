from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier



# Step 2.
# Load the dataset
dataset = datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = pd.DataFrame(dataset.target, columns = ['Target'])
data = pd.concat([X, y] , axis = 1)
column_names = list(dataset.feature_names)
print(column_names)

assert X.isnull().sum().sum() == 0
assert y.isnull().sum().sum() == 0

assert X.duplicated().sum().sum() == 0

# Summary statistics for the data, checking dtypes for our data and seeing the shape of our data

#X.describe()
#X.info()

# Hence, no duplicates and no NaN values. All of the variables are float64 type, so no problems here
def plotter(column_names, X):
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace = 1.1,wspace = 0.4)
    for index, name in enumerate(column_names):
        ax = fig.add_subplot(10, 3, index + 1)
        ax.hist(X.iloc[:, index])
        ax.set_title(label = name)
    plt.show()

#plotter(column_names, X)

corr_X = X.corr()
data_corr = data.corr().Target.sort_values(ascending = False)

for i in range(0, len(column_names), 5):
    sns.pairplot(data = data,
                 x_vars = column_names[i:i+5],
                 y_vars = ['Target'])

#plt.show()

# Splitting the dataset into training and testing data. Test_size = 0.3 indicates that testing data should
# constitute 30% of the entire dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)


# Scaling the data. We see even from the snippet below that ranges of values for different variables vary a lot
# Standard scaler performs centering and scaling by the mean and variance of every feature in the dataset
max_values = np.transpose(X.describe()[-1:][:])
min_values = np.transpose(X.describe().iloc[3, :])


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_train))

#Step 3. Algorithm/Method selection. Below, I will explore the standard classification methods that one can get
# from sklearn.

logRegClassifier = LogisticRegression()
logRegClassifier.fit(X_train)
logRegClassifier.predict(X_test)

neighClassifier = KNeighborsClassifier()
neighClassifier.fit(X_train)
neighClassifier.predict(X_test)

svmClassifier = SVC()
svmClassifier.fit(X_train)
svmClassifier.predict(X_test)

decTreeClassifier = DecisionTreeClassifier()
decTreeClassifier.fit(X_train)
decTreeClassifier.predict(X_test)

randomForest = RandomForestClassifier()
randomForest.fit(X_train)
randomForest.predict(X_test)

gaussNB = GaussianNB()
gaussNB.fit(X_train)
gaussNB.predict(X_test)