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
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score


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
X_test = pd.DataFrame(scaler.transform(X_test))
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]
#Step 3. Algorithm/Method selection. Below, I will explore the standard classification methods that one can get
# from sklearn.
# Note: At the moment, no grid search over hyperparameters is present. Planning to add this a bit later. In any case
# check out sklearn, for example, GridSearchCV and read the docs.

# Look at penalty and C parameters for regularization
logRegClassifier = LogisticRegression(random_state = 24)
logRegClassifier.fit(X_train, y_train)
predLog = logRegClassifier.predict(X_test)
probaLog = logRegClassifier.predict_proba(X_test)
probaLog = probaLog[:,1]


# Look to play around with the number of neighbors, metric and etc.
neighClassifier = KNeighborsClassifier(random_state = 24)
neighClassifier.fit(X_train, y_train)
predNeigh = neighClassifier.predict(X_test)
probaNeigh = neighClassifier.predict_proba(X_test)
probaNeigh = probaNeigh[:, 1]


# Look to optimize C and kernel choice for SVC
svmClassifier = SVC(probability = True, random_state = 24)
svmClassifier.fit(X_train, y_train)
predSVC = svmClassifier.predict(X_test)
probaSVC = svmClassifier.predict_proba(X_test)
probaSVC = probaSVC[:, 1]

# Look to change criterion, max_depth, min_samples_leaf and etc.
decTreeClassifier = DecisionTreeClassifier(random_state = 24)
decTreeClassifier.fit(X_train, y_train)
predTree = decTreeClassifier.predict(X_test)
probaTree = decTreeClassifier.predict_proba(X_test)
probaTree = probaTree[:, 1]


# Look to change n_estimators (trees), criterion, max_depth and others
randomForest = RandomForestClassifier(random_state = 24)
randomForest.fit(X_train, y_train)
predForest = randomForest.predict(X_test)
probaForest = randomForest.predict_proba(X_test)
probaForest = probaForest[:, 1]


gaussNB = GaussianNB(random_state = 24)
gaussNB.fit(X_train, y_train)
predGauss = gaussNB.predict(X_test)
probaGauss = gaussNB.predict_proba(X_test)
probaGauss = probaGauss[:, 1]


names = ["Logistic", "Gaussian", "RanForest", "DecTree", "SVC", "KNN"]
predictions = {"Logistic" : [probaLog, predLog], "Gaussian" : [probaGauss, predGauss], "RanForest" : [probaForest, predForest], "DecTree" : [probaTree, predTree],\
         "SVC": [probaSVC, predSVC],"KNN" : [probaNeigh, predNeigh]}

auc_results = {} # store ROC-AUC results
accuracy_results = {}
class_report_results = {}

def metrics_calculator(y_test, predictions, names):
    for name in names:
        auc_score = roc_auc_score(y_test, predictions[name][0])
        accuracy = accuracy_score(y_test, predictions[name][1])
        class_report = classification_report(y_test, predictions[name][1])
        auc_results[name] = auc_score
        accuracy_results[name] = accuracy
        class_report_results[name] = class_report

metrics_calculator(y_test, predictions, names)
# To access classification report for say logistic regression, one needs to do this:
# class_report_results["Logistic"]. This contains precision, recall, f1-score and the size of the support cols

