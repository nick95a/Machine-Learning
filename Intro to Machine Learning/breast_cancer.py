from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Step 1.
# Load the dataset
dataset = datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = pd.DataFrame(dataset.target)
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





# Scaling the data
max_values = np.transpose(X.describe()[-1:][:])
