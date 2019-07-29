from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Step 1.
# Load the dataset
dataset = datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = pd.DataFrame(dataset.target)

assert X.isnull().sum().sum() == 0
assert y.isnull().sum().sum() == 0

assert X.duplicated().sum().sum() == 0

# Summary statistics for the data, checking dtypes for our data and seeing the shape of our data
X.describe()
X.info()
# Hence, no duplicates and no NaN values. All of the variables are float64 type, so no problems here


max_values = np.transpose(X.describe()[-1:][:])
