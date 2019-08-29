import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score

boston_dataset = datasets.load_boston()

# Check out what the keys are in a dictionary-like Bunch object
data_keys = boston_dataset.keys()

# Load the data itself and the target. Check the range for the target variable
data_x = pd.DataFrame(boston_dataset['data'])
data_y = pd.DataFrame(boston_dataset['target'])
y_range = (data_y[0].min(), data_y[0].max())

# Get to know the data and the features
description = boston_dataset['DESCR']
feature_names = boston_dataset['feature_names']

# Look at the data and check the main characteristics
num_duplicates = data_x.duplicated().sum().sum()
num_missing = data_x.isnull().sum().sum()

# Checking if any of the columns are not of numerical type
num_object_type = (data_x.dtypes == 'object').sum()

# Start building the model
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3)

y_train, y_test = np.ravel(y_train), np.ravel(y_test)


gb_params = {'loss': ('ls', 'lad', 'huber', 'quantile'), 'learning_rate': np.linspace(0.002, 1, 50),
          'max_depth': np.linspace(2, 6, 5)}
reg_GB = GradientBoostingRegressor(random_state = 24)
gb_reg = GridSearchCV(reg_GB, gb_params, cv = 5)
gb_reg.fit(X_train, y_train)

tree_reg = DecisionTreeRegressor(max_depth = 4)
ada_reg = AdaBoostRegressor(tree_reg, random_state = 24, n_estimators = 200)
ada_reg.fit(X_train, y_train)


models = {'gb': gb_reg, 'ada': ada_reg}
def compare_models(models : dict, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        explained_variance = explained_variance_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)
        mean_ae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = [explained_variance, mse, median_ae, mean_ae, r2]
    return results

results = compare_models(models, X_test, y_test)
