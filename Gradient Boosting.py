import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

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

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3)

params = {'loss': ('ls', 'lad', 'huber', 'quantile'), 'learning_rate': np.linspace(0, 1, 50),
          'max_depth': np.linspace(2, 6, 5)}
regressor = GradientBoostingRegressor(random_state = 24)
grid = GridSearchCV(regressor, params, cv = 5)
regressor.fit(X_train, y_train)

'''
Добавить проверку isnull, duplicates
Посмотреть dtypes, прописать, что раз GBDT, то пофигу тип, он и так нормально работает
'''