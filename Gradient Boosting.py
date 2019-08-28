import numpy as np
import pandas as pd
from sklearn import datasets

boston_dataset = datasets.load_boston()

# Check out what the keys are in a dictionary-like Bunch object
print(boston_dataset.keys())

data_x = boston_dataset['data']
data_y = boston_dataset['target']

# Get to know the data and the features
description = boston_dataset['DESCR']
feature_names = boston_dataset['feature_names']

#