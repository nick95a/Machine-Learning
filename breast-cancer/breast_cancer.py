from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib as plt


# Load the dataset
dataset = datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = pd.DataFrame(dataset.target)
