import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC

dataset = datasets.load_digits()

X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)

X.shape
# We see that there is no need to transform the data from the original shape. Normally, when processing
# an image dataset some flattening is necessary. This can be done as below:
new_dataset = np.array(dataset.images)
new_dataset = np.reshape(new_dataset, newshape = (new_dataset.shape[0], -1))
new_dataset.shape
