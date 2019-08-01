import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset = datasets.load_digits()

X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)

X.shape
# We see that there is no need to transform the data from the original shape. Normally, when processing
# an image dataset some flattening is necessary. This can be done as below:
new_dataset = np.array(dataset.images)
new_dataset = np.reshape(new_dataset, newshape = (new_dataset.shape[0], -1))
new_dataset.shape

y_series = y.iloc[:,0]
y_series.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# visualise the dataset of images

images = list(zip(dataset.images, dataset.target))
for index, (image, label) in enumerate(images[:4], 1):
    plt.subplot(2, 4, index)
    plt.axis('off')
    plt.imshow(image, cmap = "Greys")
    plt.title("Training image for number {}".format(label))

plt.show()