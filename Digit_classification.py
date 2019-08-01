import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
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

y = y.iloc[:,0]
y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# visualise the dataset of images

images = list(zip(dataset.images, dataset.target))
for index, (image, label) in enumerate(images[:4], 1):
    plt.subplot(2, 4, index)
    plt.axis('off')
    plt.imshow(image, cmap = "Greys")
    plt.title("Training image for number {}".format(label))

params = {'C': np.linspace(0.1,2,20), 'kernel' : ['rbf', 'linear', 'rbf','sigmoid']}
classifier = SVC()
classifier = GridSearchCV(classifier, param_grid = params,cv = 5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probas = classifier.predict_proba(X_test)
roc_auc_score(y_test, probas)
