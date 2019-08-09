import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

dataset = datasets.load_digits()

X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)
y = y.iloc[:, 0]

X.shape
# We see that there is no need to transform the data from the original shape. Normally, when processing
# an image dataset some flattening is necessary. This can be done as below:
new_dataset = np.array(dataset.images)
new_dataset = np.reshape(new_dataset, newshape = (new_dataset.shape[0], -1))
new_dataset.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
# visualise the dataset of images

images = list(zip(dataset.images, dataset.target))
for index, (image, label) in enumerate(images[:4], 1):
    plt.subplot(2, 4, index)
    plt.axis('off')
    plt.imshow(image, cmap = "Greys")
    plt.title("Training image for number {}".format(label))

params = {'C': np.linspace(0.1,2,10), 'kernel' : ['rbf', 'linear','sigmoid']}

# gamma is small since we do not need the classifier to overfit.
classifier = SVC(gamma = 0.0001)
classifier = GridSearchCV(classifier, param_grid = params,cv = 5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
clf_report = classification_report(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
print(clf_report, accuracy)

# TO-DO: ROC-AUC for this problem, plot them.
"""
To plot roc_auc for multuclass classification one needs to binarize the variables first.
Then use the OneVsRestClassifier
y = label_binarize(y, classes = list(range(10)))

fpr = {}
tpr = {}
roc_auc = {}
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
"""

#To be continued