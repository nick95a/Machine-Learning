"""
The dataset for the little experiment below was taken from the UCI repo. Link: https://archive.ics.uci.edu/ml/datasets/Covertype
The dataset explanation is taken from the website:


Predicting forest cover type from cartographic variables only (no remotely sensed data). 
The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. 
Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types). 
This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. 
Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value. 
As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4). 
The Rawah and Comanche Peak areas would tend to be more typical of the overall dataset than either the Neota or Cache la Poudre, due to their assortment of tree species and range of predictive variable values (elevation, etc.) Cache la Poudre would probably be more unique than the others, due to its relatively low elevation range and species composition.

"""

import sklearn.datasets as datasets
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

dataset = datasets.fetch_covtype(shuffle = True)
X = dataset.data[:,:]
y = dataset.target

# Применение train_test_split для разбиения, чтобы проверить гипотезу о том, что происходит явное переобучение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 24)

kf = KFold(n_splits = 10)
criterion = ['gini', 'entropy']
split = 1
average_list = ['micro','samples', 'macro', 'weighted']
average = 'weighted'
# можно еще попробовать разные значения параметра average при расчете метрики качества

def kfoldClassifier(X,y, criterion, average):
    for criteria in criterion:
        for train, test in kf.split(X, y):
            clf = tree.DecisionTreeClassifier(criterion = criteria)
            clf = clf.fit(X[train], y[train])
            pred = clf.predict(X[test])
            precision = metrics.precision_score(y[test], pred, average = average)
            recall = metrics.recall_score(y[test], pred, average = average)
            f1_score = (2 * precision * recall) / (precision + recall)
            print("Precision,recall, f1_score for sample {} with {} is equal to {:.2f},{:.2f}, {:.2f}".format(
                split,criteria, precision, recall, f1_score))
            split += 1
        split = 1


def traintestClassifier(X, y, criterion, average):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    metrics_results = {}
    for criteria in criterion:
        clf = tree.DecisionTreeClassifier(criterion = criteria)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        precision = metrics.precision_score(y_test, pred, average = average)
        recall = metrics.recall_score(y_test, pred, average = average)
        f1_score = (2 * precision * recall) / (precision + recall)
        scores = [precision, recall, f1_score]
        metrics_results[criteria] = scores
    return metrics_results

def printDict(dict):
    for key, value in dict.items():
        print(key, "Precision, recall, f1_score {}".format(value))

resultClassic = traintestClassifier(X, y, criterion, average)
printDict(resultClassic)
