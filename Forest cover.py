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
