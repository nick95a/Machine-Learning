from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


dataframe = pd.DataFrame([[0.0,14.41],[0.3,12.24],[0.5,14.43],[0.8,14.88],[1.1,16.29],[1.3,20.2],[1.6,19.04],[1.8,23.5],[2.1,23.35],[2.4,24.61],[2.6,28.61],[2.9,28.58],[3.2,30.93],[3.4,33.52],[3.7,38.81],[3.9,42.88]])

def reshape(vectors):
    '''
    Reshapes list of pandas Series of size (n,) into (n, 1)
    :param vector:
    :return:
    '''
    vectors = [item.values.reshape(-1, 1) for item in vectors]
    for item in vectors:
        assert item.shape == (len(item), 1)
    return vectors

y = dataframe.iloc[:,1]
x = dataframe.iloc[:,0]


data = train_test_split(x, y, test_size = 0.4)
data = reshape(data)
X_train, X_test, y_train, y_test = data

'''
Особого смысла делать здесь кросс-валидацию и подбор коэффициентов нет, но для полноты я привел.
Выбор моделей слишком очевидный, но я решил сделать регрессию, поэтому не видел особого смысла пробовать какие-то вычурные
модели регрессии и взял базовые для быстроты.
'''
reg = LinearRegression(normalize = True)
lasso = LassoCV(cv = 5, n_alphas = 20)
ridge = RidgeCV(cv = 5, alphas = np.linspace(-5, 5, 20))

def pipeline(model, data: list):
    '''
    Builds a pipeline for fitting, calculating R-squared and prediction 5-year salary
    '''
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r_squared = r2_score(y_test, predictions)
    pred_5_year_salary = model.predict([[5]])[0]
    print("Current model is {}".format(str(model)))
    print("R-squared is {}".format(r_squared))
    print("Salary prediction for year 5 is {}".format(pred_5_year_salary))
    print()
    return r_squared, pred_5_year_salary

def winner(models: tuple, data: list):
    max_r2_score = 0
    prediction = 0
    best_model = ''
    for model in models:
        curr_r2, curr_5_year_prediction = pipeline(model, data)
        if curr_r2 > max_r2_score:
            max_r2_score = curr_r2
            salary_prediction = curr_5_year_prediction
            best_model = model
    return print("Best model is {0} with R-squared of {1} and 5-year salary prediction of {2}".format(best_model, max_r2_score, salary_prediction))

models = (reg, lasso, ridge)

winner(models, data)