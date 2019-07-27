import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from catboost import Pool, CatBoostRegressor

dataset = data.fetch_california_housing()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = pd.DataFrame(dataset.target)
cal_housing = pd.concat([X,y], axis = 1)
cal_housing = cal_housing.rename({0 : "Response"}, axis = 1)

# check general info
X.info()
y.info()


# All variables are of type float so regression seems to be a valid model from the start
# Checking dtypes is important to see whether any preprocessing for the variables is needed, i.e. enconding string variables
# as categorical.

# Check for duplicates and missing data
num_duplicates = X.duplicated().sum()
num_missing = X.isnull().sum().sum()
# We found out that the data is pretty clean and no duplicates or missing values are present. All of the data is also of tyoe float64

"""
Splitting the data into training and testing parts. 

In our experiments we cannot learn the model and then test it on exactly the same data points.
That means that we have to partition the data so that the part of the dataset on which we are learning the model 
and the part on which we train the model do not overlap. Otherwise, the predictions of the model will be unreliable
as the so-called 'overfitting' takes place where the model already learnt the underlying patterns in the dataset
and then when tested on it, the model indeed performs really well.

The approach is to perform cross-validation procedure on the dataset. The dataset is still split into training
and testing parts, but the validation part is not explicitly partitioned. For example, the k-fold cross-validation
implies that the training dataset is split into k sets and each set will be (k - 1) times in the training partition and
1 time it will be used to validate the model. The measure used to describe the procedure is taken as the average
over all the iterations. Then the final model is tested on the prepartitioned testing part of the 
original dataset.
"""

# Perform some visual analysis of the data

X.hist(bins = 50, figsize = (20, 15))
plt.show()
plt.close()
# inspect boxplots
def boxplot(X):
    fig = plt.figure(figsize = (30,15))
    for index, col in enumerate(list(X.columns)):
        ax = fig.add_subplot(2,4, index + 1)
        ax.set_title(col)
        ax.boxplot(X[col])
    plt.show()

def scatterplot(X,y):
    fig = plt.figure(figsize=(20, 20))
    for index, col in enumerate(list(X.columns)):
        ax = fig.add_subplot(2, 4, index + 1)
        ax.set_title(col)
        ax.scatter(X[col],y)
    plt.show()

boxplot(X)
plt.close()
scatterplot(X, y)
plt.close()

#scatterplot(X,y) to see how median house price is distributed between geographical coordinates
cal_housing.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4, label="Median house price", figsize = (15,8),
        c='Response', cmap=plt.get_cmap("jet"),colorbar=True)

plt.close()

corr_values_Response = cal_housing.corr().Response.sort_values(ascending = False)
corr_values_features = X.corr()
print(corr_values_Response)
# Below is a heatmap of correlations that show which features seem to be correlated with one other and to what extent
ax = sns.heatmap(data = corr_values_features, cmap = sns.color_palette('coolwarm', 8))
plt.show()

"""
From simple exploratory analysis a few little patterns may be spotted:
1) Latitude and Longitude seem to be negatively correlated in our data. That actually confirms what we saw earlier in the plot

2) Median Income is correlated with Average Rooms 
Off the top of the head thought is that wealthier people may and do on average afford themselves more spacious places to live

3) Also, Population and House Age seem to be interrelated in some way
 
4) Median Income, by far, has the greatest strength of correlation with the target variable. 
Intuitively, we may feel that it is okay to assume the following relationship Higher Median Income => Higher Median House Price and
out intuition may indeed be correct, but it is important to remember that old statistical saying of "Correlation does
NOT imply causation"

5) Also, Average Rooms has some degree of correlation with the target variable(Potential multicollinearity - point 2 of the list))
and Latitude has a higher degree of correlation that Longitude. 

These are all of the relationships that we may want to think about and analyse both statistically and in our heads.
"""



# Creating the basic regression models for which gridsearch and cross validation will be used
linearReg = LinearRegression()
lassoReg = Lasso()
ridgeReg = Ridge()
elasticReg = ElasticNet()
parameters = {'alpha' : np.linspace(0.1, 10, 50)}
paramElastic = {'alpha' : np.linspace(0.1, 10, 50), 'l1_ratio' : np.linspace(0.01, 1, 10)}
n_folds = 10
r2_scores = []
reg_names = ["Linear","Lasso", "Ridge", "Elastic", "Catboost"]
results = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 24)


# Linear regression
linearReg.fit(X_train,y_train)
linearPredictions = linearReg.predict(X_test)
# Metrics on the test test
median_error_linear = median_absolute_error(y_test, linearPredictions)
r2_squared_linear = r2_score(y_test, linearPredictions)
r2_scores.append(r2_squared_linear)

# Lasso regression
lassoReg = GridSearchCV(lassoReg, parameters, cv = n_folds)
lassoReg.fit(X_train, y_train)
#lassoReg.best_params_
lassoPredictions = lassoReg.predict(X_test)
# Metrics on the test test
median_error_lasso = median_absolute_error(y_test, lassoPredictions)
r2_squared_lasso = r2_score(y_test, linearPredictions)
r2_scores.append(r2_squared_lasso)

# Ridge regression
ridgeReg = GridSearchCV(ridgeReg, parameters)
ridgeReg.fit(X_train, y_train)
#ridgeReg.best_params_
ridgePredictions = ridgeReg.predict(X_test)
# Metrics on the test test
median_error_ridge = median_absolute_error(y_test, ridgePredictions)
r2_squared_ridge = r2_score(y_test, linearPredictions)
r2_scores.append(r2_squared_ridge)


elasticReg = GridSearchCV(elasticReg, paramElastic)
elasticReg.fit(X_train, y_train)
elasticPredictions = elasticReg.predict(X_test)
median_error_elastic = median_absolute_error(y_test, elasticPredictions)
r2_squared_elastic = r2_score(y_test, elasticPredictions)
r2_scores.append(r2_squared_elastic)


# Catboost regressor. Takes some time to train.
lrate = {'learning_rate' : np.linspace(0.01,0.1, 10)}
train_data = Pool(X_train, y_train)
cat = CatBoostRegressor(iterations = 10, depth = 2)
grid_search_results = cat.grid_search(param_grid = lrate, X = train_data)
params = grid_search_results[0]
r2_cat = cat.score(Pool(X_test, y_test))
r2_scores.append(r2_cat)
results = dict(zip(reg_names,r2_scores))

# Clearly, the linear models here are not done really well because they produce similar results. Catboost performs
# much better but its internal mechanics is very different to the other models(uses gradient boosting) and it takes
# much more time to train. The project is unfinished and I intend to come back and rewrite parts of it when I learn
# more about ML if I have time.