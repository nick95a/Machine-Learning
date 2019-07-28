"""
Замечание:
Много повторяющегося кода, но нужно учитывать, что было три разных файла:
Бустинг и две лог.регрессии

Также нужно прописать путь для файлов features.csv и features_test.csv


Код для задания номер один с бустингом
"""


#Выполняем все необходимые загрузки
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import time
import datetime


dataset = pd.read_csv('.../features.csv')
dataset_test = pd.read_csv('.../features_test.csv')

start_time = datetime.datetime.now()


#Удаляем признаки,связанные с итогами матча
X = dataset.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'], axis = 1)

#Считаем количество пропусков в столбцах
l = X.isnull().sum(axis = 0).tolist()
newl = pd.Series(l)
c = newl.nonzero()
#находим столбцы, в которых были пропуски
features_nan = X.iloc[:,[83,84,85,86,87,88,89,94,95,96,97,102]]
cols_nan = list(features_nan)

#заменяем пропуски NaN на 0
X.fillna(0,inplace = True)

#Выбираем целевую переменную
y = dataset.loc[:,'radiant_win']

#Разбиение для кросс-валидации
kf = KFold(n_splits = 5 , shuffle = True,random_state= 42)
est = [10,20,30]


auc = []
for train,test in kf.split(X,y):
        grdboost = GradientBoostingClassifier(n_estimators=100, random_state=42,max_depth = 1)
        X_train,X_test = X.iloc[train,:],X.iloc[test,:]
        y_train,y_test = y.iloc[train],y.iloc[test]
        pred = grdboost.fit(X_train,y_train).predict_proba(X_test)[:,1]
        auc.append(roc_auc_score(y_test,pred))

print(auc)

#Замеряем время работы кода
print('Time elapsed:', datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

"""
Код для задания 2 с логистической регрессией
Подзадания 1-4
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

dataset = pd.read_csv('/Users/nikolaymaltsev/Dropbox/учеба/ML/features.csv', index_col = 'match_id')
dataset_test = pd.read_csv('/Users/nikolaymaltsev/Dropbox/учеба/ML/features_test.csv')
scaler = StandardScaler()

#Удаляем признаки,связанные с итогами матча
X = dataset
dataset_test.fillna(0,inplace = True)
#заменяем пропуски NaN на 0
X.fillna(0,inplace = True)

#Выбираем целевую переменную
y = dataset.loc[:,'radiant_win']
#Разбиение для кросс-валидации
kf = KFold(n_splits = 5 , shuffle = True,random_state = 42)



#Получаем мешок слов
X_p = np.zeros((X.shape[0],112))
for i, match_id in enumerate(X.index):
    for p in range(5):
        X_p[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_p[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

#Удаляем признаки
X_pick = pd.DataFrame(X_p,index = X.index)
X_new = X.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], axis = 1)
X_new_new = X_new.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'], axis = 1)
X_new_new.fillna(0,inplace = True)

#Сшиваем два фрейма
x = pd.concat([X_new_new,X_pick],axis = 1)



#Проводим разбиение на подвыборки и тренируем модель логистической регрессии
#Получаем результат метрики качества и выводим его
auc = []
for train,test in kf.split(X,y):
        logreg = LogisticRegression(random_state=42, C = 0.1)
        x_train,x_test = x.iloc[train,:],x.iloc[test,:]
        y_train,y_test = y.iloc[train],y.iloc[test]
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)
        pred = logreg.fit(x_train_scaled,y_train).predict_proba(x_test_scaled)[:,1]
        score = roc_auc_score(y_test,pred)
        auc.append(score)

print(auc)


#Замеряем время работы кода
print('Time elapsed:', datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

"""
Код для задания 2 с логистической регрессией
Последнее задание с тестовой выборкой
"""

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

dataset = pd.read_csv('/Users/nikolaymaltsev/Dropbox/учеба/ML/features.csv', index_col = 'match_id')
dataset_test = pd.read_csv('/Users/nikolaymaltsev/Dropbox/учеба/ML/features_test.csv',index_col = 'match_id')
scaler = StandardScaler()

#Удаляем признаки,связанные с итогами матча
H = dataset_test
dataset_test.fillna(0,inplace = True)
#заменяем пропуски NaN на 0
H.fillna(0,inplace = True)


#Удаляем признаки,связанные с итогами матча
X = dataset
dataset_test.fillna(0,inplace = True)
#заменяем пропуски NaN на 0
X.fillna(0,inplace = True)

#Выбираем целевую переменную

#
y = dataset.loc[:,'radiant_win']

#Разбиение для кросс-валидации
kf = KFold(n_splits = 5 , shuffle = True,random_state = 42)


#Настраиваем dataset с тестовой выборкой

H_p = np.zeros((H.shape[0],112))
for i, match_id in enumerate(H.index):
    for p in range(5):
        H_p[i, H.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        H_p[i, H.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

H_pick = pd.DataFrame(H_p,index = H.index)
H_new = H.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], axis = 1)
H_new.fillna(0,inplace = True)
h = pd.concat([H_new,H_pick],axis = 1)


#dataset с тренировочной выборкой

X_p = np.zeros((X.shape[0],112))
for i, match_id in enumerate(X.index):
    for p in range(5):
        X_p[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_p[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1


X_pick = pd.DataFrame(X_p,index = X.index)
X_new = X.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], axis = 1)
X_new_new = X_new.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'], axis = 1)
X_new_new.fillna(0,inplace = True)

x = pd.concat([X_new_new,X_pick],axis = 1)


#pred выводит значения прогнозов при разных разбиениях
pred = []
for train,test in kf.split(X,y):
        logreg = LogisticRegression(random_state=42, C = 0.1)
        x_train,x_test = x.iloc[train,:],x.iloc[test,:]
        y_train,y_test = y.iloc[train],y.iloc[test]
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)
        h_test_scaled = scaler.fit_transform(h)
        pred.append(logreg.fit(x_train_scaled,y_train).predict_proba(h_test_scaled)[:,1])


#Выводим максимальное и минимальное значение прогноза на тестовой выборке
min = []
max = []
for i in range(5):
    max.append(np.amax(pred[i]))
    min.append(np.amin(pred[i]))

print(min,max)







