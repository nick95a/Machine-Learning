import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from scipy import sparse

datatrain = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/salary_train.csv')
dataset = pd.read_csv('/Users/nikolaymaltsev/Documents/ML/salary_test.csv')
#datatrain.apply(lambda x: x.astype(str).str.lower())
#dataset.apply(lambda x: x.astype(str).str.lower())
#datatrain = pd.concat([datatrain[col].astype(str).str.lower() for col in datatrain.columns], axis=1)
#dataset = pd.concat([dataset[col].astype(str).str.lower() for col in dataset.columns],axis = 1)

def text_transform(text):
    # Приведите тексты к нижнему регистру (text.lower()).
    text = text.map(lambda t: t.lower())

    # Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. Для такой
    # замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться
    # методом replace у DataFrame, чтобы сразу преобразовать все тексты:
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text

datatrain.iloc[:,1].fillna('nan',inplace = True)
datatrain.iloc[:,2].fillna('nan',inplace = True)
#dataset.iloc[:,1].fillna('NaN',inplace = True)
#dataset.iloc[:,2].fillna('NaN',inplace = True)


#datatrain = pd.concat([datatrain[col].replace('[^a-zA-Z0-9]', ' ',regex = True) for col in datatrain.columns],axis = 1)
#dataset = pd.concat([dataset[col].replace('[^a-zA-Z0-9]', ' ',regex = True) for col in dataset.columns],axis = 1)
datatrain['FullDescription'] = datatrain['FullDescription'].replace('[^a-zA-Z0-9]',' ',regex = True)
dataset['FullDescription'] = dataset['FullDescription'].replace('[^a-zA-Z0-9]',' ',regex = True)
#datatrain['ContractTime'] = datatrain['ContractTime'].replace('[^a-zA-Z0-9]',' ',regex = True)
#dataset['ContractTime'] = dataset['ContractTime'].replace('[^a-zA-Z0-9]',' ',regex = True)
#datatrain['SalaryNormalized'] = datatrain['SalaryNormalized'].replace('[^a-zA-Z0-9]',' ',regex = True)
#dataset['SalaryNormalized'] = dataset['SalaryNormalized'].replace('[^a-zA-Z0-9]',' ',regex = True)



vectorizer = TfidfVectorizer(min_df = 5)
dict = DictVectorizer()

X_train = vectorizer.fit_transform(text_transform(datatrain['FullDescription']))
m = dict.fit_transform(datatrain[['LocationNormalized','ContractTime']].to_dict('records'))
Y_train = datatrain['SalaryNormalized']
Xt = dict.transform(dataset[['LocationNormalized','ContractTime']].to_dict('records'))
n = vectorizer.transform(text_transform(dataset['FullDescription']))


X = hstack([X_train,m])
X_test = hstack([Xt,n])

clf = linear_model.Ridge(alpha = 1 ,random_state = 241)
clf = clf.fit(X,Y_train)
pred = clf.predict(X_test)

print(pred)


#Не понимаю почему не сходится результат.
#Ниже приложен чужой кож,который дает правильный ответ



import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack



# 1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv.

train = pandas.read_csv('/Users/nikolaymaltsev/Documents/ML/salary_train.csv')

# 2. Проведите предобработку:


def text_transform(text):
    # Приведите тексты к нижнему регистру (text.lower()).
    text = text.map(lambda t: t.lower())

    # Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. Для такой
    # замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться
    # методом replace у DataFrame, чтобы сразу преобразовать все тексты:
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text

# Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова,
# которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).

vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_transform(train['FullDescription']))

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.

enc = DictVectorizer()
X_train_cat = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и
# категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией
# scipy.sparse.hstack.

X_train = hstack([X_train_text, X_train_cat])

# 3. Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце SalaryNormalized.

y_train = train['SalaryNormalized']
model = Ridge(alpha=1,random_state=241)
model.fit(X_train, y_train)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются
# ответом на задание. Укажите их через пробел.

test = pandas.read_csv('/Users/nikolaymaltsev/Documents/ML/salary_test.csv')
X_test_text = vec.transform(text_transform(test['FullDescription']))
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print(1, '{:0.2f} {:0.2f}'.format(y_test[0], y_test[1]))