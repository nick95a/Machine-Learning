{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:04.021789Z",
     "start_time": "2020-05-21T10:13:04.013121Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn import datasets\n",
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Download data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:04.451468Z",
     "start_time": "2020-05-21T10:13:04.427261Z"
    }
   },
   "outputs": [],
   "source": [
    "dataset = datasets.load_diabetes(return_X_y = False)\n",
    "cols = dataset.feature_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:04.636311Z",
     "start_time": "2020-05-21T10:13:04.610984Z"
    }
   },
   "outputs": [],
   "source": [
    "X, y = datasets.load_diabetes(return_X_y = True)\n",
    "X = pd.DataFrame(X, columns = cols)\n",
    "y = pd.Series(y, name = 'target')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:04.816427Z",
     "start_time": "2020-05-21T10:13:04.811419Z"
    }
   },
   "outputs": [],
   "source": [
    "def print_dataset_descr(dataset):\n",
    "    \n",
    "    try:\n",
    "        \n",
    "        import pprint\n",
    "        pp = pprint.PrettyPrinter()\n",
    "        pp.pprint(dataset['DESCR'])\n",
    "        \n",
    "    except NameError:\n",
    "        \n",
    "        print('Cannot import pprint')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:05.020431Z",
     "start_time": "2020-05-21T10:13:05.006059Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('.. _diabetes_dataset:\\n'\n",
      " '\\n'\n",
      " 'Diabetes dataset\\n'\n",
      " '----------------\\n'\n",
      " '\\n'\n",
      " 'Ten baseline variables, age, sex, body mass index, average blood\\n'\n",
      " 'pressure, and six blood serum measurements were obtained for each of n =\\n'\n",
      " '442 diabetes patients, as well as the response of interest, a\\n'\n",
      " 'quantitative measure of disease progression one year after baseline.\\n'\n",
      " '\\n'\n",
      " '**Data Set Characteristics:**\\n'\n",
      " '\\n'\n",
      " '  :Number of Instances: 442\\n'\n",
      " '\\n'\n",
      " '  :Number of Attributes: First 10 columns are numeric predictive values\\n'\n",
      " '\\n'\n",
      " '  :Target: Column 11 is a quantitative measure of disease progression one '\n",
      " 'year after baseline\\n'\n",
      " '\\n'\n",
      " '  :Attribute Information:\\n'\n",
      " '      - Age\\n'\n",
      " '      - Sex\\n'\n",
      " '      - Body mass index\\n'\n",
      " '      - Average blood pressure\\n'\n",
      " '      - S1\\n'\n",
      " '      - S2\\n'\n",
      " '      - S3\\n'\n",
      " '      - S4\\n'\n",
      " '      - S5\\n'\n",
      " '      - S6\\n'\n",
      " '\\n'\n",
      " 'Note: Each of these 10 feature variables have been mean centered and scaled '\n",
      " 'by the standard deviation times `n_samples` (i.e. the sum of squares of each '\n",
      " 'column totals 1).\\n'\n",
      " '\\n'\n",
      " 'Source URL:\\n'\n",
      " 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html\\n'\n",
      " '\\n'\n",
      " 'For more information see:\\n'\n",
      " 'Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) '\n",
      " '\"Least Angle Regression,\" Annals of Statistics (with discussion), 407-499.\\n'\n",
      " '(https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)')\n"
     ]
    }
   ],
   "source": [
    "print_dataset_descr(dataset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:06.140997Z",
     "start_time": "2020-05-21T10:13:06.135059Z"
    }
   },
   "outputs": [],
   "source": [
    "data = pd.concat([X, y], axis = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Количество пропусков и дубликатов"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:06.915386Z",
     "start_time": "2020-05-21T10:13:06.902744Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Количество пропусков в данных\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "age       0\n",
       "sex       0\n",
       "bmi       0\n",
       "bp        0\n",
       "s1        0\n",
       "s2        0\n",
       "s3        0\n",
       "s4        0\n",
       "s5        0\n",
       "s6        0\n",
       "target    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('Количество пропусков в данных')\n",
    "print()\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:07.383244Z",
     "start_time": "2020-05-21T10:13:07.370972Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Количество дубликатов: 0\n"
     ]
    }
   ],
   "source": [
    "print('Количество дубликатов: {0}'.format(data.duplicated().sum()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Наблюдение\n",
    "\n",
    "Пропусков и дубликатов нет, датасет, видимо, очищен до нас, поэтому можем двигаться дальше."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Графики"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:09.435684Z",
     "start_time": "2020-05-21T10:13:09.431861Z"
    }
   },
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Посмотрим на распределение целевой переменной\n",
    "\n",
    "Видим, что у целевой переменной есть ярковыраженная смещенность или positive skew. Таким образом, мат.ожидание целевой переменной будет больше значения медианы целевой переменной. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:10.095675Z",
     "start_time": "2020-05-21T10:13:09.822146Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plt.hist(data['target'], bins = 15, histtype = 'bar')\n",
    "sns.distplot(data['target'], bins = 15, color = 'orange')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Наблюдение\n",
    "\n",
    "Из данных можно выделить примерно три группы на глаз: группа со значением target от 50 до 120 примерно, потом от 120 до 275 и третья группа - 275+. При желании можно потом попробовать решить задачу классификации по этим, перекодировав переменные соответствующим образом по категориям."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Есть ли стат выбросы в target?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:11.548887Z",
     "start_time": "2020-05-21T10:13:11.366771Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.boxplot(data['target'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-19T06:49:03.127644Z",
     "start_time": "2020-05-19T06:49:03.121511Z"
    }
   },
   "source": [
    "#### Наблюдение\n",
    "\n",
    "Визуально стат выбросов в целевой переменной нет."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:14:59.098514Z",
     "start_time": "2020-05-21T10:14:59.071269Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>bp</th>\n",
       "      <th>s1</th>\n",
       "      <th>s2</th>\n",
       "      <th>s3</th>\n",
       "      <th>s4</th>\n",
       "      <th>s5</th>\n",
       "      <th>s6</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>0.038076</td>\n",
       "      <td>0.050680</td>\n",
       "      <td>0.061696</td>\n",
       "      <td>0.021872</td>\n",
       "      <td>-0.044223</td>\n",
       "      <td>-0.034821</td>\n",
       "      <td>-0.043401</td>\n",
       "      <td>-0.002592</td>\n",
       "      <td>0.019908</td>\n",
       "      <td>-0.017646</td>\n",
       "      <td>151.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>-0.001882</td>\n",
       "      <td>-0.044642</td>\n",
       "      <td>-0.051474</td>\n",
       "      <td>-0.026328</td>\n",
       "      <td>-0.008449</td>\n",
       "      <td>-0.019163</td>\n",
       "      <td>0.074412</td>\n",
       "      <td>-0.039493</td>\n",
       "      <td>-0.068330</td>\n",
       "      <td>-0.092204</td>\n",
       "      <td>75.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>0.085299</td>\n",
       "      <td>0.050680</td>\n",
       "      <td>0.044451</td>\n",
       "      <td>-0.005671</td>\n",
       "      <td>-0.045599</td>\n",
       "      <td>-0.034194</td>\n",
       "      <td>-0.032356</td>\n",
       "      <td>-0.002592</td>\n",
       "      <td>0.002864</td>\n",
       "      <td>-0.025930</td>\n",
       "      <td>141.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>-0.089063</td>\n",
       "      <td>-0.044642</td>\n",
       "      <td>-0.011595</td>\n",
       "      <td>-0.036656</td>\n",
       "      <td>0.012191</td>\n",
       "      <td>0.024991</td>\n",
       "      <td>-0.036038</td>\n",
       "      <td>0.034309</td>\n",
       "      <td>0.022692</td>\n",
       "      <td>-0.009362</td>\n",
       "      <td>206.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>0.005383</td>\n",
       "      <td>-0.044642</td>\n",
       "      <td>-0.036385</td>\n",
       "      <td>0.021872</td>\n",
       "      <td>0.003935</td>\n",
       "      <td>0.015596</td>\n",
       "      <td>0.008142</td>\n",
       "      <td>-0.002592</td>\n",
       "      <td>-0.031991</td>\n",
       "      <td>-0.046641</td>\n",
       "      <td>135.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>437</td>\n",
       "      <td>0.041708</td>\n",
       "      <td>0.050680</td>\n",
       "      <td>0.019662</td>\n",
       "      <td>0.059744</td>\n",
       "      <td>-0.005697</td>\n",
       "      <td>-0.002566</td>\n",
       "      <td>-0.028674</td>\n",
       "      <td>-0.002592</td>\n",
       "      <td>0.031193</td>\n",
       "      <td>0.007207</td>\n",
       "      <td>178.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>438</td>\n",
       "      <td>-0.005515</td>\n",
       "      <td>0.050680</td>\n",
       "      <td>-0.015906</td>\n",
       "      <td>-0.067642</td>\n",
       "      <td>0.049341</td>\n",
       "      <td>0.079165</td>\n",
       "      <td>-0.028674</td>\n",
       "      <td>0.034309</td>\n",
       "      <td>-0.018118</td>\n",
       "      <td>0.044485</td>\n",
       "      <td>104.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>439</td>\n",
       "      <td>0.041708</td>\n",
       "      <td>0.050680</td>\n",
       "      <td>-0.015906</td>\n",
       "      <td>0.017282</td>\n",
       "      <td>-0.037344</td>\n",
       "      <td>-0.013840</td>\n",
       "      <td>-0.024993</td>\n",
       "      <td>-0.011080</td>\n",
       "      <td>-0.046879</td>\n",
       "      <td>0.015491</td>\n",
       "      <td>132.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>440</td>\n",
       "      <td>-0.045472</td>\n",
       "      <td>-0.044642</td>\n",
       "      <td>0.039062</td>\n",
       "      <td>0.001215</td>\n",
       "      <td>0.016318</td>\n",
       "      <td>0.015283</td>\n",
       "      <td>-0.028674</td>\n",
       "      <td>0.026560</td>\n",
       "      <td>0.044528</td>\n",
       "      <td>-0.025930</td>\n",
       "      <td>220.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>441</td>\n",
       "      <td>-0.045472</td>\n",
       "      <td>-0.044642</td>\n",
       "      <td>-0.073030</td>\n",
       "      <td>-0.081414</td>\n",
       "      <td>0.083740</td>\n",
       "      <td>0.027809</td>\n",
       "      <td>0.173816</td>\n",
       "      <td>-0.039493</td>\n",
       "      <td>-0.004220</td>\n",
       "      <td>0.003064</td>\n",
       "      <td>57.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>442 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          age       sex       bmi        bp        s1        s2        s3  \\\n",
       "0    0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401   \n",
       "1   -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412   \n",
       "2    0.085299  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.032356   \n",
       "3   -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038   \n",
       "4    0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142   \n",
       "..        ...       ...       ...       ...       ...       ...       ...   \n",
       "437  0.041708  0.050680  0.019662  0.059744 -0.005697 -0.002566 -0.028674   \n",
       "438 -0.005515  0.050680 -0.015906 -0.067642  0.049341  0.079165 -0.028674   \n",
       "439  0.041708  0.050680 -0.015906  0.017282 -0.037344 -0.013840 -0.024993   \n",
       "440 -0.045472 -0.044642  0.039062  0.001215  0.016318  0.015283 -0.028674   \n",
       "441 -0.045472 -0.044642 -0.073030 -0.081414  0.083740  0.027809  0.173816   \n",
       "\n",
       "           s4        s5        s6  target  \n",
       "0   -0.002592  0.019908 -0.017646   151.0  \n",
       "1   -0.039493 -0.068330 -0.092204    75.0  \n",
       "2   -0.002592  0.002864 -0.025930   141.0  \n",
       "3    0.034309  0.022692 -0.009362   206.0  \n",
       "4   -0.002592 -0.031991 -0.046641   135.0  \n",
       "..        ...       ...       ...     ...  \n",
       "437 -0.002592  0.031193  0.007207   178.0  \n",
       "438  0.034309 -0.018118  0.044485   104.0  \n",
       "439 -0.011080 -0.046879  0.015491   132.0  \n",
       "440  0.026560  0.044528 -0.025930   220.0  \n",
       "441 -0.039493 -0.004220  0.003064    57.0  \n",
       "\n",
       "[442 rows x 11 columns]"
      ]
     },
     "execution_count": 232,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:26:05.367314Z",
     "start_time": "2020-05-21T09:26:04.732977Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 4 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(2, 2)\n",
    "\n",
    "ax[0, 0].boxplot(data.age)\n",
    "ax[0, 1].boxplot(data.bmi)\n",
    "ax[1, 0].boxplot(data.age)\n",
    "ax[1, 1].boxplot(data.bmi)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:26:05.575829Z",
     "start_time": "2020-05-21T09:26:05.370606Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.boxplot(data['age'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-16T09:54:17.393979Z",
     "start_time": "2020-05-16T09:54:17.390159Z"
    }
   },
   "source": [
    "### Модели"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:52:11.351577Z",
     "start_time": "2020-05-21T09:52:11.347067Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.linear_model import Ridge, Lasso, LinearRegression, HuberRegressor\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Создание полиномиальных фич для повышения качества регрессии"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:26:06.473287Z",
     "start_time": "2020-05-21T09:26:06.469774Z"
    }
   },
   "outputs": [],
   "source": [
    "#poly = PolynomialFeatures(2)\n",
    "#X = pd.DataFrame(poly.fit_transform(X))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Разбиение выборки на три группы"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:41:48.754371Z",
     "start_time": "2020-05-21T09:41:48.750587Z"
    }
   },
   "outputs": [],
   "source": [
    "selectK = SelectKBest(score_func = chi2, k = 4)\n",
    "X = selectK.fit_transform(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T09:52:33.269704Z",
     "start_time": "2020-05-21T09:52:33.261874Z"
    }
   },
   "outputs": [],
   "source": [
    "X = data.drop(columns = 'target', axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:16.813593Z",
     "start_time": "2020-05-21T10:13:16.801085Z"
    }
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, X_test = train_test_split(X, y, test_size = 0.2, random_state = 25)\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 25)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MinMax трансформация целевой переменной"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:09:02.289274Z",
     "start_time": "2020-05-21T10:09:02.285794Z"
    }
   },
   "outputs": [],
   "source": [
    "#minmax = MinMaxScaler()\n",
    "\n",
    "#y_train = y_train.values.reshape(-1, 1)\n",
    "#y_valid = y_valid.values.reshape(-1, 1)\n",
    "\n",
    "#y_train = minmax.fit_transform(y_train)\n",
    "#y_valid = minmax.transform(y_valid)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### StandardScaler трансформация целевой переменной"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:09:02.809717Z",
     "start_time": "2020-05-21T10:09:02.806135Z"
    }
   },
   "outputs": [],
   "source": [
    "#from sklearn.preprocessing import StandardScaler\n",
    "#scaler = StandardScaler()\n",
    "#\n",
    "#y_train = y_train.values.reshape(-1, 1)\n",
    "#y_valid = y_valid.values.reshape(-1, 1)\n",
    "#\n",
    "#y_train = scaler.fit_transform(y_train)\n",
    "#y_valid = scaler.transform(y_valid)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Полезные функции"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:19.606715Z",
     "start_time": "2020-05-21T10:13:19.601186Z"
    }
   },
   "outputs": [],
   "source": [
    "def mape(y_true, y_pred):\n",
    "    \n",
    "    '''\n",
    "Mape metric\n",
    "    '''\n",
    "    print(mape.__doc__)\n",
    "    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:19.799712Z",
     "start_time": "2020-05-21T10:13:19.794307Z"
    }
   },
   "outputs": [],
   "source": [
    "def fit_predict(model, metric, X_train, X_valid, y_train, y_valid):\n",
    "    \n",
    "    model.fit(X_train, y_train)\n",
    "    pred_valid = model.predict(X_valid)\n",
    "    print('Metric result is {0}'.format(metric(y_valid, pred_valid)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Инициализация моделей"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:21.278903Z",
     "start_time": "2020-05-21T10:13:21.274087Z"
    }
   },
   "outputs": [],
   "source": [
    "linreg = LinearRegression()\n",
    "ridge = Ridge(random_state = 25)\n",
    "lasso = Lasso(random_state = 25)\n",
    "huber = HuberRegressor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:21.464780Z",
     "start_time": "2020-05-21T10:13:21.455459Z"
    }
   },
   "outputs": [],
   "source": [
    "models = [('ridge', ridge), ('linreg', linreg), ('lasso', lasso), ('huber', huber)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:21.789489Z",
     "start_time": "2020-05-21T10:13:21.703380Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current model is ridge\n",
      "\n",
      "Mape metric\n",
      "    \n",
      "Metric result is 42.82002841842687\n",
      "\n",
      "Current model is linreg\n",
      "\n",
      "Mape metric\n",
      "    \n",
      "Metric result is 37.275663126563394\n",
      "\n",
      "Current model is lasso\n",
      "\n",
      "Mape metric\n",
      "    \n",
      "Metric result is 43.075550596693134\n",
      "\n",
      "Current model is huber\n",
      "\n",
      "Mape metric\n",
      "    \n",
      "Metric result is 36.95826794495467\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for model in models:\n",
    "    \n",
    "    print('Current model is {0}'.format(model[0]))\n",
    "    fit_predict(model[1], mape, X_train, X_valid, y_train, y_valid)\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Наблюдения\n",
    "\n",
    "Итак, мы видим, что лучший результат без тюнинга параметров показывает модель HuberRegressor."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Мысли по улучшению модели\n",
    "\n",
    "Как можно улучшить модель?\n",
    "<br>\n",
    "Самый очевидный способ - заняться подгоном параметров.\n",
    "<br>\n",
    "Также можно вернуться к фичам и попробовать погенерировать несколько составных фич и посмотреть на результаты еще раз. \n",
    "<br>\n",
    "Еще предлагаю попробовать другие метрики и посмотреть что получится на других метриках, они могут помочь понять хорошая у нас модель или нет. Проблема MSE именно в том, что это величина абсолютная, поэтому сравнить модели между собой можно - та, где меньше MSE, лучше, но общего понимания того, насколько хорошо модель предсказывает целевую переменную мы не получаем."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# FeatureSelection\n",
    "\n",
    "Может быть зашить в функцию fit_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:25.427810Z",
     "start_time": "2020-05-21T10:13:25.423366Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import explained_variance_score\n",
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:13:25.751441Z",
     "start_time": "2020-05-21T10:13:25.747025Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import SelectKBest, chi2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Метрики"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:09:14.309476Z",
     "start_time": "2020-05-21T10:09:14.299898Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4600042382337842"
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "explained_variance_score(y_valid, pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-21T10:09:14.651667Z",
     "start_time": "2020-05-21T10:09:14.640205Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.43552591604382596"
      ]
     },
     "execution_count": 199,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_valid, pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-15T06:49:30.820984Z",
     "start_time": "2020-05-15T06:49:30.814969Z"
    }
   },
   "source": [
    "Идеи:\n",
    "\n",
    "Построить графики\n",
    "Посмотреть корреляции переменных\n",
    "Посмотреть пропуски и повторы"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
