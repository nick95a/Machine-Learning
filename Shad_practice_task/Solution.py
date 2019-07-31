import pandas as pd
import numpy as np

def produce_names(data):
    l = []
    for i in range(len(data.columns)):
        l.append('feature_{}'.format(i + 1))
    return l

def produce_dict(col_names, new_names):
    d = {}
    for index, name in enumerate(col_names):
        d[name] = new_names[index]
    return d

def get_missing_indices(data, namelist):
    indices = set()
    for name in namelist:
        indices.update(list(data[data[name] == -1].index))
    return indices

# Step 1.
# At this stage we load the dataset and see what we can infer from it regarding the problem at hand.
dataset = pd.read_csv('spoiled_data.csv', index_col = 0)

dataset.shape

dataset.info()
# We see that some of the dtypes are object so we may need to address that and change variable types.
X = dataset.iloc[:,:-1]
column_names = list(X.columns)
namelist = produce_names(X)
d = produce_dict(column_names, namelist)
X = X.rename(columns = d)
y = dataset.iloc[:, -1]

data = dataset.replace('-', -1)
# We do not have typical NaN values that are detected as null, but remember that in the assignment it is stated
# that null values are put down as '-'
num_null = dataset.isnull().sum().sum()
dups = X.duplicated().sum().sum()

# I want to take all the rows with missing values out of the dataset
# Transform the variables
# Predict the missing values
# Evaluate accuracy

indices = get_missing_indices(data, column_names)
