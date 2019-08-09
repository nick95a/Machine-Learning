import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load the dataset and the names of the categories
news_bunch = fetch_20newsgroups(subset = 'all', shuffle = True, random_state = 24)
categories = news_bunch.target_names
dataset = pd.DataFrame(news_bunch.data)

# Get an idea of the dataset and the news in int
#dataset.info()

# print out the first piece line by line for illustrative purposes
'''
s = str(dataset.iloc[0, 0])
news = s.split('\n')
for n in news:
    print(n)
'''
# Get the occurencies of the words in the whole corpus of documents
vect = CountVectorizer()
X = vect.fit_transform(news_bunch.data)
X.shape
stop_words, vocab = vect.stop_words_, vect.vocabulary_

# Bring meaning to the counts by finding term frequencies relative to the size of the whole corpus
