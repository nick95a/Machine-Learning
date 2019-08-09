import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB


# Step 1-2: Load the dataset, explore it and prepare it for the model

# Load the dataset and the names of the categories
news_bunch = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 24)
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
# Get the occurencies of the words in the whole corpus of documents.
# Here we use a combination of CountVectorizer and TfidfTransformer while the TfidfVectorizer does both
vect = CountVectorizer()
X_train = vect.fit_transform(news_bunch.data)
X_train.shape
stop_words, vocab = vect.stop_words_, vect.vocabulary_

# Bring meaning to the counts by finding term frequencies relative to the size of the whole corpus
trans = TfidfTransformer()
X_train_new = trans.fit_transform(X_train)

# idf attribute show the inverse-document frequency weightings: one for each feature
trans.idf_

# Check out the shape of the transformed feature map
X_train_new.shape

# Step 3: Building the ML model. The choice is vast, but we will limit ourselves to the basic ones.
NB = MultinomialNB()
NB.fit(X_train_new, news_bunch.target)


# Little check on the model
phrases = ['Pastafarianism is a religion', 'Hockey is the best sport in the world']
X_n = vect.transform(phrases)
X_n_n = trans.transform(X_n)
pred = NB.predict(X_n_n)

for i,p in enumerate(pred):
    print(phrases[i])
    print(news_bunch.target_names[p])

# We can create a pipeline from the standard sklearn tools to make the process easier
