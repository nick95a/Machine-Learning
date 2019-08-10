import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Step 1-2: Load the dataset, explore it and prepare it for the model

# Load the dataset and the names of the categories
train_dataset = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 24)
categories = train_dataset.target_names
dataset = pd.DataFrame(train_dataset.data)



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
X_train = vect.fit_transform(train_dataset.data)
X_train.shape

# Have a look at the stop words and vocabulary
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
NB.fit(X_train_new, train_dataset.target)

SVC = LinearSVC()
SVC.fit(X_train_new, train_dataset.target)


# Little check on the model
phrases = ['Pastafarianism is a religion', 'Hockey is the best sport in the world']
X_vect = vect.transform(phrases)
X_trans = trans.transform(X_vect)
pred = NB.predict(X_trans)

for i,p in enumerate(pred):
    print(phrases[i])
    print(train_dataset.target_names[p])


# We can create a pipeline from the standard sklearn tools to make the process easier
nb_pipeline = Pipeline([('BOW', CountVectorizer()), ('Tfidf', TfidfTransformer()), ('NB', MultinomialNB())])
lin_svc_pipeline = Pipeline([('BOW', CountVectorizer()),("Tfidf", TfidfTransformer()), ("SVC", LinearSVC())])


test_dataset = fetch_20newsgroups(subset = 'test', shuffle = True, random_state = 24)

nb_pipeline.fit(train_dataset.data, train_dataset.target)
nb_predictions = nb_pipeline.predict(test_dataset.data)

lin_svc_pipeline.fit(train_dataset.data, train_dataset.target)
lin_svc_predictions = lin_svc_pipeline.predict(test_dataset.data)


# Step 4: Model evaluations and parameter tuning


print("NB accuracy: ", accuracy_score(test_dataset.target, nb_predictions))
print("SVC accuracy: ", accuracy_score(test_dataset.target, lin_svc_predictions))

print(classification_report(test_dataset.target, nb_predictions, target_names = test_dataset.target_names))
print(classification_report(test_dataset.target, lin_svc_predictions, target_names = test_dataset.target_names))

# Parameter tuning. To be continued

