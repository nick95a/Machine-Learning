import numpy as np
import pandas as pd
import re

# Step 1: Understanding the data and preprocessing it.
'''
As help on doing the preprocessing I used the following resources:
https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
https://regex101.com/
http://magma.maths.usyd.edu.au/magma/faq/extract
'''


train_reviews = []
test_reviews = []
for line in open('../imdb/movie_data/full_train.txt','r'):
    train_reviews.append(line.strip())

for line in open('../imdb/movie_data/full_test.txt','r'):
    test_reviews.append(line.strip('<>'))


'''
We will remove all the <br /> tags that indicate line breaks, but have no meaning for our sentiment analysis.
Also, we will remove all the punctuation.
'''

punctuation_re = re.compile("[.;:!\'?,\"()\[\]]")
line_break_re = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess(re_1, re_2, dataset):

    reviews = [punctuation_re.sub("", line.lower()) for line in reviews]
    reviews = [line_break_re.sub(" ", line) for line in reviews]
    return reviews

train_reviews = preprocess(train_reviews)
test_reviews = preprocess(test_reviews)

