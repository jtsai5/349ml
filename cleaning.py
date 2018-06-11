#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:04:13 2018

@author: jtsai
"""
import csv
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split


import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter


text = "Here is my list of words like very clean and not clean and so clean"
token = nltk.word_tokenize(text)
bigrams = ngrams(token,2)
unigrams = ngrams(token,1)
print unigrams

print Counter(unigrams)


filename = "/Users/jtsai/Desktop/Northwestern/Junior/EECS349/project/clean_listings2.csv"
data = pd.read_csv(filename)
y = data.review_scores_rating


"""
data.add
vocab = ["fancy", "nice", "spacious", "lovely", "luxury", "luxurious", "amazing", "modern", "gorgeous", "expensive", "lakefront"]
for index,row in dataa.iterrows():
    token = nltk.word_tokenize(row['name'])
    unigrams = ngrams(token,1)
    dataa['count']= Counter(unigrams)
   
 """
 
 
luxury = ['fancy', 'Fancy', 'nice', 'Nice', 'Spacious', 'spacious', 'Lovely', 'lovely', 
          'Luxury', 'luxury', 'Luxurious', 'luxurious', 'Amazing', 'amazing', 
          'Modern', 'modern', 'Gorgeous', 'gorgeous','Expensive', 'expensive', 'Lakefront', 'lakefront']
budget = ['cheap', 'Cheap', 'simple', 'Simple','minimal', 'Minimal', 'small', 'Small'
          'tiny', 'Tiny', 'Budget', 'budget', 'affordable', 'Affordable', 'inexpensive', 'Inexpensive']
convenience = ['convenient', 'Convenient', 'central', 'Central', 'loop', 'Loop', 
               'El', 'el', 'train', 'Train', 'Transit', 'transit', 'close', 'Close'
               'location', 'Location']
data['Count_luxury'] = 0
data['Count_budget'] = 0
data['Count_convenience'] = 0


def MasterNote(row):
    relString = ""
    for col in row:
        relString += col
    return relString

def count(words, dictionary):
    count = 0
    for word in words:
        if word in dictionary:
            count += dictionary[word]
    return count
"""
value = count(['random', 'hello', 'computer', 'hey'], {'hey':3, 'hello':4})
print value
"""

"""

mylist = ['name', 'summary']
df1 = data.iloc[:,1:9]
for row, i in df1.iteritems():
    masterN = MasterNote(row)
    print masterN
    unigrams = ngrams(nltk.word_tokenize(masterN), 1)
    Count_simple = count(mylist, Counter(unigrams))
    print Count_simple
    Count_luxury = count(luxury, unigrams)
    Count_budget = count(budget, unigrams)
    Count_convenience = count(convenience, unigrams)
    data['Count_luxury'] = Count_luxury
    data['Count_budget'] = Count_budget
    data['Count_convenience'] = Count_convenience
    
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
"""
# comparison not working
# record output when something doesn't match. python unicode string comparison 
# get first word from file and see what it is
# is s == u'beachfront' or s u"beachfront" s = "beachfront" 
# or preprocess s.encode(utf-8) == "beachfront"

# bash script for varying 3 parameters. put choices of classifiers in code (python) to try different things. 
# check file that tries classifiers
# partition into training, validation (check performacne on this) - find 3 or 4 best, and then final test on test set(each 500 examples)
    
