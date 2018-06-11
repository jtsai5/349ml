# -*- encoding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
from nltk import word_tokenize
import csv
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
from collections import Counter

filename = "/Users/jtsai/Desktop/Northwestern/Junior/EECS349/project/clean_listings2.csv" 
import pandas as pd
data = pd.read_csv(filename)
data['Count_Luxury'] = 0
data['Count_budget'] = 0
data['Count_convenience'] = 0
rowNum = 0
df1 = data.iloc[:,1:9]
nltk.download('punkt')

luxury = ['fancy', 'Fancy', 'nice', 'Nice', 'Spacious', 'spacious', 'Lovely', 'lovely', 
          'Luxury', 'luxury', 'Luxurious', 'luxurious', 'Amazing', 'amazing', 
          'Modern', 'modern', 'Gorgeous', 'gorgeous','Expensive', 'expensive', 'Lakefront', 'lakefront',
          'beachfront', 'Beachfront']
budget = ['cheap', 'Cheap', 'simple', 'Simple','minimal', 'Minimal', 'small', 'Small'
          'tiny', 'Tiny', 'Budget', 'budget', 'affordable', 'Affordable', 'inexpensive', 'Inexpensive',
          'cost-conscious', 'Cost-conscious', 'studio', 'Studio']
convenience = ['convenient', 'Convenient', 'central', 'Central', 'loop', 'Loop', 
               'El', 'el', 'CTA', 'cta', 'train', 'Train', 'Transit', 'transit', 'close', 'Close'
               'location', 'Location', 'nearby', 'Nearby'] 

#def ut8list(l):
#    for item in l:
#        item = unicode(item, 'utf-8')
#    return l

#luxury = ut8list(luxury)
#print luxury
#convenience = ut8list(convenience)

def MasterNote(row):
    relString = ""
    relString = "".join(row[1][2:10])       
    #import pdb; pdb.set_trace()

    #for col in row:
    return relString

def count(words, d):
    count = 0
    for word in words:
        if d[word] != 0:
        #if word in d:
            count += d[word]
    return count

for row in data.iterrows():
    masterN = MasterNote(row)
    y = []
#    try: 
    unigrams = ngrams(nltk.word_tokenize(masterN.decode('utf-8', 'ignore')), 1)
    x = list(unigrams)
    for e in x:
        y.append(e[0])
#    except: import pdb; pdb.set_trace()
#    print Counter(unigrams)
    data.Count_convenience[rowNum] = count(convenience, Counter(y))
    data.Count_Luxury[rowNum] = count(luxury, Counter(y))
    data.Count_budget[rowNum] = count(budget, Counter(y))

    rowNum += 1

print data
y = data['review_scores_rating']

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

"""
def read_csv(self):
    with open(self.csv_file, 'r') as input_csv:
        for item in input_csv:
            item = item.split(',')
            doc, label = re.findall('\w+')
def generate_word_features(self):
    frequency_dist = nltk.FreqDist()
    for word in self.words:
        frequency_dist[word] += 1
        self.feature_words = list(frequency_dist)[:self.featureset_size]
"""

"""
for index,row in dataa.iterrows():
    token = nltk.word_tokenize(row['name'])
    unigrams = ngrams(token,1)
    dataa['count']= Counter(unigrams)
"""
