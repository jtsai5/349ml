#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:36:42 2018

@author: emmacrowley
"""
import nltk
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import pandas as pd
from nltk.util import ngrams
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""
lb_make = LabelEncoder()
data["host_response_time_encode"] = lb_make.fit_transform(data["host_response_time"])
data[["host_response_time","host_response_time_encode"]].head(11)
data["neighborhood_clean_encode"] = lb_make.fit_transform(data["neighbourhood_cleansed"])
data[["neighbourhood_cleansed","neighborhood_clean_encode"]].head()
data["property_type_encode"] = lb_make.fit_transform(data["property_type"])
data[["property_type","property_type_encode"]].head()
data["bed_type_encode"] = lb_make.fit_transform(data["bed_type"])
data[["bed_type","bed_type_encode"]].head()
data["cancellation_policy_encode"] = lb_make.fit_transform(data["cancellation_policy"])
data[["cancellation_policy","cancellation_policy_encode"]].head()

new_data = data.drop(["name","summary","host_response_time","property_type","bed_type","cancellation_policy",
         "space","description","neighbourhood_cleansed","neighborhood_overview","notes", "transit","access","interaction",
         "house_rules","host_since","host_about","host_response_time","room_type","amenities"], axis=1)

del new_data["id"]

new_data.loc[new_data.bathrooms == '?', 'bathrooms'] = 0
new_data.loc[new_data.bedrooms == '?', 'bedrooms'] = 0
new_data = new_data.drop(["review_scores_accuracy","review_scores_cleanliness","review_scores_checkin",
                             "review_scores_communication","review_scores_location","review_scores_value","zipcode"], axis=1)
new_data = new_data[new_data.price != '?']
new_data['price'] = new_data['price'].astype(int)
new_data['bedrooms'] = new_data['bedrooms'].astype(float)
new_data['price'] = new_data['bathrooms'].astype(float)
new_data.loc[new_data.review_scores_rating =='?', 'review_scores_rating'] = 95
new_data['review_scores_rating'] = new_data['review_scores_rating'].astype(float)
new_data = new_data.drop(["Host since"],axis=1)
new_data = new_data[new_data.maximum_nights != 7000]

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(new_data)





y = new_data["review_scores_rating"]
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=0.1164, random_state=1)
    
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1315, random_state=1)

mylist = list(range(1,50))
neighbors = filter(lambda x: x % 2 != 0, mylist)
acc_scores = np.empty((7,25))
distance = list(range(1,7))


for j in distance:
    ind_scores = []
    for k in neighbors:
        knn=KNeighborsClassifier(n_neighbors=k, p=j)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_val)
        score=accuracy_score(y_val, pred)
        ind_scores.append(score)
    acc_scores[j] = ind_scores

# The best are (1,15), (1,16), (1,17), (1,18), (1,14)
  
# Now comparing across weights or not
weight_table = []
Pair = namedtuple("Pair", ["p", "k"])
pairs = [Pair(1, 31), Pair(1, 33), Pair(1, 35), Pair(1, 37), Pair(1, 29)]
for pair in pairs:
    knn = KNeighborsClassifier(n_neighbors= pair.k, p = pair.p, weights = 'distance')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_val)
    score = accuracy_score(y_val, pred)
    weight_table.append(score)     
    
noweight_table = []
for pair in pairs:
    knn = KNeighborsClassifier(n_neighbors= pair.k, p = pair.p, weights = 'uniform')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_val)
    score = accuracy_score(y_val, pred)
    noweight_table.append(score)    
 
"""    
 
# The best are (1,29, d), (1,31, d), (1,33, d), (1,37,d)
        
knn = KNeighborsClassifier(n_neighbors = 29, p = 1, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_val)
score1 = accuracy_score(y_val, pred)

knn = KNeighborsClassifier(n_neighbors = 31, p = 1, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_val)
score2 = accuracy_score(y_val, pred)

knn = KNeighborsClassifier(n_neighbors = 33, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_val)
score3 = accuracy_score(y_val, pred)

knn = KNeighborsClassifier(n_neighbors = 37, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_val)
score4 = accuracy_score(y_val, pred)



x = np.array([0,1,2,3])
y = np.array([0.342, 0.332, 0.302, 0.302])
# All p = 1 and weighted
my_xticks = ['29','31','33','37']
plt.xticks(x, my_xticks)
plt.plot(x, y, label = "Train")
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy Score on Train Set')
plt.savefig('numneighbors_train.png')



knn = KNeighborsClassifier(n_neighbors = 29, p = 1, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
score1 = accuracy_score(y_test, pred)

knn = KNeighborsClassifier(n_neighbors = 31, p = 1, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
score2 = accuracy_score(y_test, pred)

knn = KNeighborsClassifier(n_neighbors = 33, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
score3 = accuracy_score(y_test, pred)

knn = KNeighborsClassifier(n_neighbors = 37, weights = 'distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
score4 = accuracy_score(y_test, pred)



x = np.array([0,1,2,3])
y = np.array([0.354, 0.362, 0.312, 0.312])
# All p = 1 and weighted
my_xticks = ['29','31','33','37']
plt.xticks(x, my_xticks)
plt.plot(x, y, label = "Test")
plt.xlabel('Number of neighbors')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Accuracy Score on Both Sets')

plt.savefig('numneighbors_test.png')


"""
# Calculating miscalculation error
MSE = [1 - x for x in acc_scores]

optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.savefig('mse_k.png')

knn = KNeighborsClassifier(n_neighbors=30, metric="chebyshev")
knn.fit(X_train, y_train)
pred = knn.predict(X_val)
#print accuracy_score(y_val, pred)
"""
