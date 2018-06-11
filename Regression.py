#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 18:05:44 2018

@author: emmacrowley
"""

filename = "/Users/emmacrowley/Documents/out.csv"
import pandas as pd
data = pd.read_csv(filename)

from sklearn import metrics
from sklearn import datasets,linear_model
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

y = data['review_scores_rating']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1164, random_state=1)
    
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1315, random_state=1)

del X_train['review_scores_rating']
del X_val['review_scores_rating']
del X_test['review_scores_rating']

'''Regression'''
regressor = LinearRegression()
regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_val)


meanabsoluteerror = metrics.mean_absolute_error(y_val, y_pred)
print meanabsoluteerror
meansquared = metrics.mean_squared_error(y_val, y_pred)
print meansquared
rootmeansquared = metrics.mean_squared_error(y_val, y_pred)
print regressor.score(X_val,y_val)

'''
y_pred_t = regressor.predict(X_test)
plt.scatter(y_pred, y_val, c='b',label='Validation')
plt.scatter(y_pred_t, y_test,c='r',label = 'Test')
plt.xlabel("Predicted Rating")
plt.ylabel("Actual Rating")
plt.legend(loc = 'lower right')
plt.title("Predicted vs. Actual Ratings")
plt.show()
'''

c = pd.DataFrame(zip(X_train.columns, regressor.coef_), columns = ['features', 
             'Estimated Coefficients'])


from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1-.8)))
Xnew = sel.fit_transform(X_train)
reg = LinearRegression()

X_vall = X_val.drop(['host_response_rate','host_identity_verified',
                     'is_location_exact','Shared room?',
                     'TV','Wireless Internet','Kitchen'
                     ,'Smoking Allowed','Essentials','Heating','Shampoo','Pets allowed',
                     'Gym','Washer','Bathtub','Pet on property','Breakfast',
                     'patio or balcony','instant_bookable'], axis=1)


x_testt = X_test.drop(['host_response_rate','host_identity_verified',
                     'is_location_exact','Shared room?',
                     'TV','Wireless Internet','Kitchen'
                     ,'Smoking Allowed','Essentials','Heating','Shampoo','Pets allowed',
                     'Gym','Washer','Bathtub','Pet on property','Breakfast',
                     'patio or balcony','instant_bookable'], axis=1)


reg.fit(Xnew, y_train)
ypreddd = reg.predict(X_vall)
print('MAE:', metrics.mean_absolute_error(y_val,ypreddd))
print metrics.mean_squared_error(y_val,ypreddd)
print reg.score(X_vall, y_val)
y_pred_tt = reg.predict(x_testt)
plt.scatter(ypreddd, y_val, c='b', label = 'Validation')
plt.scatter(y_pred_tt, y_test, c='r', label = 'Test')
plt.xlabel("Predicted Rating")
plt.ylabel("Actual Rating")
plt.title("Predicted vs. Actual Ratings")
plt.legend(loc='lower right')
plt.show()

coef = pd.DataFrame(zip(X_vall.columns, reg.coef_), columns = ['features',
                    'Estimated Coefficients'])
coef.to_csv('coeff.csv')
