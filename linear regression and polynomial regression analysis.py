#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

import pandas as pd

import sklearn.preprocessing 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()




data = pd.read_csv('Boston.csv')

df = data.copy()

df.head()

df_ = df.corr()




X = df.drop(['median home price','Average Rooms/Dwelling.','Distance to Employment Centres'], axis=1)

Y = df[['median home price']]

print(X.shape)

print(Y.shape)




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

X_ = scaler.transform(X)

X = pd.DataFrame(data=X_, columns = X.columns)

X.head()




from sklearn.model_selection import train_test_split

xtrain ,xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3,random_state=25, shuffle=True)

print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)




from sklearn.linear_model import LinearRegression

model = LinearRegression()




model.fit(xtrain, ytrain)




train_pred = model.predict(xtrain)




test_pred= model.predict(xtest)




from sklearn.metrics import r2_score

r2_test_lr=r2_score(ytest,test_pred )

print('The testing score is:',r2_test_lr)




r2_train_lr=r2_score(ytrain,train_pred)

print('The training score is:',r2_train_lr)




from yellowbrick.regressor import ResidualsPlot




plt.figure(figsize=(15,6))

visualizer = ResidualsPlot(model)

visualizer.fit(xtrain.values, ytrain.values)  

visualizer.score(xtest.values, ytest.values)  

visualizer.poof()   




from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)





X_train_poly = poly_features.fit_transform(xtrain)




poly_model = LinearRegression()

poly_model.fit(X_train_poly, ytrain)




y_train_predicted = poly_model.predict(X_train_poly)



X_test_poly=poly_features.fit_transform(xtest)

y_test_predict = poly_model.predict(X_test_poly)



r2_train = r2_score(ytrain, y_train_predicted)



r2_test = r2_score(ytest, y_test_predict)

print ('The r2 score for training set is: ',r2_train)

print ('The r2 score for testing set is: ',r2_test)

print('ACCURACY OF THE MODEl: ', r2_test*100)


# In[ ]:




