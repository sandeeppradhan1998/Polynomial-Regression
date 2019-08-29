# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:57:29 2019

@author: Mandalson
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#import data set
dataframe=pd.read_csv('Position_Salaries.csv')
x=dataframe.iloc[:,1:2].values
y=dataframe.iloc[:,2].values


#fitting linear regression
from sklearn.linear_model import LinearRegression
l_regression=LinearRegression()
l_regression=l_regression.fit(x,y)

#fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
p_regression=PolynomialFeatures(degree=4)
x_poly=p_regression.fit_transform(x)
p_regression=p_regression.fit(x_poly,y)
l_regression2=LinearRegression()
l_regression2=l_regression2.fit(x_poly,y)

#visualising the linear regression
plt.scatter(x,y, color='red')
plt.plot(x,l_regression.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression
plt.scatter(x,y, color='red')
plt.plot(x, l_regression2.predict(p_regression.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
l_regression.predict(6.5)

# Predicting a new result with Polynomial Regression
l_regression2.predict(p_regression.fit_transform(6.5))


