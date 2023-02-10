#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:49:18 2023

@author: coreylangner
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# import our medical insurance data set 

df = pd.read_csv('/Users/coreylangner/Desktop/python-portfolio-project-starter-files/insurance.csv') 

# Goal of our project is to 
# 1. look over all of our variables and be able to gather summary statistics 
# 2. use our variables to build a machine learning model to predict a users insurance charges based on the other 6 features in our dataset
# Lets invetigate some summary statistics and central tendency stuff for our variables 

# Lets find the average age an individual in the dataset 
Mean_Age = df['age'].mean()
Median_Age = df['age'].median()
mode_Age = df['age'].mode()
# Average age in our dataset is 39.2 

# charges 
averageCHarges = df['charges'].mean() #13270
medianCharges = df['charges'].median() #9382 

# BMI 
BMI_avg = df['bmi'].mean()
BMI_median = df['bmi'].median()

# children 
child_avg = df['children'].mean()
child_median = df['children'].median()
child_mode = df['children'].mode()

# count of a regions 
Count_Regions = df['region'].value_counts()
# the counts per region are as follows 
# 1 SE 364
# 2 SW 325 
# 3 NW 325
# 4 NE 324

# Average age of smokers vs non-smokers 
smokers = df[df['smoker'] == 'yes'] # this line creates a new df where it only features those users who are smokers
nonsmokers = df[df['smoker'] == 'no'] # this line creates a new df where it only features those users who are nonsmokers

MeanAgeSmoker = smokers['age'].mean()
MeanAgeNonSmoker = nonsmokers['age'].mean()
# The avg age of smokers is slightly lower than non-smokers by 0.87 years

# lets check on the distribution of our variables using some histograms 
# plotting histograms is really simple you just use matplotlib .plot.hist()
# Age 
Age_Hist = df['age'].plot.hist()
# bmi
bmi_Hist = df['bmi'].plot.hist()
# children 
child_Hist = df['children'].plot.hist()
# charges 
charges_Hist = df['charges'].plot.hist()

# Lets build a simple linear model using BMI (since it is normally distributed) and individuals insurance charges 

BMI_chargesScat = plt.scatter(df['bmi'], df['charges'])

# correlation of pearsons r between bmi and chargea 
correlation = np.corrcoef(df['bmi'], df['charges'])
# weak positive correlation 

from sklearn.linear_model import LinearRegression # importing the model I am choosing to use 
from sklearn.model_selection import train_test_split # used to split my data into train and test 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # used to evaluate how well our model performs

# fit a simple model to the data

X = df[['bmi', 'age']].values
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# this model does a poor job of fitting the data, r^2 value of only 0.104 so it only explains about 10% of the variation 


