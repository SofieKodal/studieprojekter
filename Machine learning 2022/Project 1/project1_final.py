#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:00:42 2023

@author: signeolsen
"""

# Project 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the Weather Training csv data using the Pandas library
filename = './Weather Training Data.csv'
df = pd.read_csv(filename)
# Only take the data from the city Albury
df = df[df['Location'].str.contains('Albury')]
# Delete the row of index
del df['row ID']
# delete Cloud9am, Cloud3pm, Evaporation, Sunshine and Location
del df['Cloud9am']
del df['Cloud3pm']
del df['Evaporation']
del df['Sunshine']
del df['Location']
del df['RainToday']

# Delete wind dir to make i easier
del df['WindGustDir']
del df['WindDir9am']
del df['WindDir3pm']

# Delete all rows with nan so we go from 2142 to 2120 rows
df2=df.dropna()

# We calculate the average for the atribute that was taking at to times a day 
df2['WindSpeed9am'] = (df2['WindSpeed9am']+df2['WindSpeed3pm'])/2
del df2['WindSpeed3pm']
df2['Humidity9am'] = (df2['Humidity9am']+df2['Humidity3pm'])/2
del df2['Humidity3pm']
df2['Pressure9am'] = (df2['Pressure9am']+df2['Pressure3pm'])/2
del df2['Pressure3pm']
df2['Temp9am'] = (df2['Temp9am']+df2['Temp3pm'])/2
del df2['Temp3pm']


# Check for outliers
df2.plot.scatter(x='MinTemp', y='MaxTemp',c='DarkBlue')

df2.RainToday = df2.RainTomorrow.map(dict(Yes=1, No=0))

#%% Summary statistics
df2[['MinTemp','MaxTemp', 'WindGustSpeed', 'WindSpeed9am', 'Humidity9am', 'Pressure9am', 'Temp9am']].describe()


#%%

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df2.values  


# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 9) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df2.columns[cols])
attributeNames[4]='WindSpeed'
attributeNames[5]='Humidity'
attributeNames[6]='Pressure'
attributeNames[7]='Temp'

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:,-1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

#%% Standardization
# Center the data (subtract mean column values)
Xc = X - np.ones((N,1))*X.mean(0)

#%% Summary statistics of attributes
for i in range(8):
    print(np.mean(Xc[:,i]))
    
#%%
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show)
boxplot(Xc[:,:])
#title('Class: {0}'.format(classNames[c]))
title('Boxplot for Albury')
xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
y_up = Xc.max()+(Xc.max()-Xc.min())*0.1; y_down = Xc.min()-(Xc.max()-Xc.min())*0.1
ylim(y_down, y_up)


#%%
import math
n_bins = 30
fig, axs = plt.subplots(2, 4, tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0,0].hist(Xc[:,0], bins=n_bins)
axs[0,1].hist(Xc[:,1], bins=n_bins)
axs[0,2].hist(Xc[:,2], bins=n_bins)
axs[0,3].hist(Xc[:,3], bins=n_bins)
axs[1,0].hist(Xc[:,4], bins=n_bins)
axs[1,1].hist(Xc[:,5], bins=n_bins)
axs[1,2].hist(Xc[:,6], bins=n_bins)
axs[1,3].hist(Xc[:,7], bins=n_bins)
axs[0, 0].set_title('MinTemp')
axs[0, 1].set_title('MaxTemp')
axs[0, 2].set_title('Rainfall')
axs[0, 3].set_title('WindGus')
axs[1, 0].set_title('WindSpeed')
axs[1, 1].set_title('Humidity')
axs[1, 2].set_title('Pressure')
axs[1, 3].set_title('Temperature')


#%% Present data
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show)
tit = ('Rain Tomorrow', 'No Rain Tomorrow')
figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(Xc[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: ' +tit[c])
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = Xc.max()+(Xc.max()-Xc.min())*0.1; y_down = Xc.min()-(Xc.max()-Xc.min())*0.1
    ylim(y_down, y_up)

show()

print('Ran Exercise 4.2.4')

#%% Transformation
from scipy.stats import boxcox

Xtrans = X

# Shift to get only positive values
Xtrans[:,3] = X[:,3] - np.min(np.abs(X[:,3])) + 1e-10
Xtrans[:,4] = X[:,4] - np.min(np.abs(X[:,4])) + 1e-10

# Boxcox transformantion
Xtrans[:,3], box_gus = boxcox(Xtrans[:,3])
Xtrans[:,4], box_speed = boxcox(Xtrans[:,4])


#%%
from scipy.linalg import svd
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
from scipy.io import loadmat
import scipy.linalg as linalg


# Subtract mean value from data
Y = Xtrans - np.ones((N,1))*Xtrans.mean(0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Plot variance explained
figure()
plot(rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');

#%%
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Weather data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

#%%
i = 1
j = 2
plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))


#%%
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#%%
# The first 5 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b','p']
bw = .2
V1 = np.delete(V, 8, 0)
attributeNames1 = np.delete(attributeNames,8)
r = np.arange(1,M)
for i in pcs:
    plt.bar(r+i*bw, V1[:,i], width=bw)
plt.xticks(r+bw, attributeNames1)
plt.xlabel('Attributes')
plt.xticks(rotation=70)
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

#%%
Attributes = [1,4,5,6]
NumAtr = len(Attributes)

figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
show()





















