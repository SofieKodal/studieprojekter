#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:01:25 2023

@author: signeolsen
"""

# From Project 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
filename = 'Desktop/Machine learning/Project 2/Weather Training Data.csv'
df = pd.read_csv(filename)
df = df[df['Location'].str.contains('Albury')]
del df['row ID']
del df['Cloud9am']
del df['Cloud3pm']
del df['Evaporation']
del df['Sunshine']
del df['Location']
del df['RainToday']
del df['WindGustDir']
del df['WindDir9am']
del df['WindDir3pm']
df2=df.dropna()
df2['WindSpeed9am'] = (df2['WindSpeed9am']+df2['WindSpeed3pm'])/2
del df2['WindSpeed3pm']
df2['Humidity9am'] = (df2['Humidity9am']+df2['Humidity3pm'])/2
del df2['Humidity3pm']
df2['Pressure9am'] = (df2['Pressure9am']+df2['Pressure3pm'])/2
del df2['Pressure3pm']
df2['Temp9am'] = (df2['Temp9am']+df2['Temp3pm'])/2
del df2['Temp3pm']
df2.RainToday = df2.RainTomorrow.map(dict(Yes=1, No=0))
raw_data = df2.values  
cols = range(0, 9) 
X = raw_data[:, cols]
attributeNames = np.asarray(df2.columns[cols])
attributeNames[4]='WindSpeed'
attributeNames[5]='Humidity'
attributeNames[6]='Pressure'
attributeNames[7]='Temp'

#%% Project 2
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from toolbox_02450 import rlr_validate
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, legend, boxplot
from toolbox_02450 import feature_selector_lr, bmplot
from sklearn import model_selection, tree
from statistics import mode
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st



from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch

#%% Standardize
X_trans=np.zeros((2120,9))
for i in range(9):
    X_trans[:,i]=  (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

 
#%% Linear model with regularization parameter
X_reg = X_trans

y = X_reg[:,2] # y = Rainfall
X_reg = np.delete(X_reg, 2, 1) # Delete Rainfall

attributeNames_reg = np.delete(attributeNames, 2) # Delete Rainfall
attributeNames_reg = np.insert(attributeNames_reg, 0, 'Offset') # Insert Offset

N, M = X_reg.shape

# Add offset attribute
X_reg = np.concatenate((np.ones((X_reg.shape[0],1)),X_reg),1) #Tilføjer en søjle med 1'er
attributeNames = [u'Offset']+attributeNames
M = M+1


# Values of lambda
lambdas = np.logspace(-0.5,3)

cross_validation = 10    
    
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_reg, y, lambdas, cross_validation)

print(opt_lambda)

figure(1, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()



#%% Two cross validation annd ANN
# outer loop
K = 3
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
true_lamda_linreg = np.empty(K)

# Values of lambda
lambdas = np.logspace(-0.5,3)

# Define the model structure
n_hidden_units = [1,2,3] # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = [0,0,0]
for i in range(len(n_hidden_units)):
    model[i] = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units[i]), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units[i], 1), # H hidden units to 1 output neuron
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.MSELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
# Do cross-validation:
errors = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:

#%% Train for three models

k=0
for train_index, test_index in CV.split(X_reg):
    
    # extract training and test set for current CV fold
    X_train = X_reg[train_index,:]
    y_train = y[train_index]
    X_test = X_reg[test_index,:]
    y_test = y[test_index]
    
    # For ANN
    X_train_torch = torch.Tensor(X_train)
    y_nn = y_train.reshape(len(y_train),1)
    y_train_torch = torch.Tensor(y_nn)
    X_test_torch = torch.Tensor(X_test)
    y_test_torch = torch.Tensor(y_test)
    
    internal_cross_validation = 3
    
    # BASELINE
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    
       
    # ANN
    for i in range(len(model)):
        net, final_loss, learning_curve = train_neural_net(model[i],
                                                       loss_fn,
                                                       X=X_train_torch,
                                                       y=y_train_torch,
                                                       n_replicates=internal_cross_validation,
                                                       max_iter=max_iter)
        print('\n\tBest loss for ANN[{}]: {}\n'.format(i,final_loss))
    
        # Determine estimated class labels for test set
        y_test_est = net(X_test_torch) # activation of final note, i.e. prediction of network
        y_test = y_test_torch.type(dtype=torch.uint8)
        # Determine errors and error rate
        error_rate = sum((y_test_est - y_test_torch)**2)/len(y_test_est)
        errors.append(error_rate) # store error rate for current CV fold  

    
    # LINEAR REGRESSION
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    true_lamda_linreg[k] = opt_lambda
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    k+=1

#%%
#import torch
#import numpy as np
#errors_arr = np.array(errors)
#errors = torch.reshape(errors,(len(errors)/len(n_hidden_units),len(n_hidden_units)))
#np.array_split(errors, len(n_hidden_units))


# Print the lowest error and corresponding number of hidden units
optimal_n_hidden_units = None
lowest_error = float('inf')
for i in range(len(n_hidden_units)):
    for n_units, err in zip(n_hidden_units, errors[:,i]):
        if err < lowest_error:
            lowest_error = err
            optimal_n_hidden_units = n_units
        print(f"Number of hidden units: {n_units}, error: {err}")
    print(f"\nOptimal number of hidden units: {optimal_n_hidden_units}, lowest error: {lowest_error}")


#%%
print('\n\tError for baseline: {}\n'.format(Error_test_nofeatures))
print('\n\tError for linear regression: {}\n'.format(Error_test_rlr))

#%% Find the controlling parameters
# LINEAR REGRESSION
for j in range(0,10):
    print(true_lamda_linreg[j])
    print(Error_test[j])



















































#%% Classification
classLabels = raw_data[:,-1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
N, M = X.shape
C = len(classNames)
X_class = X_trans[:,range(0,8)]
attributeNames_class = np.delete(attributeNames, 8) # Delete RainTomorrow

#%% Classifications trees
## Crossvalidation
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(1, 16, 1)

# Outer loop
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))
Error_test_nofeatures = np.empty(K)
Error_train_nofeatures = np.empty(K)
lambda_log = np.logspace(-4,4,num=50)
#lambda_log = np.power(10.,range(-5,2)) # Invers af lambda
Error_log_train = np.empty((K,np.shape(lambda_log)[0]))
Error_log_test = np.empty((K,np.shape(lambda_log)[0]))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))



k=0


for train_index, test_index in CV.split(X_reg):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X_class[train_index,:]
    y_train = y[train_index]
    X_test = X_class[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    ## BASELINE
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-mode(y_train)).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-mode(y_test)).sum()/y_test.shape[0]
    
    
    ## LOGISTIC
    # Set the regularization type (L1, L2, Elastic Net)
    penalty = 'l2'
    
    i = 0
    for complexity in lambda_log:
        
        # Create a logistic regression model
        model = LogisticRegression(penalty=penalty, C=complexity)
    
        # Train the model on the training set
        model = model.fit(X_train,y_train)
    
        # Classify and assess probabilities
        y_est_train_log = model.predict(X_train)
        y_est_test_log = model.predict(X_test)
   
        y_est_log_prob = model.predict_proba(X_train)[:, 0] 
        
        # Compute error
        Error_log_train[k,i] = np.sum(y_est_train_log != y_train) / float(len(y_est_train_log))
        Error_log_test[k,i] = np.sum(y_est_test_log != y_test) / float(len(y_est_test_log))
        
        i+= 1
    
    ## DecisionTreeClassifier
    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train

    k+=1
    
f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])

# Display results
# Hvis vi bruger alle features er din Training error lav men din Test error høj
# Hvis vi bruger feature selection vil vores Test error være meget bedre
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))


np.where(Error_test[:,9] == min(Error_test[:,9]))


f = figure();
class0_ids = np.nonzero(y_train==0)[0].tolist()
plot(class0_ids, y_est_log_prob[class0_ids], '.y')
class1_ids = np.nonzero(y_train==1)[0].tolist()
plot(class1_ids, y_est_log_prob[class1_ids], '.b')
xlabel('Data object'); ylabel('Predicted prob. of class NoRain');
legend(['NoRain', 'Rain'])



#%% Find the controlling parameters
# LOGISTICS
true_lamda_log = 1/lambda_log
for j in range(0,10):
    print(np.where(Error_log_test[j,:] == min(Error_log_test[j,:]))) #Print the index
    print(true_lamda_log [np.where(Error_log_test[j,:] == min(Error_log_test[j,:]))]) # Print the lamda
    print(min(Error_log_test[j,:])) #Print the error
    
# DecisionTreeClassifier
for j in range(0,10):
    print(np.where(Error_test[:,j] == min(Error_test[:,j])))
    print(min(Error_test[:,j]))

#%% Nulhypotese
yhatA = y_est_test_log # linear regression
yhatB = np.repeat(mode(y_test),len(y_est_test_log))  # baseline
yhatC = y_est_test  # logistic

# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - yhatA ) ** 2
zB = np.abs(y_test - yhatB ) ** 2
zC = np.abs(y_test - yhatC ) ** 2

alpha = 0.05

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(f"The CI is {CI} and the p-value is {p}.")

# Compute confidence interval of z = zA-zC and p-value of Null hypothesis
z = zA - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(f"The CI is {CI} and the p-value is {p}.")

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zB - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(f"The CI is {CI} and the p-value is {p}.")














