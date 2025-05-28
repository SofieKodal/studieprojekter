"""
Week 7
"""
# '02450Toolbox_Python/Data/...'

#%% 7.1.1 - Calculate accuracy
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# load data from exercise 1.5.1
import pandas as pd
filename = '02450Toolbox_Python/Data/iris.csv'
df = pd.read_csv(filename)
raw_data = df.values  
cols = range(0, 4) 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,-1] 
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
N, M = X.shape
C = len(classNames)

# This script crates predictions from three KNN classifiers using cross-validation

# Maximum number of neighbors
L=[1, 20, 80]

CV = model_selection.LeaveOneOut()
i=0

# store predictions.
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
    for l in L:
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)

        dy.append( y_est )
        # errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
yhat[:,0] # predictions made by first classifier.
# Compute accuracy here.

difA = y_true-yhat[:,0] # 0 = correctly classified
difB = y_true-yhat[:,1]
difC = y_true-yhat[:,2]
accA = 1-np.count_nonzero(difA)/len(difA)
accB = 1-np.count_nonzero(difB)/len(difB)
accC = 1-np.count_nonzero(difC)/len(difC)

print(f"The accuracies are A = {accA}, B = {accB}, C = {accC}.")

#%% 7.1.2 - Jeffrey interval
from toolbox_02450 import jeffrey_interval

# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)
[thetahatB, CIB] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)
[thetahatC, CIC] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)
print("Theta point estimate", thetahatB, " CI: ", CIB)
print("Theta point estimate", thetahatC, " CI: ", CIC)

print("Classifier C has a lower estimate than the other classifiers")

#%% 7.1.3 - change alpha
alpha = 0.01
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)
[thetahatB, CIB] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)
[thetahatC, CIC] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)
print("Theta point estimate", thetahatB, " CI: ", CIB)
print("Theta point estimate", thetahatC, " CI: ", CIC)

print("The bigger alpha gives a smaller Jeffrey interval")


#%% 7.1.4  - McNeamer A and B
from toolbox_02450 import mcnemar

print("Theta estimates accuracy - higher theta -> better model")
# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

print("Since the theta_hat < 0, it indicates B is better than A, but 0 is in the CI, so there is actually no difference.")
print("But the p-value is relativly large, which also says there is no significant difference.")

#%% 7.1.5  - McNeamer A and C
from toolbox_02450 import mcnemar

# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)

print("theta = theta_A-theta_C point estimate", thetahat, " CI: ", CI, "p-value", p)

print("Since the theta_hat > 0, A is better than C according to this method, because 0 isn't part of the CI.")
print("The p-value is low and the CI doesn't contain 0.")
print("The difference is likely not due to chance, and C")

#%% 7.1.6 - Deicision tree classifier for model B
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread


# Maximum number of neighbors
L=[1, 20, 80]

CV = model_selection.LeaveOneOut()
i=0

# store predictions.
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
    for l in L:
        if l == 20:
            dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
            dtc = dtc.fit(X_train, y_train)
            y_est = np.asarray(dtc.predict(X_test),dtype=int)
        else:
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)

        dy.append( y_est )
        # errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)

yhat[:,0] # predictions made by first classifier.
# Compute accuracy here.

difA = y_true-yhat[:,0] # 0 = correctly classified
difB = y_true-yhat[:,1]
difC = y_true-yhat[:,2]
accA = 1-np.count_nonzero(difA)/len(difA)
accB = 1-np.count_nonzero(difB)/len(difB)
accC = 1-np.count_nonzero(difC)/len(difC)

print(f"The accuracies are A = {accA}, B = {accB}, C = {accC}.")
print("Everytime we run the code, B gets a different accuracy.")

# Compute the mcNemars interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

#%% 7.2.1 - Training and t-test, confidence interval CI
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st

# requires data from exercise 1.5.1
import os
from scipy.io import loadmat
workingDir = os.getcwd()
print("Running from: " + workingDir)
mat_data = loadmat('./02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]
attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
X = X[:,0:11]
attributeNames = attributeNames[0:11]
N, M = X.shape


X,y = X[:,:10], X[:,10:]
# This script crates predictions from three KNN classifiers using cross-validation

test_proportion = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

mA = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

yhatA = mA.predict(X_test)
yhatB = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings

# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - yhatA ) ** 2
zB = np.abs(y_test - yhatB ) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(f"The CI is {CI} and the p-value is {p}.")



