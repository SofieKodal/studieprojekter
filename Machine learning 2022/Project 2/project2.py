
"""
Project 2
"""

#%% Load data
# From Project 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
filename = './Weather Training Data.csv'
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
classLabels = raw_data[:,-1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
N, M = X.shape
C = len(classNames)

#%% Standardize and find y estimate

X_trans = np.zeros((2120,9))
means = np.zeros(9)
stds = np.zeros(9)
for i in range(9):
    X_trans[:,i]=  (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    means[i] = np.mean(X[:,i])
    stds[i] = np.std(X[:,i])
    
X_reg = X_trans
y = X_reg[:,2] # y = Rainfall
X_reg = np.delete(X_reg, 2, 1) # Delete Rainfall

attributeNames_reg = np.delete(attributeNames, 2) # Delete Rainfall
attributeNames_reg = np.insert(attributeNames_reg, 0, 'Offset') # Insert Rainfall

w_stand = np.array([0.01, 0.26, -0.18, 0.04, 0.09, 0.38, -0.17, 0.07, -0.06])
#w = w_stand*stds + means

y_est_lin = np.zeros(2120)
for i in range(2120):
    x_stand = X_trans[i,:]
    y_stand = w_stand@x_stand
    y_est_lin[i] = y_stand*stds[2] + means[2]

error = y_est_lin - y
days = np.linspace(0, 2120, 2120)
plt.plot(days,error)

#%% Two level cross validation

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X_reg):
    
    # extract training and test set for current CV fold
    X_train = X_reg[train_index,:]
    y_train = y[train_index]
    X_test = X_reg[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) == 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
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

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) == 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    y_est= m.predict(X[:,ff])
    residual=y-y_est
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2, int( np.ceil(len(ff) // 2)),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')
    
    
show()


#%% Comparing models (linear and baseline)
#7.2.1 - Training and t-test, confidence interval CI
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st


yhatA = y_est_lin # linear regression
yhatB = np.mean(y)  # baseline

# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y - yhatA ) ** 2
zB = np.abs(y - yhatB ) ** 2

# compute confidence interval of model A and B
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(f"The CI is {CI} and the p-value is {p}.")

