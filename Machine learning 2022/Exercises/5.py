"""
Week 5
"""
#%%
"""
1 - Decision trees
"""
#%% 5.1.1 - load data
import numpy as np

# Names of data objects
dataobjectNames = [
    'Human',
    'Python',
    'Salmon',
    'Whale',
    'Frog',
    'Komodo dragon',
    'Bat',
    'Pigeon',
    'Cat',
    'Leopard shark',
    'Turtle',
    'Penguin',
    'Porcupine',
    'Eel',
    'Salamander',
    ]

# Attribute names
attributeNames = [
    'Body temperature',
    'Skin cover',
    'Gives birth',
    'Aquatic creature',
    'Aerial creature',
    'Has legs',
    'Hibernates'
    ]

# Attribute values
X = np.asarray(np.mat('''
    1 1 1 0 0 1 0;
    0 2 0 0 0 0 1;
    0 2 0 1 0 0 0;
    1 1 1 1 0 0 0;
    0 0 0 2 0 1 1;
    0 2 0 0 0 1 0;
    1 1 1 0 1 1 1;
    1 3 0 0 1 1 0;
    1 4 1 0 0 1 0;
    0 2 1 1 0 0 0;
    0 2 0 2 0 1 0;
    1 3 0 2 0 1 0;
    1 5 1 0 0 1 1;
    0 2 0 1 0 0 0;
    0 0 0 2 0 1 1 '''))

# Class indices
y = np.asarray(np.mat('3 4 2 3 0 4 3 1 3 2 4 1 3 2 0').T).squeeze()

# Class names
classNames = ['Amphibian', 'Bird', 'Fish', 'Mammal', 'Reptile']
    
# Number data objects, attributes, and classes
N, M = X.shape
C = len(classNames)

print('Ran Exercise 5.1.1')

#%% 5.1.2 - decision tree
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Fit regression tree classifier, Gini split criterion, no pruning
criterion = 'gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X, y)

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
# to get a readable plot. Hint: Try to maximize the figure after it displays.
fname='tree_ex512_' + criterion + '.png'

fig = plt.figure(figsize=(4,4),dpi=100) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.show()

print('Ran Exercise 5.1.2')

#%% 5.1.3 - decision tree, entropy
import os
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread
#import graphviz
#import pydotplus


# Fit regression tree classifier, Gini split criterion, no pruning
criterion='entropy'
# dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
dtc = dtc.fit(X,y)

# convert the tree into a png file using the Graphviz toolset
fname='tree_ex513_' + criterion + '.png'

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
# to get a useful plot. Hint: Try to maximize the figure after it displays.
fig = plt.figure(figsize=(4,4),dpi=100) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.show()

print('Ran Exercise 5.1.3')

#%% 5.1.4 - classify from given attributes
# Define a new data object (a dragon) with the attributes given in the text
x = np.array([0, 2, 1, 2, 1, 1, 1]).reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

# Print results
print('\nNew object attributes:')
print(dict(zip(attributeNames,x[0])))
print('\nClassification result:')
print(classNames[x_class])

print('Ran Exercise 5.1.4')

#%% 5.1.5 - load wine data and remove outliers
import numpy as np
import os
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
workingDir = os.getcwd()
print("Running from: " + workingDir)

mat_data = loadmat('./Exercises/02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]


# Remove outliers
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:,0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

print('Ran Exercise 5.1.5')

#%% 5.1.6 - decision tree wine
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Fit classification tree using, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
dtc = dtc.fit(X,y)

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your screen resolution and setup you may need to decrease or increase 
# the figsize and DPI setting to get a useful plot. 
# Hint: Try to open the generated png file in an external image editor as it can be easier 
# to inspect outside matplotlib's figure environment.
fname='tree_ex516_' + criterion + '_wine_data.png'
fig = plt.figure(figsize=(12,12),dpi=300) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.close() 

fig = plt.figure()
plt.imshow(imread(fname))
plt.axis('off')
plt.box('off')
plt.show()

print('Min_samples_split bestemmer hvor mange. der mindst skal i hver branch, når der splittes.')
print('Jo højere min_samples_split, jo mindre bliver decision tree.')

print('Ran Exercise 5.1.6')

#%% 5.1.7 - classify wine from attributes
# Define a new data object (new type of wine) with the attributes given in the text
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

# Print results
print('\nNew object attributes:')
for i in range(len(attributeNames)):
    print('{0}: {1}'.format(attributeNames[i],x[0][i]))
print('\nClassification result:')
print(classNames[x_class])

print('Ran Exercise 5.1.7')
#%%
"""
2 - Linear and logistic regression
"""
#%% 5.2.1 - generate data
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show
import numpy as np

# Number of data objects
N = 100

# Attribute values
X = np.array(range(N))

# Noise
eps_mean, eps_std = 0, 0.1
eps = np.array(eps_std*np.random.randn(N) + eps_mean)

# Model parameters
w0 = -0.5
w1 = 0.01

# Outputs
y = w0 + w1*X + eps

# Make a scatter plot
figure()
plot(X,y,'o')
xlabel('X'); ylabel('y')
title('Illustration of a linear relation with noise')

show()

print('Ran Exercise 5.2.1')

#%% 5.2.2 - estimate model parameters
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import numpy as np

# Use dataset as in the previous exercise
N = 100
X = np.array(range(N)).reshape(-1,1)
eps_mean, eps_std = 0, 0.1
eps = np.array(eps_std*np.random.randn(N) + eps_mean).reshape(-1,1)
w0 = -0.5
w1 = 0.01
y = w0 + w1*X + eps
y_true = y - eps

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y)
# Compute model output:
y_est = model.predict(X)
# Or equivalently:
#y_est = model.intercept_ + X @ model.coef_


# Plot original data and the model output
f = figure()

plot(X,y,'.')
plot(X,y_true,'-')
plot(X,y_est,'-')
xlabel('X'); ylabel('y')
legend(['Training data', 'Data generator', 'Regression fit (model)'])

show()

print('Ran Exercise 5.2.2')

#%% 5.2.3 - model more parameters
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, ylim
import numpy as np
import sklearn.linear_model as lm

# Parameters
Kd = 5  # no of terms for data generator
Km = 3  # no of terms for regression model
N = 50  # no of data objects to train a model
Xe =  np.linspace(-2,2,1000).reshape(-1,1) # X values to visualize true data and model
eps_mean, eps_std = 0, 0.5          # noise parameters

# Generate dataset (with noise)
X = np.linspace(-2,2,N).reshape(-1,1)
Xd = np.power(X, range(1,Kd+1))
eps = (eps_std*np.random.randn(N) + eps_mean)
w = -np.power(-.9, range(1,Kd+2))
y = w[0] + Xd @ w[1:] + eps 


# True data generator (assuming no noise)
Xde = np.power(Xe, range(1,Kd+1))
y_true = w[0] + Xde @ w[1:]



# Fit ordinary least squares regression model
Xm = np.power(X, range(1,Km+1))
model = lm.LinearRegression()
model = model.fit(Xm,y)

# Predict values
Xme = np.power(Xe, range(1,Km+1))
y_est = model.predict(Xme)

# Plot original data and the model output
f = figure()
plot(X,y,'.')
plot(Xe,y_true,'-')
plot(Xe,y_est,'-')
xlabel('X'); ylabel('y'); ylim(-2,8)
legend(['Training data', 'Data generator K={0}'.format(Kd), 'Regression fit (model) K={0}'.format(Km)])

show()

print('Ran Exercise 5.2.3')

#%% 5.2.4 - fit model wine
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

########################### import data from 5.1.5
import numpy as np
import os
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
workingDir = os.getcwd()
print("Running from: " + workingDir)

mat_data = loadmat('./Exercises/02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]

# Remove outliers
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:,0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

###########################

# Split dataset into features and target vector
alcohol_idx = attributeNames.index('Alcohol')
y = X[:,alcohol_idx]

X_cols = list(range(0,alcohol_idx)) + list(range(alcohol_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

denisity_idx = attributeNames.index('Density')
print(f'Dentisty is the attribute with index {denisity_idx}')

print('Ran Exercise 5.2.4')


#%% 5.2.5 -transform attributes
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

########################### import data from 5.1.5
import numpy as np
import os
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
workingDir = os.getcwd()
print("Running from: " + workingDir)

mat_data = loadmat('./Exercises/02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]

# Remove outliers
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:,0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

###########################


# Split dataset into features and target vector
alcohol_idx = attributeNames.index('Alcohol')
y = X[:,alcohol_idx]

X_cols = list(range(0,alcohol_idx)) + list(range(alcohol_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Additional nonlinear attributes
fa_idx = attributeNames.index('Fixed acidity')
va_idx = attributeNames.index('Volatile acidity')
Xfa2 = np.power(X[:,fa_idx],2).reshape(-1,1)
Xva2 = np.power(X[:,va_idx],2).reshape(-1,1)
Xfava = (X[:,fa_idx]*X[:,va_idx]).reshape(-1,1)
X = np.asarray(np.bmat('X, Xfa2, Xva2, Xfava'))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display plots
figure(figsize=(12,8))

subplot(2,1,1)
plot(y, y_est, '.g')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)')

subplot(4,1,3)
hist(residual,40)

subplot(4,3,10)
plot(Xfa2, residual, '.r')
xlabel('Fixed Acidity ^2'); ylabel('Residual')

subplot(4,3,11)
plot(Xva2, residual, '.r')
xlabel('Volatile Acidity ^2'); ylabel('Residual')

subplot(4,3,12)
plot(Xfava, residual, '.r')
xlabel('Fixed*Volatile Acidity'); ylabel('Residual')

show()

print('Ran Exercise 5.2.5')

#%% 5.2.6 - logistic regression

########################### import data from 5.1.5
import numpy as np
import os
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
workingDir = os.getcwd()
print("Running from: " + workingDir)

mat_data = loadmat('./Exercises/02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]

# Remove outliers
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:,0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

###########################

model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object (wine sample)'); ylabel('Predicted prob. of class White');
legend(['White', 'Red'])
ylim(-0.01,1.5)

show()

print('Ran Exercise 5.2.6')



