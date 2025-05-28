"""
Week 4
"""
import os
os.chdir('Desktop/Machine learning/Exercises')

"""
4.1 Univariate and multivariate normal distribution
"""
#%% 4.1.1 - normal distribution
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import numpy as np

# Number of samples
N = 200

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 20

# Generate samples from the Normal distribution
X = np.random.normal(mu,s,N).T 
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the samples and histogram
figure(figsize=(12,4))
title('Normal distribution')
subplot(1,2,1)
plot(X,'.')
subplot(1,3,3)
hist(X, bins=nbins)
show()

print('Ran Exercise 4.1.1')

#%% 4.1.2 - theoretical vs. simlulated
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import numpy as np
# Number of samples
N = 200

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 20

# Generate samples from the Normal distribution
X = np.random.normal(mu,s,N).T 
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the samples and histogram
figure()
title('Normal distribution')
subplot(1,2,1)
plot(X,'x')
subplot(1,2,2)
hist(X, bins=nbins)

# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

show()

print('Ran Exercise 4.1.2')

#%% 4.1.3 - normal pdf
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import numpy as np
from scipy import stats

# Number of samples
N = 500

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 20

# Generate samples from the Normal distribution
X = np.random.normal(mu,s,N).T 
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the histogram
f = figure()
title('Normal distribution')
hist(X, bins=nbins, density=True)

# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X.min(), X.max(), 1000)
pdf = stats.norm.pdf(x,loc=17,scale=2)
plot(x,pdf,'.',color='red')

# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

show()

print('Ran Exercise 4.1.3')

#%% 4.1.4 - 2D normal distribution (multivariat)
import numpy as np

# Number of samples
N = 1000

# Mean
mu = np.array([13, 17])

# Covariance matrix
S = np.array([[4,3],[3,9]])

# Generate samples from the Normal distribution
X = np.random.multivariate_normal(mu, S, N)

print('Ran Exercise 4.1.4')

#%% 4.1.5 - covariance, scatter plot and histogram
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show, 
                               xlabel, ylabel, xticks, yticks, colorbar, cm, 
                               imshow, suptitle)
import numpy as np

# Number of samples
N = 1000

# Standard deviation of x1
s1 = 2

# Standard deviation of x2
s2 = 3

# Correlation between x1 and x2
corr = 0.5

# Covariance matrix
S = np.matrix([[s1*s1, corr*s1*s2], [corr*s1*s2, s2*s2]])

# Mean
mu = np.array([13, 17])

# Number of bins in histogram
nbins = 20

# Generate samples from multivariate normal distribution
X = np.random.multivariate_normal(mu, S, N)


# Plot scatter plot of data
figure(figsize=(12,8))
suptitle('2-D Normal distribution')

subplot(1,2,1)
plot(X[:,0], X[:,1], 'x')
xlabel('x1'); ylabel('x2')
title('Scatter plot of data')

subplot(1,2,2)
x = np.histogram2d(X[:,0], X[:,1], nbins)
imshow(x[0], cmap=cm.gray_r, interpolation='None', origin='lower')
colorbar()
xlabel('x1'); ylabel('x2'); xticks([]); yticks([]);
title('2D histogram')

show()

print('Ran Exercise 4.1.5')

#%% 4.1.6 - mean and std on digit data
from matplotlib.pyplot import (figure, subplot, title, imshow, xticks, yticks, 
                               show, cm)
import scipy.linalg as linalg
from scipy.io import loadmat
import numpy as np

# Digits to include in analysis (to include all: n = range(10))
n = [0]

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat('02450Toolbox_Python/Data/zipdata.mat')['traindata']
X = traindata[:,1:]
y = traindata[:,0]
N, M = X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = (y==v)
    class_mask = class_mask | cmsk
X = X[class_mask,:]
y = y[class_mask]
N = np.shape(X)[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

figure()
subplot(1,2,1)
I = np.reshape(mu, (16,16))
imshow(I, cmap=cm.gray_r)
title('Mean')
xticks([]); yticks([])
subplot(1,2,2)
I = np.reshape(s, (16,16))
imshow(I, cmap=cm.gray_r)
title('Standard deviation')
xticks([]); yticks([])

show()

print('Ran Exercise 4.1.6')

#%% 4.1.7  generate new images (univariat and multivariat)
from matplotlib.pyplot import (figure, subplot, imshow, xticks, yticks, title,
                               cm, show)
import numpy as np
from scipy.io import loadmat

# Digits to include in analysis (to include all, n = range(10) )
n = [9]

# Number of digits to generate from normal distributions
ngen = 10

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat('02450Toolbox_Python/Data/zipdata.mat')['traindata']
X = traindata[:,1:]
y = traindata[:,0]
N, M = np.shape(X) #or X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = (y==v)
    class_mask = class_mask | cmsk
X = X[class_mask,:]
y = y[class_mask]
N = np.shape(X)[0] # or X.shape[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

# Generate 10 samples from 1-D normal distribution
Xgen = np.random.randn(ngen,256)
for i in range(ngen):
    Xgen[i] = np.multiply(Xgen[i],s) + mu

# Plot images
figure()
for k in range(ngen):
    subplot(2, int(np.ceil(ngen/2.)), k+1)
    I = np.reshape(Xgen[k,:], (16,16))
    imshow(I, cmap=cm.gray_r);
    xticks([]); yticks([])
    if k==1: title('Digits: 1-D Normal')


# Generate 10 samples from multivariate normal distribution
Xmvgen = np.random.multivariate_normal(mu, S, ngen)
# Note if you are investigating a single class, then you may get: 
# """RuntimeWarning: covariance is not positive-semidefinite."""
# Which in general is troublesome, but here is due to numerical imprecission


# Plot images
figure()
for k in range(ngen):
    subplot(2, int(np.ceil(ngen/2.)), k+1)
    I = np.reshape(Xmvgen[k,:], (16,16))
    imshow(I, cmap=cm.gray_r);
    xticks([]); yticks([])
    if k==1: title('Digits: Multivariate Normal')

show()
print('The second method is best')

print('Ran Exercise 4.1.7')

#%%
"""
4.2 Visualizing Fisher's Iris data
"""
#%% 4.2.1 - load Iris data
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('02450Toolbox_Python/Data/iris.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,0,4)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(4,1,151)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((150,4))
for i in range(4):
    X[:,i] = np.array(doc.col_values(i,1,151)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print('Ran Exercise 4.2.1')
#

#%% 4.2.2 - histogram
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show
import numpy as np

figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(int(u),int(v),i+1)
    hist(X[:,i], color=(0.2, 0.8-i*0.2, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()

print('Ran Exercise 4.2.2')

#%% 4.2.3 - boxplot
from matplotlib.pyplot import boxplot, xticks, ylabel, title, show

boxplot(X)
xticks(range(1,5),attributeNames)
ylabel('cm')
title('Fisher\'s Iris data set - boxplot')
show()

print('Ran Exercise 4.2.3')

#%% 4.2.4 - boxplot each class
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show)

figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(X[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)

show()

print('Ran Exercise 4.2.4')

#%% 4.2.5 - scatter plot all attributes
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()

print('Ran Exercise 4.2.5')

#%% 4.2.6 - 3D plot
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D

# Indices of the variables to plot
ind = [0, 1, 2]
colors = ['blue', 'green', 'red']

f = figure()
ax = f.add_subplot(111, projection='3d') #Here the mpl_toolkits is used
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X[class_mask,ind[0]], X[class_mask,ind[1]], X[class_mask,ind[2]], c=colors[c])

ax.view_init(20, 170)
ax.set_xlabel(attributeNames[ind[0]])
ax.set_ylabel(attributeNames[ind[1]])
ax.set_zlabel(attributeNames[ind[2]])

show()

print('Ran Exercise 4.2.6')

#%% 4.2.7 - matrix plot
from matplotlib.pyplot import (figure, imshow, xticks, xlabel, ylabel, title, 
                               colorbar, cm, show)
from scipy.stats import zscore


X_standarized = zscore(X, ddof=1)

figure(figsize=(12,6))
imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(4), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title('Fisher\'s Iris data matrix')
colorbar()

show()

print('Ran Exercise 4.2.7')

#%%
"""
4.3 Visualizing Wine Data
"""
#%% 4.3.1 - remove outliers
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

# Load Matlab data file and extract variables of interest
mat_data = loadmat('02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]
attributeNames = [name[0][0] for name in mat_data['attributeNames']]
classNames = [cls[0][0] for cls in mat_data['classNames']]

# We start with a box plot of each attribute
figure()
title('Wine: Boxplot')
boxplot(X)
xticks(range(1,M+1), attributeNames, rotation=45)

# From this it is clear that there are some outliers in the Alcohol
# attribute (10x10^14 is clearly not a proper value for alcohol content)
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).
figure(figsize=(12,6))
title('Wine: Boxplot (standarized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1,M+1), attributeNames, rotation=45)

# This plot reveals that there are clearly some outliers in the Volatile
# acidity, Density, and Alcohol attributes, i.e. attribute number 2, 8,
# and 11. 

# Next, we plot histograms of all attributes.
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(int(u),int(v),i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram')
    

# This confirms our belief about outliers in attributes 2, 8, and 11.
# To take a closer look at this, we next plot histograms of the 
# attributes we suspect contains outliers
figure(figsize=(14,9))
m = [1, 7, 10]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Wine: Histogram (selected attributes)')


# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask,:]
y = y[valid_mask]
N = len(y)


# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(int(u),int(v),i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram (after outlier detection)')

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

show()

print('Ran Exercise 4.3.1')

#%% 4.3.2 - useful attributes to discriminate

from matplotlib.pyplot import figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

# Load Matlab data file and extract variables of interest
mat_data = loadmat('02450Toolbox_Python/Data/wine.mat')
X = mat_data['X']
y = np.squeeze(mat_data['y'])
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [name[0][0] for name in mat_data['attributeNames']]
classNames = [cls[0] for cls in mat_data['classNames'][0]]
    
# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask,:]
y = y[valid_mask]
N = len(y)
Xnorm = zscore(X, ddof=1)

## Next we plot a number of atttributes
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

print('Volatile and total sulfur seems to be useful')

print('Ran Exercise 4.3.2')
