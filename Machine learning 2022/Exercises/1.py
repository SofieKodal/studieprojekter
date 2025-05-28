"""
Week 1
"""

#%% Packages
import toolbox_02450
import xlrd
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

#%%
"""
Iris data
"""
#%% 1.5.1 - Open CSV file

# inspect data
with open("./02450Toolbox_Python/Data/iris.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    print(row)

# read data
filename = './02450Toolbox_Python/Data/iris.csv'

# dataframe of CSV values
df = pd.read_csv(filename)

# raw data and values
raw_data = df.values

# make into df of 4 coloumns (4 attributes)
cols = range(0, 4) 
X = raw_data[:, cols]

# extract headers from CSV
attributeNames = np.asarray(df.columns[cols])

# give str. classes numerical values
classLabels = raw_data[:,-1] # -1 takes the last column

# find unique classes
classNames = np.unique(classLabels)

# give each class a number 
classDict = dict(zip(classNames,range(len(classNames))))


# vector of classes
y = np.array([classDict[cl] for cl in classLabels])

# shape of X
N, M = X.shape

# number of classes
C = len(classNames)

#%% 1.5.2 - Open excel file

# Load xls sheet with data
doc = xlrd.open_workbook('./02450Toolbox_Python/Data/iris.xls').sheet_by_index(0)

# same as before
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=4)
classLabels = doc.col_values(4,1,151) # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((150,4))
for i in range(4):
    X[:,i] = np.array(doc.col_values(i,1,151)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

#%% 1.5.3 - Open mat files
from scipy.io import loadmat

# Load Matlab data file to python dict structure
iris_mat = loadmat('./02450Toolbox_Python/Data/iris.mat', squeeze_me=True)

X = iris_mat['X']
y = iris_mat['y']
M = iris_mat['M']
N = iris_mat['N']
C = iris_mat['C']
attributeNames = iris_mat['attributeNames']
classNames = iris_mat['classNames']

#%% 1.5.4 - Classes and regression

# continue with data from 1.5.3

## Classification problem
# Copy X and y and attributenames
X_c = X.copy();
y_c = y.copy();
attributeNames_c = attributeNames.copy();
i = 1; j = 2;
color = ['r','g', 'b']

# plot classes
plt.title('Iris classification problem')
for c in range(len(classNames)):
    idx = y_c == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()

# there is a clear line between iris-setosa and the others


## Regression problem
# We want to predict petal length, therefor
# create new data withour petal length (remove)
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
# We need to do expand_dims to y_c for the dimensions of X_c and y_c to fit.

# remove petal length from y and X (col 3)
y_r = data[:, 2]
X_r = data[:, [0, 1, 3, 4]]

# Since the iris class information (which is now the last column in X_r) is a
# categorical variable, we will do a one-out-of-K encoding of the variable:
species = np.array(X_r[:, -1], dtype=int).T
K = species.max()+1

# 150x3 matrix with [1,0,0] for species 1 and [0,1,0] for species 2 etc.
species_encoding = np.zeros((species.size, K))
species_encoding[np.arange(species.size), species] = 1


# We need to replace the last column in X (which was the not encoded
# version of the species data) with the encoded version:
X_r = np.concatenate( (X_r[:, :-1], species_encoding), axis=1) 

# update attribute naes
targetName_r = attributeNames_c[2]
attributeNames_r = np.concatenate((attributeNames_c[[0, 1, 3]], classNames), 
                                  axis=0)

# Lastly, we update M, since we now have more attributes:
N,M = X_r.shape

# A relevant figure for this regression problem could
# for instance be one that shows how the target, that is the petal length,
# changes with one of the predictors in X:
i = 2  
plt.title('Iris regression problem')
plt.plot(X_r[:, i], y_r, 'o')
plt.xlabel(attributeNames_r[i]);
plt.ylabel(targetName_r);
# Consider if you see a relationship between the predictor variable on the
# x-axis (the variable from X) and the target variable on the y-axis (the
# variable y). Could you draw a straight line through the data points for
# any of the attributes (choose different i)? 
# Note that, when i is 3, 4, or 5, the x-axis is based on a binary 
# variable, in which case a scatter plot is not as such the best option for 
# visulizing the information. 

#%%
"""
Toy data (messy)
"""
#%% 1.5.5
# read data
file_path = './02450Toolbox_Python/Data/messy_data/messy_data.data'

# read data
messy_data = pd.read_csv(file_path, sep='\t', header=1)

# remove added header line (check messy_data.head())
messy_data.head()
messy_data = messy_data.drop(messy_data.index[0]) 

# extract attribute names
attributeNames = np.asarray(messy_data.columns)

# remove coloums with car name
car_names = np.array(messy_data.carname)
messy_data = messy_data.drop(['carname'], axis=1)

# inspect messy data
print(messy_data.to_string())

# replace ? with NaN
messy_data.displacement = messy_data.displacement.str.replace('?','NaN')

# remove ' as thousand seperator (3'840 -> 3840)
messy_data.weight = messy_data.weight.str.replace("'", '')

# replace , with .
messy_data.acceleration = messy_data.acceleration.str.replace(",", '.')

# remove 0 in he attributes 'mpg' and 'displacement'
messy_data.mpg = messy_data.mpg.replace({'0': np.nan})
messy_data.displacement = messy_data.displacement.replace({'0': np.nan})

# remove faulty information (99 in 'displacement')
messy_data.displacement = messy_data.displacement.replace({'99': np.nan})

## X,y-format
# for classification problem:
data = np.array(messy_data.values, dtype=np.float64)
X_c = data[:, :-1].copy()
y_c = data[:, -1].copy()

# regression problem
X_r = data[:, 1:].copy()
y_r = data[:, 0].copy()

# do a one-out-of-K encoding:
origin = np.array(X_r[:, -1], dtype=int).T-1
K = origin.max()+1
origin_encoding = np.zeros((origin.size, K))
origin_encoding[np.arange(origin.size), origin] = 1
X_r = np.concatenate((X_r[:, :-1], origin_encoding),axis=1)
# Since the README.txt doesn't supply a lot of information about what the
# levels in the origin variable mean, you'd have to either make an educated
# guess based on the values in the context, or preferably obtain the
# information from any papers that might be references in the README.
# In this case, you can inspect origin and car_names, to see that (north)
# american makes are all value 0 (try looking at car_names[origin == 0],
# whereas origin value 1 is European, and value 2 is Asian.

## Missing values
# In the above X,y-matrices, we still have the missing values. In the
# following we will go through how you could go about handling the missing
# values before making your X,y-matrices as above.

# Once we have identified all the missing data, we have to handle it
# some way. Various apporaches can be used, but it is important
# to keep it mind to never do any of them blindly. Keep a record of what
# you do, and consider/discuss how it might affect your modelling.

"""
Handling missing data
"""

### Method 1 - remove values
# remove missing values (NaN)
missing_idx = np.isnan(data)
# remove row_sum > 0 (rows with missing values)
obs_w_missing = np.sum(missing_idx, 1) > 0
data_drop_missing_obs = data[np.logical_not(obs_w_missing), :]
# This reduces us to 15 observations of the original 29.


### Method 2
# visualize where the missing values are
plt.title('Visual inspection of missing values')
plt.imshow(missing_idx)
plt.ylabel('Observations'); plt.xlabel('Attributes');
plt.show()

# check number os missing values in coloumn
np.sum(missing_idx, 0)

# remove displacement (col 3) with alot of missing values
cols = np.ones((data.shape[1]), dtype=bool)
cols[2] = 0
data_wo_displacement = data[:, cols] 
obs_w_missing_wo_displacement = np.sum(np.isnan(data_wo_displacement),1)>0
data_drop_disp_then_missing = data[np.logical_not(obs_w_missing_wo_displacement), :]
# Now we have kept all but two of the observations. This however, doesn't
# necesarrily mean that this approach is superior to the previous one,
# since we have now also lost any and all information that we could have
# gotten from the displacement attribute. 

### Method 3 - replace missing values with median
# do it for attribute 1 and 3
data_imputed = data.copy();
for att in [0, 2]:
     # We use nanmedian to ignore the nan values
    impute_val = np.nanmedian(data[:, att])
    idx = missing_idx[:, att]
    data_imputed[idx, att] = impute_val;

