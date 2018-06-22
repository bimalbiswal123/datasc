
# coding: utf-8

# In[1]:

import numpy as np
from sklearn import preprocessing


# In[2]:

#Sample data
data =  np.array([[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]])
print data


# In[7]:

#Scaling
data_scaled = preprocessing.scale(data)
print data_scaled
print data_scaled.mean(axis=0)
print data_scaled.std(axis=0)


# In[12]:

#Creating scaler instance
scaler = preprocessing.StandardScaler().fit(data)


# In[11]:

print scaler
print scaler.mean_                                      
print scaler.scale_                                       

scaler.transform(data) 


# In[13]:

scaler.transform([[-1.,  1., 0.]]) #New element


# It is possible to disable either centering or scaling by either passing with_mean=False or with_std=False to the constructor of StandardScaler.

# In[14]:

#Scaling features to a range
min_max_scaler = preprocessing.MinMaxScaler()
data_train_minmax = min_max_scaler.fit_transform(data)
data_train_minmax


# In[16]:

data_test = np.array([[ -3., -1.,  4.]]) #New instance
data_test_minmax = min_max_scaler.transform(data_test)
data_test_minmax


# MaxAbsScaler works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data. 

# In[19]:

#MaxAbsScaler 
max_abs_scaler = preprocessing.MaxAbsScaler()
data_train_maxabs = max_abs_scaler.fit_transform(data)
print data_train_maxabs                # doctest +NORMALIZE_WHITESPACE^
data_test = np.array([[ -3., -1.,  4.]])
data_test_maxabs = max_abs_scaler.transform(data_test)
print data_test_maxabs                 

max_abs_scaler.scale_         


# In[20]:

#Normalization (Dot product or Matrices)
X = [[ 1., -1.,  2.], [ 2.,  0.,  0.],[ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized 


# In[25]:

from sklearn.preprocessing import Normalizer
normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer


# In[26]:

normalizer.transform(X)


# In[29]:

#Feature binarization
X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]
print X
binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
print binarizer

binarizer.transform(X)


# In[30]:

#Adjust Threshold
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)


# In[31]:

#Imputation of missing values
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))    


# In[32]:

#Custom Transformation
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)


# In[ ]:



