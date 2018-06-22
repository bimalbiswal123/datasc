
# coding: utf-8

# Data Set Information:
# Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. ADAP is an adaptive learning routine that generates and executes digital analogs of perceptron-like devices. It is a unique algorithm; see the paper for details.
# 
# Attribute Information:
# 1. Number of times pregnant 
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
# 3. Diastolic blood pressure (mm Hg) 
# 4. Triceps skin fold thickness (mm) 
# 5. 2-Hour serum insulin (mu U/ml) 
# 6. Body mass index (weight in kg/(height in m)^2) 
# 7. Diabetes pedigree function 
# 8. Age (years) 
# 9. Class variable (0 or 1) 

# In[1]:

# Univariate Histograms
import matplotlib.pyplot as plt
import pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
data.hist()
plt.show()


# In[6]:

# Univariate Density Plots

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()


# In[3]:

# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()


# In[7]:

# Correction Matrix Plot

import numpy

correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[8]:

# Scatterplot Matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
plt.show()


# In[ ]:



