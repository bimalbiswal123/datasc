
# coding: utf-8

# In[20]:

import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
print A
Af = np.array([1, 2, 3], float)
print Af


# In[21]:

np.arange(0, 1, 0.2)


# In[3]:

np.linspace(0, 2*np.pi, 4)


# In[4]:

A = np.zeros((2,3))


# In[5]:

A.shape


# In[8]:

np.random.random((2,3))


# In[9]:

a = np.random.normal(loc=1.0, scale=2.0, size=(2,2))


# In[10]:

np.savetxt("a_out.txt", a)


# In[11]:

# save to file
b = np.loadtxt("a_out.txt")


# In[14]:

A = np.zeros((2, 2))
print A


# In[16]:

a = np.arange(10).reshape((2,5))  #Array Attributes
print a.ndim # 2 dimension
print a.shape # (2, 5) shape of array
print a.size # 10 # of elements
print a.T # transpose
print a.dtype # data type


# In[18]:

#Array broadcasting with scalars
A = np.ones((3,3))
print A


# In[19]:

print 3 * A - 1


# In[ ]:



