
# coding: utf-8

# In[15]:


from sklearn import cross_validation
#from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import numpy as np
import pylab as pl

#Boston Data Set Information:
#Concerns housing values in suburbs of Boston.

Attribute Information:
1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per $10,000 
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. MEDV: Median value of owner-occupied homes in $1000's
    
 
# In[3]:



from sklearn.datasets import load_boston
boston = load_boston()


# In[14]:


boston.feature_names   #Unicode string


# In[16]:


print(boston.data.shape)
print(boston.target.shape)


# In[20]:


boston.target


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(boston.target)

#counts, bins, bars = plt.hist(boston.target)


# In[7]:


np.set_printoptions(precision=2, linewidth=120,suppress=True, edgeitems=4)


# In[8]:


# In order to do multiple regression we need to add a column of 1s for x0

x =boston.data
y = boston.target


# In[9]:



# First 10 elements of the data
print(x[:10])


# In[10]:


# First 10 elements of the response variable
print (y[:10])


# In[21]:


#comparisons across methods
a = 0.3
for name,method in [
        ('linear regression', LinearRegression()),
        ('lasso', Lasso(fit_intercept=True, alpha=a)),
        ('ridge', Ridge(fit_intercept=True, alpha=a)),
        ('elastic-net', ElasticNet(fit_intercept=True,alpha=a))
        ]:
    
     
    cv = cross_validation.KFold(len(x), n_folds=10)
     
    err = 0
    for train,test in cv:
        method.fit(x[train],y[train])
        rsq=method.score(x[train],y[train])
        #print('RSquare : %.4f' %rsq)
        py = method.predict(x[test])
        e = py-y[test]
        err += np.dot(e,e)

    rmse_10cv = np.sqrt(err/len(x[test]))
    rsq_10cv=np.average(rsq)
    
    
    print('Method: %s' %name)
    
    print('RSquare 10fold avg  : %.4f' %rsq_10cv)

    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    print('Coefficients:n',method.coef_)
    print('Intercept:n',method.intercept_)
    print ("\n")
   
    # print('R-Sqaure(Data):%.4f'%met.score(x,y))
    #print('RMSE on training: %.4f' %rmse_train)
    #print('RSquare : %.4f' %rsq)


