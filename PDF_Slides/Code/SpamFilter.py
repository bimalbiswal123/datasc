
# coding: utf-8

# In[56]:


#Logistic regression for Spam filtering

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score


df = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/NLP_data/smsspamcollection/SMSSpamCollection',delimiter='\t',names=['smscategory','sms'])


# In[57]:


df.head()


# In[63]:


df.smscategory.value_counts()


# In[66]:


X_train_raw, X_test_raw, y_train, y_test = train_test_split(df.sms, df.smscategory)


# #Convert a collection of raw documents to a matrix of TF-IDF features
# #Apply Term Frequency Inverse Document Frequency normalization to a sparse matrix of occurrence counts.
# #Transform a count matrix to a normalized tf or tf-idf representation

# In[67]:


vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)



# Term Frequency: is a scoring of the frequency of the word in the current document.
# Inverse Document Frequency: is a scoring of how rare the word is across documents.

# In[68]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_test, y_test, cv=5)
print(np.mean(scores), scores)


# In[75]:


msg = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/NLP_data/Newdata/smsnew.txt', 
                 delimiter='\t',names=['smscategory','sms'])
msg.head()


# In[77]:


smsnew=msg.sms
newdata=vectorizer.transform(smsnew)
##smsnew.head()
type(df)


# In[78]:


predict_smscategory=classifier.predict(newdata)

predict_smscategory=pd.DataFrame(predict_smscategory)
predict_smscategory.head()


# In[81]:


predict_smscategory.rename(columns={0:'predict_smscategory'}, inplace=True)


# In[82]:


type(predict_smscategory)


# In[83]:


predict_smscategory.head()


# In[85]:


predict_smscategory.to_csv('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/NLP_data/Newdata/smscategory.txt')

