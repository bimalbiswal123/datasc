
# coding: utf-8

# In[23]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML


# In[24]:


df = pd.read_excel("C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/churn2.xls")
display(df.head(5))


# In[25]:


df.describe()


# In[14]:


# Drop the columns that we have decided won't be used in prediction


features = df.drop(["Churn"], axis=1).columns

df_train, df_test = train_test_split(df, test_size=0.25)


# In[15]:


# Set up our RandomForestClassifier instance and fit to data
clf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1) # construct a forest.
clf.fit(df_train[features], df_train["Churn"])

# Make predictions
predictions = clf.predict(df_test[features])
probs = clf.predict_proba(df_test[features])
display(predictions)


# In[26]:


score = clf.score(df_test[features], df_test["Churn"])
print("Accuracy: ", score)


# In[27]:


get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_test["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_test["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[29]:


import numpy as np

fig = plt.figure(figsize=(20, 18))
ax = fig.add_subplot(111)

df_f = pd.DataFrame(clf.feature_importances_, columns=["importance"])
df_f["labels"] = features
df_f.sort_values("importance", inplace=True, ascending=False)
display(df_f.head(5))

index = np.arange(len(clf.feature_importances_))
bar_width = 0.5
rects = plt.barh(index , df_f["importance"], bar_width, alpha=0.4, color='b', label='Main')
plt.yticks(index, df_f["labels"])
plt.show()


# In[30]:


df_test["prob_true"] = probs[:, 1]
df_risky = df_test[df_test["prob_true"] > 0.9]
display(df_risky[["prob_true"]])

