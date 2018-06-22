
# coding: utf-8

# In[3]:

# Import Library

import pandas as pd


# In[4]:

# Read data from a url

url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"

df = pd.read_csv(url)


# In[5]:

# Type of the df object

type(df)


# In[6]:

# Column names

list(df)


# In[7]:

# Show first few rows

df.head()


# In[8]:

# Show last few rows

df.tail()


# In[9]:

# Data type of each column

df.dtypes


# In[10]:

# Return number of columns and rows of dataframe

df.shape


# In[11]:

#  Number of rows

len(df.index)


# In[12]:

# Number of columns

len(df.columns)


# In[13]:

# Basic statistics

df.describe()


# In[14]:

# Extract first three rows

df[0:3]

# or
    
#df.iloc[:3]


# In[15]:

# Filter for black hair

#df[df['Hair']=="Black"]

     # or
    
df.query("Hair =='Black'")


# In[16]:

# Filter for males who have black hair

#df[(df['Hair']=="Black")  & (df["Sex"]=="Male")]


# or
df.query("Hair == 'Black' & Sex =='Male'")


# In[17]:

#WAP to Filter for those who have brown eye or black hair



# In[18]:

#Ans:
z = df[(df['Hair']=="Black") | (df["Eye"]=="Brown")]


# or
z = df.query("Hair == 'Black' | Eye =='Brown'")

z.head(6)


# In[19]:

# Filter for eye color of blue, hazel and green

df[df.Eye.isin(['Blue','Hazel','Green'])].head()


# In[20]:

# Select one column

df[["Eye"]].head()

# or

df.Eye.head()


# In[21]:

# Select two columns

df[["Eye","Sex"]].head()


# In[22]:

# Unique Eye colors

df["Eye"].unique()


# In[23]:

# Maximum of the "Freq" column

df.Freq.max()


# In[24]:

# Call functions on multiple columns 

import numpy as np

pd.DataFrame({'Max_freq': [df.Freq.max()],              'Min_freq': [df.Freq.min()],             'Std_freq': [np.std(df.Freq)]})


# In[25]:

# Maximum Frequency by Sex

df.groupby("Sex").agg({"Freq":"max"})


# In[28]:

#Display max Freq by color


# In[26]:

df.groupby("Eye").agg({"Freq":"max"})


# In[27]:

# Count by Eye color and Sex

df.groupby(["Eye","Sex"]).agg({"Freq":"count"}).rename(columns={"Freq":"Count"})


# In[28]:

# Call functions for grouping

df.assign(Gt50 = (df.Freq > 50)).groupby("Gt50").agg({"Gt50":"count"}).rename(columns ={"Gt50":"Count"})


# In[29]:

# Do the analysis on selected rows only

pd.DataFrame({'Max_freq': [df[0:10].Freq.max()],              'Min_freq': [df[0:10].Freq.min()],             'Std_freq': [np.std(df[0:10].Freq)]})


# In[30]:

# Remove a column

df.drop('Unnamed: 0', 1).head()


# In[31]:

# Return the first occurance

df.query("Eye == 'Blue'")[:1]


# In[32]:

# Return the last occurance

df.query("Eye == 'Blue'")[-1:]


# In[33]:

# Return a count

df[df.Eye.isin(['Blue','Hazel']) & (df.Sex=="Male")].shape[0]


# In[34]:

# Count for each group

df[df.Eye.isin(['Blue','Hazel']) & (df.Sex=="Male")].groupby(["Eye","Sex"]).agg({"Freq":"count"}).rename(columns={"Freq":"Count"})


# In[35]:

# Order in ascending order

df.sort_values(by='Freq').tail(6)


# In[36]:

# Order in descending order

df.sort_values(by='Freq', ascending = False).tail(6)


# In[37]:

# "Freq" in descending and "Eye" in ascending

df.sort_values(by=['Freq','Eye'], ascending = [False,True]).tail(6)


# In[38]:

# Rename columns

df.rename(columns = {"Freq":"Frequency","Eye":"Eye_Color"}).tail()


# In[39]:

# Unique rows

df[["Eye","Sex"]].drop_duplicates()


# In[40]:

# Create new column

df.assign(Eye_Hair =df.Eye + df.Hair)[["Eye","Hair","Eye_Hair"]].head()


# In[ ]:



