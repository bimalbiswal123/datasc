
# coding: utf-8

# In[6]:

import pandas as pd
df = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/Camera.csv', sep=';')


# In[7]:

columns = ['Model', 'Date', 'MaxRes', 'LowRes', 'EffPix', 'ZoomW', 'ZoomT',
           'NormalFR', 'MacroFR', 'Storage', 'Weight', 'Dimensions', 'Price']
df.columns = columns
df.head()


# In[8]:

df['Maker'] = df['Model'].apply(lambda s:s.split()[0])


# In[26]:

df[['Maker','Model','Date','MaxRes','LowRes','Weight','Dimensions','Price']].head()


# In[9]:

#Sorting data: display 5 most recent models
df.sort(['Date'], ascending = False).head()


# In[10]:

#Filtering columns by value: show only models made by Nikon
df[df['Maker'] == 'Nikon']


# In[11]:

#Filtering columns by range of values: return cameras with prices above 350 and below 500 
df[(df['Price'] > 350) & (df['Price'] <= 500)]


# In[30]:

#Get statistical descriptions of the data set: find maxima, minima, averages, standard deviations, percentiles 
df[['MaxRes','LowRes','Storage','Weight','Dimensions','Price']].describe()


# In[34]:

#Plotting data frames with pandas
#Pandas comes with handy wrappers around standard matplotlib routines that allow to plot data frames very easily
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)
get_ipython().magic(u'matplotlib inline')
matplotlib.rcParams.update({'font.size': 16})
df[(df['Price'] < 500)][['Price']].hist(figsize=(8,5), bins=10, alpha=0.5)
plt.title('Histogram of camera prices/n')
plt.savefig('PricesHist.png', bbox_inches='tight')


# In[ ]:

#Grouping data with pandas


# In[58]:

gDate = df[df['Date'] > 1998].groupby('Date').mean()
dates = [str(s) for s in gDate.index]
gDate.head()


# In[61]:

gDate = df[df['Date'] > 1998].groupby('Date').mean()
dates = [str(s) for s in gDate.index]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
cols = ['b', 'r', 'g']
vars = ['EffPix', 'Weight', 'Storage']
titles = ['effective pixels', 'weight', 'storage']
for i, var in enumerate(vars):
    gDate[[var]].plot(ax=axes[i], alpha=0.5, legend=False, lw=4, c=cols[i])
    axes[i].set_xticklabels(dates, rotation=40)
    axes[i].set_title('Evolution of %s/n' % titles[i])
plt.savefig('CameraEvolution.png', bbox_inches='tight')


# In[62]:

gMak = df.groupby('Maker').median()
gMak.index.name = ''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
c = ['y','c']
vars = ['Dimensions', 'Price']
for i, var in enumerate(vars):
    gMak[[var]].plot(kind='barh', ax=axes[i], alpha=0.5, legend=False, color=c[i])
    axes[i].set_title('Average %s by maker/n' % vars[i])
    axes[i].grid(False)
plt.savefig('MeanDimensionsPrices.png', bbox_inches='tight')


# In[63]:

###Example 2
#Data Structures
#pandas introduces two new data structures to Python - Series and DataFrame


# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)
get_ipython().magic(u'matplotlib inline')


# In[4]:

#Series(one dimensional object,index from 0 to N)
# create a Series with an arbitrary list
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'])


# In[5]:

s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],  #specifying Index
              index=['A', 'Z', 'C', 'Y', 'E'])
s


# In[9]:

#dictionaries to series
d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,
     'Austin': 450, 'Boston': None}
cities = pd.Series(d)
cities


# In[70]:

cities['Chicago']


# In[7]:

cities[['Chicago', 'Portland', 'San Francisco']]


# In[72]:

cities[cities < 1000] #boolean indexing


# In[8]:

less_than_1000 = cities < 1000
print(less_than_1000)
print('/n')
print(cities[less_than_1000])


# In[10]:

#change the values in a Series 
# changing based on the index
print('Old value:', cities['Chicago'])
cities['Chicago'] = 1400
print('New value:', cities['Chicago'])


# In[11]:

# changing values using boolean logic
print(cities[cities < 1000])
print('/n')
cities[cities < 1000] = 750

print cities[cities < 1000]


# In[12]:

#idiomatic Python.
print('Seattle' in cities)
print('San Francisco' in cities)


# In[13]:

#Mathematical operations can be done using scalars and functions.
# divide city values by 3
cities / 3


# In[14]:

# square city values
np.square(cities)


# In[15]:

print(cities[['Chicago', 'New York', 'Portland']])
print('/n')
print(cities[['Austin', 'New York']])
print('/n')
print(cities[['Chicago', 'New York', 'Portland']] + cities[['Austin', 'New York']])


# In[16]:

# returns a boolean series indicating which values aren't NULL
cities.notnull()


# In[17]:

# use boolean logic to grab the NULL cities
print(cities.isnull())
print('/n')
print(cities[cities.isnull()])


# In[18]:

#DataFrame
#A DataFrame is a tablular data structure comprised of rows and columns, akin to a spreadsheet, database table, or R's data.frame object. You can also think of a DataFrame as a group of Series objects that share an index (the column names).
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions', 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
football


# In[20]:

get_ipython().magic(u'cd ~/C:/MJ_Syn/Manisha_Notes/Training/Python/Data')


# In[21]:

from_csv = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/Python/Data/mariano-rivera.csv')
from_csv.head()


# In[22]:

#Excel(require xlrd library)
# this is the DataFrame we created from a dictionary earlier
football.head()


# In[23]:

# since our index on the football DataFrame is meaningless, let's not write it
football.to_excel('football.xlsx', index=False)


# In[24]:

get_ipython().system(u'ls -l *.xlsx')


# In[31]:

import sys
import os
import subprocess
get_ipython().system(u'ls -l *.xlsx')


# In[33]:

# delete the DataFrame
#del football


# In[34]:

# read from Excel
football = pd.read_excel('football.xlsx', 'Sheet1')
football


# In[ ]:

#Database
from pandas.io import sql
import sqlite3

conn = sqlite3.connect('/Users/gjreda/Dropbox/gregreda.com/_code/towed')
query = "SELECT * FROM towed WHERE make = 'FORD';"

results = sql.read_sql(query, con=conn)
results.head()


# In[35]:

url = 'https://raw.github.com/gjreda/best-sandwiches/master/data/best-sandwiches-geocode.tsv'

# fetch the text from the URL and read it into a DataFrame
from_url = pd.read_table(url, sep='/t')
from_url.head(3)


# In[39]:

#Dataframe -Data Exploration
# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/Python/Data/ml_100k/u_user.csv', sep='|', names=u_cols,
                    encoding='latin-1')


# In[40]:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/Python/Data/ml_100k/u.data', sep='/t', names=r_cols,
                      encoding='latin-1')




# In[41]:

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('C:/MJ_Syn/Manisha_Notes/Training/Python/Data/ml_100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')


# In[42]:

#inspection 
movies.info()


# In[43]:

movies.dtypes


# In[44]:

users.describe()


# In[45]:

movies.head()


# In[46]:

movies.tail(3)


# In[47]:

#index slicing
movies[20:22]


# In[51]:

users['occupation'].head() #selection
#users['occupation'].unique()


# In[49]:

print(users[['age', 'zip_code']].head())
print('/n')

# can also store in a variable to use later
columns_you_want = ['occupation', 'sex'] 
print(users[columns_you_want].head())


# In[52]:

# users older than 25
print(users[users.age > 25].head(3))
print('/n')

# users aged 40 AND male
print(users[(users.age == 40) & (users.sex == 'M')].head(3))
print('/n')

# users younger than 30 OR female
print(users[(users.sex == 'F') | (users.age < 30)].head(3))


# In[53]:

print(users.set_index('user_id').head())
print('/n')


# In[54]:

print(users.head())
print("/n^^^ I didn't actually change the DataFrame. ^^^/n")


# In[55]:

with_new_index = users.set_index('user_id')
print(with_new_index.head())
print("/n^^^ set_index actually returns a new DataFrame. ^^^/n")


# In[56]:

users.set_index('user_id', inplace=True)#less efficient
users.head()


# In[58]:

print(users.iloc[99]) #selection by location
print('/n')
print(users.iloc[[1, 50, 300]])


# In[59]:

#selection by label rows
print(users.loc[100])
print('/n')
print(users.loc[[2, 51, 301]])


# In[60]:

users.reset_index(inplace=True)
users.head()


# In[61]:

#merge/join datasets 
left_frame = pd.DataFrame({'key': range(5), 
                           'left_value': ['a', 'b', 'c', 'd', 'e']})
right_frame = pd.DataFrame({'key': range(2, 7), 
                           'right_value': ['f', 'g', 'h', 'i', 'j']})
print(left_frame)
print('/n')
print(right_frame)


# In[62]:

pd.merge(left_frame, right_frame, on='key', how='inner')


# In[64]:

#if diff key names
#pd.merge(left_frame, right_frame, left_on='left_key', right_on='right_key')


# In[65]:

#left outer join
pd.merge(left_frame, right_frame, on='key', how='left')


# In[66]:

#right outer join
pd.merge(left_frame, right_frame, on='key', how='right')


# In[67]:

#full outer join
pd.merge(left_frame, right_frame, on='key', how='outer')


# In[68]:

#Combining
pd.concat([left_frame, right_frame])


# In[69]:

pd.concat([left_frame, right_frame], axis=1) #using axis parameter


# In[70]:

# create one merged DataFrame for movies data
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)


# In[71]:

#What are the 25 most rated movies?
most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
most_rated


# In[72]:

lens.title.value_counts()[:25] #another method


# In[73]:

#Which movies are most highly rated?
movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
movie_stats.head()


# In[74]:

# sort by rating average
movie_stats.sort_values([('rating', 'mean')], ascending=False).head()


# In[75]:

atleast_100 = movie_stats['rating']['size'] >= 100
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]


# In[84]:

most_50 = lens.groupby('movie_id').size().sort_values(ascending=False)[:50]#contraversial movies


# In[85]:

#Which movies are most controversial amongst different ages?
users.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age');


# In[86]:

#binning users
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
lens[['age', 'age_group']].drop_duplicates()[:10]


# In[87]:

lens.groupby('age_group').agg({'rating': [np.size, np.mean]})


# In[88]:

lens.set_index('movie_id', inplace=True)
by_age = lens.loc[most_50.index].groupby(['title', 'age_group'])
by_age.rating.mean().head(15)


# In[90]:

by_age.rating.mean().unstack(1).fillna(0)[10:20] #unstacking


# In[91]:

by_age.rating.mean().unstack(0).fillna(0)


# In[92]:

#Which movies do men and women most disagree on?
lens.reset_index('movie_id', inplace=True)
pivoted = lens.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
pivoted.head()


# In[93]:

pivoted['diff'] = pivoted.M - pivoted.F
pivoted.head()


# In[94]:

pivoted.reset_index('movie_id', inplace=True)
disagreements = pivoted[pivoted.movie_id.isin(most_50.index)]['diff']
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings/n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference');


# In[ ]:



