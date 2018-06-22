
# coding: utf-8

# In[1]:

#Using Python as a Calculator
2+2


# In[2]:

(50-5*6)/4


# In[3]:

7/3


# In[4]:

7/3.


# In[5]:

from __future__ import division
7/3


# In[6]:


7/float(3)


# In[7]:

sqrt(81)


# In[8]:

from math import sqrt
sqrt(81)


# In[9]:

import math
math.sqrt(81)


# In[10]:

#Define Variables 
width = 20
length = 30
area = length*width
area


# In[11]:


volume #Undefined variable


# In[12]:

depth = 10
volume = area*depth
volume


# In[14]:

#Reserved words
#and, as, assert, break, class, continue, def, del, elif, else, except, 
#exec, finally, for, from, global, if, import, in, is, lambda, not, or,
#pass, print, raise, return, try, while, with, yield

return=0


# In[13]:

#Python Data types
'Hello, World!'


# In[14]:

"Hello, World!"


# In[15]:

"He's a Rebel"  #Not both on same time unless specifically required


# In[16]:

'She asked, "How are you today?"'


# In[17]:

greeting = "Hello, World!" #Object assignment


# In[18]:


print greeting


# In[19]:

print "The area is ",area


# In[20]:

statement = "Hello," + "World!"  #String concatination
print statement


# In[23]:

#Don't forget the space between the strings, if you want one there.
statement = "Hello, " + "World!"
print statement


# In[21]:

#Multiple strings
print "This " + "is " + "a " + "longer " + "statement."


# In[22]:

#String Methods

len() 
lower()
upper()
str()


# In[23]:

#Lists (Grouping similar items together)
days_of_the_week = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]


# In[24]:

#Access members of the list using the index of that item:

days_of_the_week[2]


# In[25]:

#Negative index
days_of_the_week[-1]


# In[26]:

#List Append
languages = ["Fortran","C","C++"]
languages.append("Python")
print languages


# In[27]:

#sequential list of numbers
range(10)


# In[28]:

range(2,8)


# In[29]:

evens = range(0,20,2)
evens


# In[30]:

evens[3]


# In[31]:


["Today",7,99.3,""]


# In[34]:

#When to use lists and when to use tuples


# In[32]:

help(len)


# In[36]:

len(evens)


# In[33]:

integer_list = [1, 2, 3]

heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list, [] ]
print list_of_lists

list_length = len(integer_list) 
print list_length
list_sum = sum(integer_list) 
print list_sum


# In[35]:

# You can get or set the nth element of a list with square brackets:
x = range(10) 
print x
zero = x[0] 
one = x[1] 
nine = x[-1] 
eight = x[-2] 
x[0] = -1 


# In[36]:

#List slicing
print x
first_three = x[:3] 
three_to_end = x[3:] 
one_to_four = x[1:5] 
last_three = x[-3:] 
without_first_and_last = x[1:-1]
copy_of_x = x[:] 


# In[37]:

# Python has an in operator to check for list membership:

1 in [1, 2, 3] # True


# In[38]:

0 in [1, 2, 3] # False


# In[39]:

# It is easy to concatenate lists together:
x = [1, 2, 3]
x.extend([4, 5, 6]) # x is now [1,2,3,4,5,6]
print x

# If you don’t want to modify x you can use list addition:
x = [1, 2, 3]
y = x + [4, 5, 6] # y is [1, 2, 3, 4, 5, 6]; x is unchanged
print y
# More frequently we will append to lists one item at a time:
x = [1, 2, 3]
x.append(0) # x is now [1, 2, 3, 0]
print x


# In[53]:

#List unpacking
x, y = [1, 2] # now x is 1, y is 2
print x,y


# In[41]:

# Tuples
# Tuples are lists’ immutable cousins.
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3 
print my_list


# In[43]:

# try:
my_tuple[1] = 3


# In[44]:

# Tuples are a convenient way to return multiple values from functions:
def sum_and_product(x, y):
    return (x + y),(x * y)
sp = sum_and_product(2, 3) # equals (5, 6)
s, p = sum_and_product(5, 10) # s is 15, p is 50


print("sumnproduct {0}, sum {1} ,product {2}".format(sp,s,p)) 




# In[45]:

# Tuples (and lists) can also be used for multiple assignment:
x, y = 1, 2 # now x is 1, y is 2
x, y = y, x # Pythonic way to swap variables; now x is 2, y is 1


# In[68]:

# Dictionaries
# Another fundamental data structure is a dictionary, which associates values with keys
# and allows you to quickly retrieve the value corresponding to a given key:


# In[46]:

empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95 } # dictionary literal
print grades
# You can look up the value for a key using square brackets:
joels_grade = grades["Joel"] # equals 80.
print joels_grade


# In[47]:

kates_grade = grades["Kate"] #Key


# In[76]:

# Dictionaries have a get method that returns a default value 
joels_grade = grades.get("Joel", 0) 
print joels_grade
kates_grade = grades.get("Kate", 0) 
print kates_grade
no_ones_grade = grades.get("No One") # default default is None
print no_ones_grade


# In[48]:

# You assign key-value pairs using the same square brackets:
grades["Tim"] = 99 # replaces the old value
grades["Kate"] = 100 # adds a third entry
num_students = len(grades) # equals 3


# In[49]:

# We will frequently use dictionaries as a simple way to represent structured data:
tweet = {
"user" : "joelgrus",
"text" : "Data Science is Awesome",
"retweet_count" : 100,
"hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}
# Besides looking for specific keys we can look at all of them:
tweet_keys = tweet.keys() # list of keys
tweet_values = tweet.values() # list of values
tweet_items = tweet.items() # list of (key, value) tuples
"user" in tweet_keys # True, but uses a slow list in
"user" in tweet # more Pythonic, uses faster dict in
"joelgrus" in tweet_values # True
# Dictionary keys must be immutable; in particular, you cannot use lists as keys. If
# you need a multipart key, you should use a tuple or figure out a way to turn the key
# into a string.


# In[50]:

#Iteration, Indentation, and Blocks

days_of_the_week = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
for day in days_of_the_week:
    print day


# In[51]:

for day in days_of_the_week:
    statement = "Today is " + day
    print statement


# In[52]:

for i in range(20):
    print "The square of ",i," is ",i*i


# In[53]:

#Slicing
for letter in "Sunday":
    print letter


# In[54]:


days_of_the_week[0]


# In[55]:

days_of_the_week[0:2]


# In[56]:

days_of_the_week[:2]


# In[57]:

days_of_the_week[-2:]


# In[ ]:



