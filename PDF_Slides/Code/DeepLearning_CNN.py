
# coding: utf-8

# In[1]:


# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
get_ipython().system('pip install --upgrade tensorflow')

# Installing Keras
#!pip install --upgrade keras

# Part 1 - Building the CNN


# In[2]:


# Importing the Keras libraries and packages
from keras.models import Sequential #initialise our neural network model as a sequential network
from keras.layers import Conv2D # convolution operation for 2D images ,3D for Videos
from keras.layers import MaxPooling2D #pooling operation
from keras.layers import Flatten #Flattening is the process of converting all the resultant 2 dimensional arrays into a single long continuous linear vector
from keras.layers import Dense


# In[ ]:


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[14]:


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

 #‘units’ is where we define the number of nodes that should be present in this hidden layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Optimizer parameter is to choose the stochastic gradient descent algorithm.
#Loss parameter is to choose the loss function.
#Finally, the metrics parameter is to choose the performance metric.


# Part 2 - Fitting the CNN to the images
#https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/CNN_Data/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/CNN_Data/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
classifier.fit_generator(training_set,
steps_per_epoch = 50,
epochs = 5,
validation_data = test_set,
validation_steps = 20)

#‘steps_per_epoch’ holds the number of training images

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/MJ_Syn/Manisha_Notes/Training/ML_Python/Code/Data/CNN_Data/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    


# In[13]:


prediction


# In[15]:


result

