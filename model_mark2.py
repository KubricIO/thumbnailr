
# coding: utf-8

# In[1]:


import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
from keras import initializers, regularizers, applications
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import datetime
import time
from PIL import Image
import cv2


# In[2]:


img_width, img_height = 150, 150


train_data_dir = './mark_1/train'
validation_data_dir = './mark_1/val'
test_data_dir = './mark_1/test'


# In[3]:


def get_filecount(path_to_directory):
    if os.path.exists(path_to_directory):
        path, dirs, files = os.walk(path_to_directory).__next__()
        file_count = len(files)
        return file_count
    else:
        print("path does not exist")
        return 0


# In[4]:


epochs = 20
batch_size = 8


# In[6]:


nb_train_1_samples = get_filecount("mark_1/train/rate1")
nb_train_2_samples = get_filecount("mark_1/train/rate2")
nb_train_3_samples = get_filecount("mark_1/train/rate3")
nb_train_4_samples = get_filecount("mark_1/train/rate4")
nb_train_5_samples = get_filecount("mark_1/train/rate5")

# nb_train_samples = 3472

nb_train_1_samples = nb_train_1_samples - nb_train_1_samples % batch_size
nb_train_2_samples = nb_train_2_samples - nb_train_2_samples % batch_size
nb_train_3_samples = nb_train_3_samples - nb_train_3_samples % batch_size
nb_train_4_samples = nb_train_4_samples - nb_train_4_samples % batch_size
nb_train_5_samples = nb_train_5_samples - nb_train_5_samples % batch_size
nb_train_samples = nb_train_1_samples + nb_train_2_samples + nb_train_3_samples + nb_train_4_samples + nb_train_5_samples



# In[7]:


nb_val_1_samples = get_filecount("mark_1/val/rate1")
nb_val_2_samples = get_filecount("mark_1/val/rate2")
nb_val_3_samples = get_filecount("mark_1/val/rate3")
nb_val_4_samples = get_filecount("mark_1/val/rate4")
nb_val_5_samples = get_filecount("mark_1/val/rate5")

# nb_validation_samples =740

nb_val_1_samples = nb_val_1_samples - nb_val_1_samples % batch_size
nb_val_2_samples = nb_val_2_samples - nb_val_2_samples % batch_size
nb_val_3_samples = nb_val_3_samples - nb_val_3_samples % batch_size
nb_val_4_samples = nb_val_4_samples - nb_val_4_samples % batch_size
nb_val_5_samples = nb_val_5_samples - nb_val_5_samples % batch_size
nb_validation_samples = nb_val_1_samples + nb_val_2_samples + nb_val_3_samples + nb_val_4_samples + nb_val_5_samples



# In[8]:


nb_test_1 = get_filecount("mark_1/test/rate1")
nb_test_2 = get_filecount("mark_1/test/rate2")
nb_test_3 = get_filecount("mark_1/test/rate3")
nb_test_4 = get_filecount("mark_1/test/rate4")
nb_test_5 = get_filecount("mark_1/test/rate5")

nb_test_1 = nb_test_1 - nb_test_1 % batch_size
nb_test_2 = nb_test_2 - nb_test_2 % batch_size
nb_test_3 = nb_test_3 - nb_test_3 % batch_size
nb_test_4 = nb_test_4 - nb_test_4 % batch_size
nb_test_5 = nb_test_5 - nb_test_5 % batch_size
nb_test_samples = nb_test_1 + nb_test_2 + nb_test_3 + nb_test_4 + nb_test_5



# In[9]:


seed = 7
np.random.seed(seed)


# In[42]:


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                 horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[54]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)


# In[55]:


validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)


# In[56]:


model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation = 'softmax'))


# In[57]:


adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = adam


# In[58]:


model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])


# In[59]:
checkpointer = ModelCheckpoint(filepath='model_best3.h5', verbose=1, save_best_only=True)
callbacks_list = [checkpointer]




# In[60]:





# In[61]:


model.fit_generator(
    train_generator,
    epochs=40,
    steps_per_epoch=96,
    callbacks=callbacks_list,
    validation_data=validation_generator)

