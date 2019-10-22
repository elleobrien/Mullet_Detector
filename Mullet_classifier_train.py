#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:36:59 2019

This model uses transfer learning from VGG16 to learn to classify mullets.
Architecture: Use VGG16 to generate feature weights for each image. Then train two feedforward, fully connected layers to categorize
as the VGG16 features as mullet or no.

@author: eobrien
"""


from keras.applications import VGG16
from keras import models, optimizers
from keras import layers
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
# Base variables
base_dir = '.'
pic_dir = os.path.join(base_dir, 'data')


img_width, img_height = 224,224 # Default input size for VGG16

# Load in the convolutional base
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))  

# How many files are there in the model directory?
n_files = len(os.listdir(train_dir + "/Mullet")) + len(os.listdir(train_dir + "/No_Mullet"))
batch_size = 32
train_size = int(np.floor((n_files)/batch_size)) * batch_size # How many fit nicely in batches

### Use this base to extract features from our input images ####

# Dataset augmentation
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=10, # Some gentle tilt allowed
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Define a function to extract features from a given image.
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='binary')
    # Pass each batch of images through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i +=1 
        if i * batch_size >= sample_count:
            break
    return features, labels
    
# Run our function on our dataset
features, labels = extract_features(pic_dir, train_size)  
# Split into train and test with 10% held out for validation
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.1,
                                                    stratify = labels)


# Define feedforward model model
epochs = 10 
model = models.Sequential()
model.add(layers.Flatten(input_shape=(7,7,512)))
model.add(layers.Dense(512,activation='relu', input_dim=(7*7*512)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size, 
                    validation_data=(X_test, Y_test),
                    #class_weight=class_weights, DON'T NEED, BALANCED CLASSES
                    verbose = 1)
model.save("mullet_classifier.h5")
