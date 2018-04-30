#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 00:45:07 2018

@author: ajinkya-pc
"""

from keras.models import Sequential      #initiates CNN
from keras.layers import Conv2D,Convolution2D   #Does 2d convolution
from keras.layers import MaxPooling2D    #Does MAxPooling
from keras.layers import Flatten         #Does Flattening
from keras.layers import Dense 
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

a=unpickle('data_batch_1')
b=unpickle('data_batch_2')
c=unpickle('data_batch_3')
d=unpickle('data_batch_4')
e=unpickle('data_batch_5')

data_set=np.vstack((a[b'data'],b[b'data'],c[b'data'],d[b'data'],e[b'data']))
label_set=np.append(a[b'labels'],[b[b'labels'],c[b'labels'],d[b'labels'],e[b'labels']])
label_set=label_set.reshape(len(label_set),1)
dataset=list()
for i in np.arange(50000):
    p=data_set[i].reshape(32,32,3)
    dataset.append(p)
    
from sklearn.preprocessing import OneHotEncoder as ohe
ohec=ohe(categorical_features=[0]) #index of the column is to be specified for the onehot encoding
label_set=ohec.fit_transform(label_set).toarray()
classifier=Sequential()
#Step 1-Convolution1
'''Creating 32 feature detectors of six=ze 3*3'''
classifier.add(Convolution2D(64,(3,3),input_shape=(32,32,3),activation='relu')) #output of 30*30 size
'''MAx pooling with pool size of 2*2'''
classifier.add(MaxPooling2D(pool_size=(2,2)))   #output of 15*15 size
#Step3- Convolution 2
classifier.add(Convolution2D(128,(3,3),input_shape=(15,15,3),activation='relu'))  #output of 13*13
#Step 4- Maxpooling 2
classifier.add(MaxPooling2D(pool_size=(2,2)))       #output of 7*7
#Step 5- Convolution 3
classifier.add(Convolution2D(256,(3,3),input_shape=(7,7,3),activation='relu'))
#step 6-convolution 4
classifier.add(Convolution2D(256,(3,3),input_shape=(7,7,3),activation='relu'))
#step 7- Maxpooling 3
classifier.add(MaxPooling2D(pool_size=(2,2)))       #output of 7*7
classifier.add(Flatten())

classifier.add(Dense(units=1024,activation='relu'))
classifier.add(Dense(units=1024,activation='relu'))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
classifier.add(Dense(units=10,activation='softmax'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=5)
classifier.fit(np.array(dataset), np.array(label_set), epochs=20,callbacks=[early_stopping_monitor], validation_split=0.1)

