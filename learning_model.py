import h5py
from keras.utils import np_utils
from keras.models import Input, Model, Sequential
from keras.layers import Dropout, Conv2D, MaxPool2D, Flatten, Dense, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

labeldir = "/content/gdrive/My Drive/2016-01-30--11-24-51_label.h5"
datadir = "/content/gdrive/My Drive/2016-01-30--11-24-51_data.h5"
dataset = h5py.File(datadir, 'r')
labelset = h5py.File(labeldir, 'r')

n_labelset = []
n_dataset = dataset['X'][10000:20000]
for i in range(50000,100000,5):
  n_labelset.append(labelset['steering_angle'][i])

ch, row, col = 3, 160, 320  # camera format
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
          input_shape=(ch, row, col),
          output_shape=(ch, row, col)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.summary()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(n_dataset[2000:], n_labelset[2000:], validation_data=(n_dataset[:2000], n_labelset[:2000]), epochs=15, batch_size=16)
model.fit(n_dataset[2000:], n_labelset[2000:], validation_data=(n_dataset[:2000], n_labelset[:2000]), epochs=5, batch_size=100)

model.save('./20153409_1.h5')