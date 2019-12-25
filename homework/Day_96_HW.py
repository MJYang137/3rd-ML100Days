# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:10:22 2019

@author: mingjay
"""

from keras.layers import Conv2D, SeparableConv2D, Input
from keras.models import Model

input_image = Input((224, 224, 3))
feature_maps = Conv2D(filters=32, kernel_size=(3,3))(input_image)
feature_maps2 = Conv2D(filters=64, kernel_size=(3,3))(feature_maps)
model = Model(inputs=input_image, outputs=feature_maps2)

model.summary()

#可以看到經過兩次 Conv2D，如果沒有設定 padding="SAME"，圖就會越來越小，同時特徵圖的 channel 數與 filters 的數量一致

input_image = Input((224, 224, 3))
feature_maps = SeparableConv2D(filters=32, kernel_size=(3,3))(input_image)
feature_maps2 = SeparableConv2D(filters=64, kernel_size=(3,3))(feature_maps)
model = Model(inputs=input_image, outputs=feature_maps2)

model.summary()

#可以看到使用 Seperable Conv2D，即使模型設置都一模一樣，但是參數量明顯減少非常多！


# Introduction
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
# import packages, mnist dataset and keras
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# set parameter as follow
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions 28*28
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 判斷圖片格式是否為channels_first，如果是就設定input_shape的第一維是channel，否則放在最後一維
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# set the format of pixel as float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalization
x_train /= 255
x_test /= 255

# print out the shape of training data & testing data
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#build CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#是否有對資料做標準化 (normalization)? 如果有，在哪幾行?  
'''
# normalization
#x_train /= 255
#x_test /= 255'''

#使用的優化器 Optimizer 為何? 
'''Adadelta
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
#模型總共疊了幾層卷積層? 
''' two
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
'''
#模型的參數量是多少? 
'''1,199,882
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
'''