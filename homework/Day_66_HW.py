# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:35:16 2019

@author: mingjay
"""

import keras
from keras import backend as K
from keras.layers import Layer

print(keras.__version__)

import numpy 
id(numpy.dot) == id(numpy.core.multiarray.dot)

#檢查Keras float 
K.floatx()

#設定浮點運算值
K.set_floatx('float16')
K.floatx()

from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Dense


a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

config = model.get_config()
print(config)

model.summary()