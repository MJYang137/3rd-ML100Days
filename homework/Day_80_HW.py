# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:39:06 2019

@author: mingjay
"""

import os
import keras

train, test = keras.datasets.cifar10.load_data()

## 資料前處理
def preproc_x(x, flatten=True):
    x = x / 255.
    if flatten:
        x = x.reshape((len(x), -1))
    return x

def preproc_y(y, num_classes=10):
    if y.shape[-1] == 1:
        y = keras.utils.to_categorical(y, num_classes)
    return y

x_train, y_train = train #train is tuple where first element is an array of x_train and the second element is an array of output 
x_test, y_test = test

# Preproc the inputs
x_train = preproc_x(x_train)
x_test = preproc_x(x_test)

# Preprc the outputs
y_train = preproc_y(y_train)
y_test = preproc_y(y_test)

def build_mlp(input_shape,output_units=10, num_neurons=[512, 256, 128]):
    input_layer = keras.layers.Input(input_shape)
    
    for i,n_units in enumerate(num_neurons):
        if i == 0:
            x = keras.layers.Dense(units = n_units,activation='relu',name = 'hidden_layer' + str(i))(input_layer)
        else:
            x = keras.layers.Dense(units = n_units,activation='relu',name = 'hidden_layer' + str(i))(x)
        
    out = keras.layers.Dense(units = output_units,activation='softmax',name = 'output')(x)
    model = keras.models.Model(inputs = [input_layer],outputs=[out])

    return model


LEARNING_RATE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
EPOCHS = 5
BATCH_SIZE = 256
MOMENTUM = 0.95
optimizer = ['sgd','adam','RMSprop']

results = {}
for i,lr in enumerate (LEARNING_RATE):
    for j,opt in enumerate (optimizer):
        print("Experiment with LR = %.6f , opt = %s" % (lr,opt))
        keras.backend.clear_session()
        model = build_mlp(input_shape = x_train.shape[1:])
        #model.summary()
        model.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],optimizer = opt)
        model.fit(x_train,y_train,epochs = EPOCHS,batch_size=BATCH_SIZE,validation_data = (x_test,y_test),shuffle = 'True')
        
        train_loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        train_acc = model.history.history['accuracy']
        val_acc = model.history.history['val_accuracy']
        
        tag = "lr-%s,opt-%s" %(lr,opt)
        results[tag] = {
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'train_acc' : train_acc,
            'val_acc' : val_acc,
        }

import matplotlib.pyplot as plt
#%matplotlib inline

color_bar = ["r", "g", "b", "y", "m", "k" , "gray" , "plum" , "ivory" , "maroon" , "olive" , "skyblue" , "purple" , "lime" , "orchid" ]
plt.figure(figsize = (16,12))

for i,cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train_loss'])),results[cond]['train_loss'],'-',label = cond,color = color_bar[i])
    plt.plot(range(len(results[cond]['val_loss'])),results[cond]['val_loss'],'--',label = cond,color = color_bar[i])
plt.title('loss')
plt.legend()
plt.show()

plt.figure(figsize = (16,12))
for i,cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train_acc'])),results[cond]['train_acc'],'-',label = cond,color = color_bar[i])
    plt.plot(range(len(results[cond]['val_acc'])),results[cond]['val_acc'],'--',label = cond,color = color_bar[i])
plt.title('acc')
plt.legend()
plt.show()