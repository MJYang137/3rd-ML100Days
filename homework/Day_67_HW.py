# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:16:51 2019

@author: mingjay
"""

import numpy
from keras.datasets import cifar100
import numpy as np
np.random.seed(100)


(x_img_train,y_label_train), \
(x_img_test, y_label_test)=cifar100.load_data()


print('train:',len(x_img_train))
print('test :',len(x_img_test))

# 查詢檔案維度資訊
x_img_train.shape

# 查詢檔案維度資訊
y_label_train.shape

#針對物件圖像數據集的類別編列成字典
#
#label_dict = {}
#
#for i in range(0,101,1):
#    label_dict[str(i)] = i

#導入影像列印模組
import matplotlib.pyplot as plt

#宣告一個影像標記的函數
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    

    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(labels[i][0])
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
    
#針對不同的影像作標記

plot_images_labels_prediction(x_img_train,y_label_train,[],10)


#-----------------------------------------------------------------------------
#fig = plt.gcf()
#plt.imshow(x_img_train[0],cmap='binary');
#fig = plt.gcf()
#plt.imshow(x_img_train[1],cmap='binary');
#fig = plt.gcf() 
#plt.imshow(x_img_train[2],cmap='binary');

#-------------------------------------------------------------------------------

x_img_train[0][0][0]
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
x_img_train_normalize[0][0][0]


y_label_train.shape
y_label_train[:5]
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

y_label_train_OneHot.shape

y_label_train_OneHot[:5]